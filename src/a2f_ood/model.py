"""Atom-aligned, reliability-gated graph/SMILES fusion in pure PyTorch.

Property predictions never depend on ``target_ids``. Teacher-forced targets
are used exclusively by the optional graph-to-SMILES decoder. Reliability
features are disagreement proxies rather than calibrated uncertainties.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim) / weights.sum(dim).clamp_min(1)


def _mlp(in_dim: int, out_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class GINELayer(nn.Module):
    """Bond-aware GIN with sum aggregation over actual edges only."""

    def __init__(self, hidden_dim: int, bond_dim: int, dropout: float) -> None:
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(()))
        self.edge_encoder = nn.Linear(bond_dim, hidden_dim)
        self.update = _mlp(hidden_dim, hidden_dim, hidden_dim * 2, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: Tensor, bonds: Tensor, adjacency: Tensor, mask: Tensor) -> Tensor:
        batch_size, num_atoms, hidden_dim = nodes.shape
        edge_mask = adjacency.bool() & mask[:, :, None] & mask[:, None, :]
        batch_idx, destination, source = edge_mask.nonzero(as_tuple=True)
        aggregate = nodes.new_zeros((batch_size * num_atoms, hidden_dim))
        if batch_idx.numel():
            edge_embedding = self.edge_encoder(bonds[batch_idx, destination, source])
            messages = F.relu(nodes[batch_idx, source] + edge_embedding)
            aggregate.index_add_(0, batch_idx * num_atoms + destination, messages)
        aggregate = aggregate.reshape(batch_size, num_atoms, hidden_dim)
        update = self.update((1 + self.eps) * nodes + aggregate)
        result = self.norm(nodes + self.dropout(update))
        return result * mask.unsqueeze(-1)


class MoleculeGraphEncoder(nn.Module):
    def __init__(
        self, atom_dim: int, bond_dim: int, hidden_dim: int, layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(atom_dim, hidden_dim)
        self.layers = nn.ModuleList(
            GINELayer(hidden_dim, bond_dim, dropout) for _ in range(layers)
        )
        self.readout = _mlp(hidden_dim, hidden_dim, hidden_dim, dropout)

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        mask = batch["atom_mask"].bool()
        nodes = self.input_projection(batch["atom_features"]) * mask.unsqueeze(-1)
        for layer in self.layers:
            nodes = layer(nodes, batch["bond_features"], batch["adjacency"], mask)
        return nodes, self.readout(masked_mean(nodes, mask, dim=1))


class SmilesEncoder(nn.Module):
    def __init__(
        self, vocab_size: int, hidden_dim: int, layers: int, heads: int,
        dropout: float, max_length: int, pad_id: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.position = nn.Embedding(max_length, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, token_ids: Tensor, token_mask: Tensor) -> tuple[Tensor, Tensor]:
        length = token_ids.size(1)
        if length > self.position.num_embeddings:
            raise ValueError(
                f"SMILES sequence length {length} exceeds positional capacity "
                f"{self.position.num_embeddings}; increase data.max_smiles_length"
            )
        if not token_mask.bool().any(dim=1).all():
            raise ValueError("Every SMILES view must contain an unmasked token")
        positions = torch.arange(length, device=token_ids.device)
        hidden = self.dropout(self.embedding(token_ids) + self.position(positions)[None])
        hidden = self.norm(self.transformer(hidden, src_key_padding_mask=~token_mask.bool()))
        hidden = hidden * token_mask.unsqueeze(-1)
        return hidden, hidden[:, 0]


class AtomTokenAligner(nn.Module):
    """Gather atom-token correspondences; graph padding maps to token zero."""

    def forward(self, tokens: Tensor, positions: Tensor, atom_mask: Tensor) -> Tensor:
        valid_positions = positions[atom_mask]
        if valid_positions.numel() and (
            bool((valid_positions < 0).any()) or bool((valid_positions >= tokens.size(1)).any())
        ):
            raise ValueError("An atom token position is outside the SMILES sequence")
        gathered = tokens.gather(1, positions.clamp_min(0).unsqueeze(-1).expand(-1, -1, tokens.size(-1)))
        return gathered * atom_mask.unsqueeze(-1)


class GraphSmilesDecoder(nn.Module):
    """Autoregressive decoder conditioned only on encoded graph atoms."""

    def __init__(
        self, vocab_size: int, hidden_dim: int, heads: int, layers: int,
        dropout: float, max_length: int, pad_id: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.position = nn.Embedding(max_length, hidden_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.pad_id = pad_id

    def forward(self, input_ids: Tensor, memory: Tensor, atom_mask: Tensor) -> Tensor:
        length = input_ids.size(1)
        if length > self.position.num_embeddings:
            raise ValueError("Decoder sequence exceeds configured maximum length")
        positions = torch.arange(length, device=input_ids.device)
        target = self.embedding(input_ids) + self.position(positions)[None]
        causal_mask = torch.ones(length, length, dtype=torch.bool, device=input_ids.device).triu(1)
        hidden = self.decoder(
            target, memory, tgt_mask=causal_mask,
            tgt_key_padding_mask=input_ids.eq(self.pad_id),
            memory_key_padding_mask=~atom_mask.bool(),
        )
        return self.output(hidden)


class A2FModel(nn.Module):
    """Dual-view fusion model with atom alignment and optional SMILES generation."""

    def __init__(
        self, config: dict[str, Any], vocab_size: int, atom_feature_dim: int,
        bond_feature_dim: int, pad_id: int = 0, bos_id: int = 3, eos_id: int = 4,
    ) -> None:
        super().__init__()
        self.config = config
        model_config, data_config = config["model"], config["data"]
        self.fusion_config = config.get("fusion", {})
        self.mode = self.fusion_config.get("mode", "adaptive")
        if self.mode not in {"adaptive", "concat", "graph", "smiles"}:
            raise ValueError("fusion.mode must be adaptive, concat, graph, or smiles")
        hidden = int(model_config.get("hidden_dim", 256))
        heads = int(model_config.get("attention_heads", 4))
        dropout = float(model_config.get("dropout", 0.1))
        max_length = int(data_config.get("max_smiles_length", 256)) + 2
        if data_config["task_type"] == "multiclass":
            num_outputs = int(data_config["num_classes"])
        else:
            num_outputs = len(data_config["label_columns"])
        self.hidden_dim, self.num_outputs = hidden, num_outputs
        self.pad_id, self.bos_id, self.eos_id = pad_id, bos_id, eos_id
        self.graph_encoder = MoleculeGraphEncoder(
            atom_feature_dim, bond_feature_dim, hidden,
            int(model_config.get("graph_layers", 4)), dropout,
        )
        self.smiles_encoder = SmilesEncoder(
            vocab_size, hidden, int(model_config.get("smiles_layers", 4)), heads,
            dropout, max_length, pad_id,
        )
        self.aligner = AtomTokenAligner()
        self.graph_atom_projection = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.smiles_atom_projection = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.shared = _mlp(hidden * 3, hidden, hidden * 2, dropout)
        self.private_graph = nn.Linear(hidden, hidden)
        self.private_smiles = nn.Linear(hidden, hidden)
        self.atom_gate_network = _mlp(hidden * 2 + 2, 2, hidden, dropout)
        self.local_norm = nn.LayerNorm(hidden)
        self.graph_to_smiles = _mlp(hidden, hidden, hidden * 2, dropout)
        self.smiles_to_graph = _mlp(hidden, hidden, hidden * 2, dropout)
        self.global_gate = _mlp(hidden * 2 + 2, 2, hidden, dropout)
        self.graph_queries_smiles = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.smiles_queries_graph = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.cross_graph_norm = nn.LayerNorm(hidden)
        self.cross_smiles_norm = nn.LayerNorm(hidden)
        self.adaptive_fusion = _mlp(hidden * 4, hidden, hidden * 2, dropout)
        self.concat_fusion = _mlp(hidden * 2, hidden, hidden * 2, dropout)
        self.predictor = _mlp(hidden, num_outputs, hidden, dropout)
        self.graph_head = _mlp(hidden, num_outputs, hidden, dropout)
        self.smiles_head = _mlp(hidden, num_outputs, hidden, dropout)
        self.decoder = None
        if model_config.get("use_cross_modal_decoder", False):
            self.decoder = GraphSmilesDecoder(
                vocab_size, hidden, heads, int(model_config.get("decoder_layers", 2)),
                dropout, max_length, pad_id,
            )

    def forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        graph_nodes, graph_embedding = self.graph_encoder(batch)
        batch_size, views, sequence_length = batch["token_ids"].shape
        atoms, hidden = graph_nodes.size(1), graph_nodes.size(2)
        token_ids = batch["token_ids"].reshape(batch_size * views, sequence_length)
        token_mask = batch["token_mask"].reshape(batch_size * views, sequence_length).bool()
        smiles_tokens, smiles_embedding = self.smiles_encoder(token_ids, token_mask)
        atom_mask = batch["atom_mask"].bool()
        flat_atom_mask = atom_mask[:, None].expand(-1, views, -1).reshape(batch_size * views, atoms)
        token_positions = batch["atom_token_positions"].reshape(batch_size * views, atoms)
        smiles_atoms = self.aligner(smiles_tokens, token_positions, flat_atom_mask)
        graph_atoms = self.graph_atom_projection(graph_nodes) * atom_mask.unsqueeze(-1)
        smiles_atoms = self.smiles_atom_projection(smiles_atoms) * flat_atom_mask.unsqueeze(-1)
        expanded_graph_atoms = graph_atoms[:, None].expand(-1, views, -1, -1).reshape(-1, atoms, hidden)
        expanded_graph_embedding = graph_embedding[:, None].expand(-1, views, -1).reshape(-1, hidden)

        predicted_smiles = self.graph_to_smiles(graph_embedding)
        predicted_graph = self.smiles_to_graph(smiles_embedding)
        graph_error = (
            F.normalize(predicted_graph, dim=-1) - F.normalize(expanded_graph_embedding.detach(), dim=-1)
        ).square().mean(-1)
        smiles_error = (
            F.normalize(predicted_smiles[:, None].expand(-1, views, -1).reshape(-1, hidden), dim=-1)
            - F.normalize(smiles_embedding.detach(), dim=-1)
        ).square().mean(-1)
        errors = torch.stack([graph_error, smiles_error], dim=-1)

        difference = (expanded_graph_atoms - smiles_atoms).abs()
        shared = self.shared(torch.cat([
            expanded_graph_atoms + smiles_atoms,
            expanded_graph_atoms * smiles_atoms, difference,
        ], dim=-1))
        graph_private = expanded_graph_atoms - self.private_graph(shared)
        smiles_private = smiles_atoms - self.private_smiles(shared)
        if self.fusion_config.get("reliability_gate", True):
            reliability = errors.detach()[:, None].expand(-1, atoms, -1)
            atom_gate = self.atom_gate_network(torch.cat([shared, difference, reliability], dim=-1)).softmax(-1)
        else:
            atom_gate = shared.new_full((*shared.shape[:-1], 2), 0.5)
        local = self.local_norm(
            shared + atom_gate[..., :1] * graph_private + atom_gate[..., 1:] * smiles_private
        )
        if not self.fusion_config.get("atom_alignment", True):
            local = expanded_graph_atoms
        local = local * flat_atom_mask.unsqueeze(-1)
        local_pool = masked_mean(local, flat_atom_mask, dim=1)

        if self.fusion_config.get("bidirectional_cross_attention", True):
            graph_cross, _ = self.graph_queries_smiles(
                expanded_graph_atoms, smiles_tokens, smiles_tokens,
                key_padding_mask=~token_mask, need_weights=False,
            )
            smiles_cross, _ = self.smiles_queries_graph(
                smiles_tokens, expanded_graph_atoms, expanded_graph_atoms,
                key_padding_mask=~flat_atom_mask, need_weights=False,
            )
            graph_cross = self.cross_graph_norm(expanded_graph_atoms + graph_cross)
            smiles_cross = self.cross_smiles_norm(smiles_tokens + smiles_cross)
            graph_context = masked_mean(graph_cross, flat_atom_mask, dim=1)
            smiles_context = masked_mean(smiles_cross, token_mask, dim=1)
        else:
            graph_context = expanded_graph_embedding
            smiles_context = smiles_embedding

        if self.fusion_config.get("reliability_gate", True):
            global_weights = self.global_gate(torch.cat([
                expanded_graph_embedding, smiles_embedding, errors.detach(),
            ], dim=-1)).softmax(-1)
        else:
            global_weights = graph_embedding.new_full((batch_size * views, 2), 0.5)
        global_context = (
            global_weights[:, :1] * expanded_graph_embedding
            + global_weights[:, 1:] * smiles_embedding
        )
        if self.mode == "adaptive":
            fused = self.adaptive_fusion(torch.cat([
                local_pool, graph_context, smiles_context, global_context,
            ], dim=-1))
        elif self.mode == "concat":
            fused = self.concat_fusion(torch.cat([expanded_graph_embedding, smiles_embedding], dim=-1))
        elif self.mode == "graph":
            fused = expanded_graph_embedding
        else:
            fused = smiles_embedding

        output = {
            "logits": self.predictor(fused).reshape(batch_size, views, -1),
            "graph_logits": self.graph_head(graph_embedding),
            "smiles_logits": self.smiles_head(smiles_embedding).reshape(batch_size, views, -1),
            "graph_embedding": graph_embedding,
            "smiles_embedding": smiles_embedding.reshape(batch_size, views, hidden),
            "fused_embedding": fused.reshape(batch_size, views, hidden),
            "graph_atoms": graph_atoms,
            "smiles_atoms": smiles_atoms.reshape(batch_size, views, atoms, hidden),
            "predicted_smiles_embedding": predicted_smiles,
            "predicted_graph_embedding": predicted_graph.reshape(batch_size, views, hidden),
            "cross_modal_error": errors.reshape(batch_size, views, 2),
            "atom_gate": atom_gate.reshape(batch_size, views, atoms, 2) * atom_mask[:, None, :, None],
            "global_gate": global_weights.reshape(batch_size, views, 2),
        }
        if self.decoder is not None and "target_ids" in batch:
            target_ids = batch["target_ids"]
            if target_ids.size(1) < 2:
                raise ValueError("Decoder targets must contain at least BOS and EOS")
            output["generation_logits"] = self.decoder(target_ids[:, :-1], graph_nodes, atom_mask)
        return output

    @torch.no_grad()
    def generate_smiles(self, batch: dict[str, Any], max_length: int | None = None) -> Tensor:
        """Greedy graph-only decoding; call eval() first to disable dropout."""
        if self.decoder is None:
            raise RuntimeError("Graph-to-SMILES decoder is disabled")
        memory, _ = self.graph_encoder(batch)
        maximum = self.decoder.position.num_embeddings
        max_length = maximum if max_length is None else int(max_length)
        if max_length < 2 or max_length > maximum:
            raise ValueError(f"max_length must be between 2 and {maximum}")
        output = torch.full(
            (memory.size(0), 1), self.bos_id, device=memory.device, dtype=torch.long,
        )
        finished = torch.zeros(memory.size(0), device=memory.device, dtype=torch.bool)
        for _ in range(max_length - 1):
            logits = self.decoder(output, memory, batch["atom_mask"])
            logits[:, -1, self.pad_id] = -torch.inf
            logits[:, -1, self.bos_id] = -torch.inf
            # Vocabulary reserves 1=UNK and 2=CLS. Neither is a chemical token.
            logits[:, -1, 1] = -torch.inf
            logits[:, -1, 2] = -torch.inf
            next_token = logits[:, -1].argmax(-1)
            next_token = torch.where(finished, next_token.new_full((), self.pad_id), next_token)
            output = torch.cat([output, next_token[:, None]], dim=1)
            finished |= next_token.eq(self.eos_id)
            if bool(finished.all()):
                break
        return output
