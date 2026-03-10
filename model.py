import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class S2SMaskGITConfig:
    num_semantic_types: int = 9116
    num_acoustic_types: int = 65536
    d_model: int = 512
    num_layers: int = 8
    dropout: float = 0.1


class S2SMaskGIT(nn.Module):

    def __init__(self, cfg: S2SMaskGITConfig = S2SMaskGITConfig()):
        super().__init__()
        self.cfg = cfg

        assert cfg.d_model % 64 == 0
        self.nhead = cfg.d_model // 64

        self.semantic_pad_idx = cfg.num_semantic_types
        self.semantic_vocab_size = cfg.num_semantic_types + 1

        self.acoustic_mask_idx = cfg.num_acoustic_types
        self.acoustic_pad_idx = cfg.num_acoustic_types + 1
        self.acoustic_vocab_size = cfg.num_acoustic_types + 2

        self.semantic_embedder = nn.Embedding(
            num_embeddings=self.semantic_vocab_size,
            embedding_dim=cfg.d_model,
            padding_idx=self.semantic_pad_idx,
        )
        self.acoustic_embedder = nn.Embedding(
            num_embeddings=self.acoustic_vocab_size,
            embedding_dim=cfg.d_model,
            padding_idx=self.acoustic_pad_idx,
        )

        self.pos_encoder = PositionalEncoding(d_model=cfg.d_model, dropout=cfg.dropout)

        self.decoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=self.nhead,
                dim_feedforward=4 * cfg.d_model,
                dropout=cfg.dropout,
                batch_first=True,
            ),
            num_layers=cfg.num_layers,
        )
        self.out_proj = nn.Linear(cfg.d_model, self.acoustic_vocab_size)

        self.max_segment_len = 100
        self.segment_pos_embedder = nn.Embedding(
            num_embeddings=self.max_segment_len, embedding_dim=cfg.d_model
        )

    def forward(
        self,
        semantic_units: torch.Tensor,  # (B,T)
        acoustic_units: torch.Tensor,  # (B,T)
        src_key_padding_mask: torch.Tensor | None = None,
    ):
        semantic_embeddings = self.semantic_embedder(semantic_units)
        acoustic_embeddings = self.acoustic_embedder(acoustic_units)

        rel_positions = compute_segment_positions(semantic_units)
        rel_positions = torch.clamp(rel_positions, max=self.max_segment_len - 1)
        segment_embeddings = self.segment_pos_embedder(rel_positions)

        x = semantic_embeddings + acoustic_embeddings + segment_embeddings

        x = x * math.sqrt(self.cfg.d_model)

        x = self.pos_encoder(x)

        x = self.decoder(
            src=x,
            mask=None,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=False,
        )
        logits = self.out_proj(x)

        # ensure FP32 when using mixed precision:
        return logits.float()

    @classmethod
    def from_pretrained(cls, checkpoint_path: str):
        assert Path(checkpoint_path).exists()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = cls(S2SMaskGITConfig(**checkpoint["cfg"]))
        model.load_state_dict(checkpoint["model"])
        model.eval()
        num_params = sum(map(torch.numel, model.parameters()))
        print(f"S2SMaskGIT loaded with {num_params:,} parameters.")
        return model

    @classmethod
    def from_remote(
        cls,
        url: str = "https://storage.googleapis.com/zerospeech-checkpoints/S2SMaskGIT/S2SMaskGit-ZeroSylCollapsed-hubert-discrete-train-clean-100-10k-steps.pt",
    ):
        """
        Load model and pretrained weights from URL.
        By default it loads the S2SMaskGIT model trained on ZeroSylCollapsed segments to predict HuBERT discrete units.
        This model was trained on LibriSpeech train-clean-100 for 10k steps.
        """
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model = cls(S2SMaskGITConfig(**checkpoint["cfg"]))
        model.load_state_dict(checkpoint["model"])
        model.eval()
        num_params = sum(map(torch.numel, model.parameters()))
        print(f"S2SMaskGIT loaded with {num_params:,} parameters.")
        return model

    def generate(
        self,
        semantic_units: torch.Tensor,
        num_decoding_steps: int,
        temperature: float = 1.0,
        top_p: float = 0.85,
    ):
        device = next(self.parameters()).device

        semantic_units = semantic_units.to(device)

        acoustic_units_predicted = torch.full_like(
            semantic_units, self.acoustic_mask_idx
        )
        mask = torch.ones_like(semantic_units, dtype=torch.bool)

        for step in range(1, num_decoding_steps + 1):

            # Renamed to masked_items so the math makes semantic sense
            num_masked_items = round(
                math.cos((step - 1) / num_decoding_steps * math.pi / 2)
                * semantic_units.size(0)
            )
            assert mask.long().sum() == num_masked_items

            new_num_masked_items = round(
                math.cos((step) / num_decoding_steps * math.pi / 2)
                * semantic_units.size(0)
            )
            num_tokens_to_lock_in = num_masked_items - new_num_masked_items

            with torch.no_grad():
                logits = self(
                    semantic_units.unsqueeze(0),
                    acoustic_units_predicted.unsqueeze(0),
                ).squeeze()

            logits_to_sample_from = logits.clone()
            logits_to_sample_from[:, self.acoustic_mask_idx :] = -torch.inf
            sampled = sample_with_temperature_and_top_p(
                logits_to_sample_from, temperature=temperature, top_p=top_p
            )

            probs = torch.nn.functional.softmax(logits, dim=-1)

            confidence = probs[torch.arange(sampled.size(0)), sampled]
            confidence[~mask] = -1.0

            order = torch.argsort(confidence, descending=True)

            indices_to_lock_in = order[:num_tokens_to_lock_in]
            values_to_lock_in = sampled[indices_to_lock_in]

            acoustic_units_predicted[indices_to_lock_in] = values_to_lock_in
            mask[indices_to_lock_in] = False

        return acoustic_units_predicted


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


def compute_segment_positions(semantic_units: torch.Tensor) -> torch.Tensor:
    """
    Computes resetting positional indices for consecutive identical tokens.
    Example:
        semantic_units: [A, A, A, B, B, C]
        returns:        [0, 1, 2, 0, 1, 0]
    """
    B, T = semantic_units.shape
    device = semantic_units.device

    # 1. Find boundaries where the semantic unit changes
    # Pad with True at the beginning to always start a segment at index 0
    boundaries = torch.cat(
        [
            torch.ones((B, 1), dtype=torch.bool, device=device),
            semantic_units[:, 1:] != semantic_units[:, :-1],
        ],
        dim=1,
    )

    # 2. Create a global position index for each batch element
    global_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

    # 3. Mask out non-boundaries (zeros out indices where no segment change happens)
    segment_starts = global_idx * boundaries

    # 4. Forward-fill the start indices using cummax
    # This propogates the start index of the current segment across all its frames
    start_indices = segment_starts.cummax(dim=1).values

    # 5. Subtract the start index from the global index to get the relative offset
    relative_positions = global_idx - start_indices

    return relative_positions


def sample_with_temperature_and_top_p(logits, temperature=1.0, top_p=1.0):
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1)

    if temperature != 1.0:
        logits = logits / temperature

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        mask = torch.empty_like(logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(mask, float("-inf"))
    probs = torch.nn.functional.softmax(logits, dim=-1)
    samples = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return samples
