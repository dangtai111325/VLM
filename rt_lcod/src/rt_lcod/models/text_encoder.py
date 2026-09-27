from __future__ import annotations

import hashlib
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F


class HashTextEncoder(nn.Module):
    """Deterministic dependency-free encoder used only by tests/smoke paths."""

    def __init__(self, dim: int = 24):
        super().__init__()
        self.dim = dim

    @lru_cache(maxsize=4096)
    def _encode_one_cpu(self, text: str) -> torch.Tensor:
        digest = hashlib.sha256(text.lower().strip().encode("utf-8")).digest()
        values = list(digest)
        repeated = (values * ((self.dim + len(values) - 1) // len(values)))[: self.dim]
        tensor = torch.tensor(repeated, dtype=torch.float32)
        return F.normalize(tensor / 127.5 - 1.0, dim=0)

    def encode(self, texts: list[str], device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.stack([self._encode_one_cpu(t) for t in texts]).to(device)


class HFTextEncoder(nn.Module):
    """Frozen Hugging Face sentence embedding encoder with prompt-level caching."""

    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device_name = device
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._cache: dict[str, torch.Tensor] = {}

    @property
    def dim(self) -> int:
        return int(self.model.config.hidden_size)

    @torch.inference_mode()
    def encode(self, texts: list[str], device: torch.device | str | None = None) -> torch.Tensor:
        target_device = str(device or self.device_name)
        missing = [t for t in texts if t not in self._cache]
        if missing:
            tokens = self.tokenizer(missing, padding=True, truncation=True, return_tensors="pt")
            tokens = {k: v.to(self.device_name) for k, v in tokens.items()}
            output = self.model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (output * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooled = F.normalize(pooled.float(), dim=-1).cpu()
            for text, embedding in zip(missing, pooled):
                self._cache[text] = embedding
        return torch.stack([self._cache[t] for t in texts]).to(target_device)
