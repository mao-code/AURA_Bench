from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from authbench.eval.hf_utils import load_model, load_tokenizer


def _chunk_iterable(items: Sequence[str], chunk_size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


@dataclass
class EmbeddingResult:
    vectors: torch.Tensor
    token_embeddings: Optional[torch.Tensor] = None
    attention_masks: Optional[torch.Tensor] = None
    ids: Optional[List[str]] = None


class HuggingFaceEmbedder:
    """Thin wrapper to produce normalized embeddings from HF checkpoints."""

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        max_length: int = 512,
        pooling: str = "mean",
        normalize: bool = True,
        torch_dtype: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        tokenizer=None,
        truncation_side: str = "right",
        trust_remote_code: bool = False,
        allow_remote_code_fallback: bool = True,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = max_length
        self.pooling = pooling
        self.normalize = normalize
        self.torch_dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype

        self.tokenizer = tokenizer or load_tokenizer(
            model_name_or_path,
            truncation_side=truncation_side,
            trust_remote_code=trust_remote_code,
            allow_remote_code_fallback=allow_remote_code_fallback,
        )
        # Some causal LM tokenizers (e.g., LLaMA) ship without a pad token. Align pad
        # to EOS when available to support batched padding; otherwise add a dedicated
        # pad token and resize embeddings accordingly.
        pad_added = False
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                pad_added = True
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.model = model or load_model(
            model_name_or_path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=trust_remote_code,
            allow_remote_code_fallback=allow_remote_code_fallback,
        )
        if pad_added:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)

    @property
    def dimension(self) -> int:
        return getattr(self.model.config, "hidden_size", None) or getattr(
            self.model.config, "d_model", None
        )

    def _pool_outputs(self, outputs, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_state = outputs.last_hidden_state
        if self.pooling == "cls":
            pooled = hidden_state[:, 0]
        elif self.pooling == "last":
            lengths = attention_mask.sum(dim=1) - 1
            pooled = hidden_state[torch.arange(hidden_state.size(0)), lengths]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1)
            pooled = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
        return pooled

    def _apply_prefix(self, texts: Sequence[str], prefix: str) -> List[str]:
        if prefix:
            return [prefix + text for text in texts]
        return list(texts)

    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        prefix: str = "",
        return_tokens: bool = False,
        show_progress: bool = False,
    ) -> EmbeddingResult:
        """Encode a list of texts into embeddings."""

        texts_list = list(texts)
        was_training = self.model.training
        self.model.eval()
        outputs: List[torch.Tensor] = []
        token_outputs: List[torch.Tensor] = []
        mask_outputs: List[torch.Tensor] = []

        iterator = _chunk_iterable(texts_list, batch_size)
        if show_progress:
            iterator = tqdm(
                iterator, desc="Embedding", total=(len(texts_list) + batch_size - 1) // batch_size
            )

        with torch.inference_mode():
            for batch in iterator:
                prefixed = self._apply_prefix(batch, prefix)
                padding_strategy = "max_length" if return_tokens else True
                tokens = self.tokenizer(
                    prefixed,
                    padding=padding_strategy,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                model_outputs = self.model(**tokens)
                pooled = self._pool_outputs(model_outputs, tokens["attention_mask"])
                if self.normalize:
                    pooled = F.normalize(pooled, p=2, dim=1)
                outputs.append(pooled.cpu())

                if return_tokens:
                    token_embs = model_outputs.last_hidden_state
                    if self.normalize:
                        token_embs = F.normalize(token_embs, p=2, dim=2)
                    token_outputs.append(token_embs.cpu())
                    mask_outputs.append(tokens["attention_mask"].cpu())

        if was_training:
            self.model.train()

        vectors = torch.cat(outputs, dim=0) if outputs else torch.empty(0, device="cpu")
        token_embeddings = torch.cat(token_outputs, dim=0) if token_outputs else None
        attention_masks = torch.cat(mask_outputs, dim=0) if mask_outputs else None
        return EmbeddingResult(
            vectors=vectors, token_embeddings=token_embeddings, attention_masks=attention_masks
        )
