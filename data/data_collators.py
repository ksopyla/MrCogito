"""
Data collators for Concept Encoder training objectives.

DataCollatorForTSDAE:
    TSDAE-style (Transformer-based Sequential Denoising Auto-Encoder) training.
    Randomly "deletes" tokens by zeroing their attention_mask, forcing the
    concept encoder to reconstruct the full sequence from surviving tokens only.

    Wang et al., "TSDAE: Using Transformer-based Sequential Denoising
    Auto-Encoder for Unsupervised Sentence Embedding Learning", EMNLP 2021.

DataCollatorForPrefixGeneration:
    SODA-inspired prefix generation training (Hudson et al., CVPR 2024).
    Splits documents into prefix (encoder input) and suffix (decoder target),
    forcing the concept bottleneck to capture semantics rather than surface
    tokens, since the decoder must generate different content than the encoder saw.
"""

from typing import Any, Dict, List, Optional
import random
import torch


class DataCollatorForTSDAE:
    """
    Collates batches for TSDAE-style denoising training.

    Instead of masking tokens (MLM), this collator *deletes* tokens by setting
    their attention_mask to 0.  The encoder's key_padding_mask then prevents
    concepts from attending to deleted positions, so the decoder must
    reconstruct the full sequence purely from the surviving token information
    compressed into concept vectors.

    Output contract:
        input_ids      : [B, L]  clean token ids (unchanged)
        attention_mask  : [B, L]  1 = visible to encoder, 0 = deleted
        labels         : [B, L]  reconstruction targets at ALL positions
                                  (pad positions set to -100)

    The model's forward() should compute dense cross-entropy at every
    non-pad position, NOT sparse MLM loss.
    """

    def __init__(
        self,
        tokenizer,
        deletion_rate: float = 0.6,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.deletion_rate = deletion_rate
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self._vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else None

        self._special_ids = set()
        for attr in ("cls_token_id", "sep_token_id", "pad_token_id",
                      "bos_token_id", "eos_token_id"):
            tid = getattr(tokenizer, attr, None)
            if tid is not None:
                self._special_ids.add(tid)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [f["input_ids"] for f in features]

        max_len = min(max(len(x) for x in input_ids_list), self.max_length)
        batch_size = len(input_ids_list)

        padded_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        original_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

        for i, ids in enumerate(input_ids_list):
            length = min(len(ids), max_len)
            padded_ids[i, :length] = torch.tensor(ids[:length], dtype=torch.long)
            original_mask[i, :length] = 1

            # Defensive guard for pathological empty examples. The AR denoising
            # path appends EOS during preprocessing, but if a cached/legacy row is
            # empty, give the encoder one visible token rather than letting an
            # all-masked row produce NaNs in cross-attention.
            if length == 0 and max_len > 0:
                padded_ids[i, 0] = self.eos_token_id if self.eos_token_id is not None else self.pad_token_id
                original_mask[i, 0] = 1

        # Build a boolean mask of deletable (non-special, non-pad) positions
        deletable = original_mask.clone().bool()
        for sid in self._special_ids:
            deletable &= (padded_ids != sid)

        # Sample deletion: each deletable token is independently dropped
        delete_probs = torch.full_like(padded_ids, self.deletion_rate, dtype=torch.float)
        delete_draw = torch.bernoulli(delete_probs).bool()
        delete_mask = deletable & delete_draw

        # Ensure at least one token survives per sequence so the encoder
        # receives *some* information (avoid completely empty inputs).
        for i in range(batch_size):
            surviving = original_mask[i].bool() & ~delete_mask[i]
            if surviving.sum() == 0:
                # Restore the first deletable position
                first_deletable = deletable[i].nonzero(as_tuple=True)[0]
                if len(first_deletable) > 0:
                    delete_mask[i, first_deletable[0]] = False
                else:
                    # No deletable content token exists (e.g. a single EOS-only row).
                    # Keep the first real token visible so the encoder never receives
                    # an all-zero attention mask.
                    first_real = original_mask[i].nonzero(as_tuple=True)[0]
                    if len(first_real) > 0:
                        delete_mask[i, first_real[0]] = False

        # Apply deletion to attention_mask: deleted tokens become invisible
        encoder_mask = original_mask.clone()
        encoder_mask[delete_mask] = 0

        # Labels: original token ids at every non-pad position, -100 at padding
        labels = padded_ids.clone()
        labels[original_mask == 0] = -100

        if self._vocab_size is not None:
            ids_min = int(padded_ids.min().item())
            ids_max = int(padded_ids.max().item())
            if ids_min < 0 or ids_max >= self._vocab_size:
                raise ValueError(
                    f"DataCollatorForTSDAE produced input_ids outside tokenizer range: "
                    f"min={ids_min}, max={ids_max}, vocab_size={self._vocab_size}"
                )
            valid_labels = labels[labels != -100]
            if valid_labels.numel() > 0:
                label_min = int(valid_labels.min().item())
                label_max = int(valid_labels.max().item())
                if label_min < 0 or label_max >= self._vocab_size:
                    raise ValueError(
                        f"DataCollatorForTSDAE produced labels outside tokenizer range: "
                        f"min={label_min}, max={label_max}, vocab_size={self._vocab_size}"
                    )
            if (encoder_mask.sum(dim=1) == 0).any():
                raise ValueError("DataCollatorForTSDAE produced an all-zero encoder attention_mask row.")

        return {
            "input_ids": padded_ids,
            "attention_mask": encoder_mask,
            "labels": labels,
        }


class DataCollatorForPrefixGeneration:
    """
    SODA-inspired collator: split each document into prefix (encoder) and
    suffix (decoder target).

    The encoder sees clean prefix tokens and compresses them into concept
    vectors. The decoder must generate the suffix conditioned only on those
    concepts. Because the suffix never appears in the encoder input, the
    concept bottleneck is forced to capture semantic gist rather than
    surface-level token patterns.

    Sequence template
    -----------------
    The raw tokenized sequence ``[CLS] content_tokens [SEP]`` is split into::

        Encoder input  :  [CLS]  prefix_content  [SEP]   +  [PAD]...
        Decoder target :         suffix_content   [SEP]   +  [PAD]...

    The final ``[SEP]`` in the suffix acts as end-of-generation marker so the
    model learns *when* to stop.  All suffix tokens including ``[SEP]`` are
    subject to diffusion masking (masking is applied inside the model, not
    here).

    Output contract
    ---------------
    ::

        prefix_input_ids      : [B, P]  -- [CLS] prefix [SEP] + padding
        prefix_attention_mask : [B, P]  -- 1 = real, 0 = pad
        suffix_input_ids      : [B, S]  -- suffix [SEP] + padding
        suffix_attention_mask : [B, S]  -- 1 = real, 0 = pad
        labels                : [B, S]  -- same as suffix_input_ids but -100 at pad
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        prefix_ratio_min: float = 0.3,
        prefix_ratio_max: float = 0.5,
        min_prefix_content: int = 5,
        min_suffix_content: int = 10,
        split_strategy: str = "sentence_boundary",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prefix_ratio_min = prefix_ratio_min
        self.prefix_ratio_max = prefix_ratio_max
        self.min_prefix_content = min_prefix_content
        self.min_suffix_content = min_suffix_content
        self.split_strategy = split_strategy

        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.cls_token_id = getattr(tokenizer, "cls_token_id", None)
        self.sep_token_id = getattr(tokenizer, "sep_token_id", None)
        self._token_text_cache: Dict[int, str] = {}

        if self.sep_token_id is None:
            raise ValueError(
                "Tokenizer must have a sep_token_id for prefix generation "
                "(used as end-of-sequence marker in the suffix)."
            )
        if self.split_strategy not in {"token_random", "sentence_boundary"}:
            raise ValueError(
                "split_strategy must be one of {'token_random', 'sentence_boundary'}."
            )

    def _extract_content(self, ids: List[int]) -> List[int]:
        """Strip leading [CLS] and trailing [SEP] to get pure content tokens."""
        start = 0
        if self.cls_token_id is not None and len(ids) > 0 and ids[0] == self.cls_token_id:
            start = 1
        end = len(ids)
        if self.sep_token_id is not None and end > start and ids[end - 1] == self.sep_token_id:
            end -= 1
        # Also strip any trailing padding
        while end > start and ids[end - 1] == self.pad_token_id:
            end -= 1
        return ids[start:end]

    def _get_token_text(self, token_id: int) -> str:
        if token_id not in self._token_text_cache:
            token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            self._token_text_cache[token_id] = token
        return self._token_text_cache[token_id]

    def _is_sentence_boundary_token(self, token_id: int) -> bool:
        token = self._get_token_text(token_id)
        normalized = token.replace("##", "").strip()
        if normalized in {".", "!", "?", ";", ":"}:
            return True
        return normalized.endswith((".", "!", "?"))

    def _choose_random_split(self, content_len: int) -> int:
        min_p = self.min_prefix_content
        min_s = self.min_suffix_content
        if content_len <= 1:
            return content_len
        if content_len < min_p + min_s:
            return max(1, min(content_len - 1, content_len // 2))

        lo = max(min_p, int(content_len * self.prefix_ratio_min))
        hi = min(content_len - min_s, int(content_len * self.prefix_ratio_max))
        if lo > hi:
            lo = min_p
            hi = content_len - min_s
        return random.randint(lo, max(lo, hi))

    def _choose_sentence_boundary_split(self, content: List[int]) -> int:
        content_len = len(content)
        split = self._choose_random_split(content_len)

        min_p = min(self.min_prefix_content, max(1, content_len - 1))
        max_p = max(min_p, content_len - self.min_suffix_content)
        target = split

        candidates = [
            idx
            for idx, token_id in enumerate(content[:-1], start=1)
            if min_p <= idx <= max_p and self._is_sentence_boundary_token(token_id)
        ]
        if not candidates:
            return split

        in_ratio_band = [
            idx
            for idx in candidates
            if int(content_len * self.prefix_ratio_min) <= idx <= int(content_len * self.prefix_ratio_max)
        ]
        pool = in_ratio_band or candidates
        return min(pool, key=lambda idx: abs(idx - target))

    def _choose_split(self, content: List[int]) -> int:
        if self.split_strategy == "token_random":
            return self._choose_random_split(len(content))
        return self._choose_sentence_boundary_split(content)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)

        prefix_seqs: List[List[int]] = []
        suffix_seqs: List[List[int]] = []

        for f in features:
            raw_ids = f["input_ids"]
            if isinstance(raw_ids, torch.Tensor):
                raw_ids = raw_ids.tolist()

            content = self._extract_content(raw_ids)
            split = self._choose_split(content)

            prefix_content = content[:split]
            suffix_content = content[split:]

            # Build: [CLS] prefix_content [SEP]
            prefix_ids: List[int] = []
            if self.cls_token_id is not None:
                prefix_ids.append(self.cls_token_id)
            prefix_ids.extend(prefix_content)
            prefix_ids.append(self.sep_token_id)

            # Build: suffix_content [SEP]
            suffix_ids = list(suffix_content)
            suffix_ids.append(self.sep_token_id)

            prefix_seqs.append(prefix_ids)
            suffix_seqs.append(suffix_ids)

        max_prefix_len = min(max(len(s) for s in prefix_seqs), self.max_length)
        max_suffix_len = min(max(len(s) for s in suffix_seqs), self.max_length)

        prefix_input_ids = torch.full((batch_size, max_prefix_len), self.pad_token_id, dtype=torch.long)
        prefix_attention_mask = torch.zeros(batch_size, max_prefix_len, dtype=torch.long)
        suffix_input_ids = torch.full((batch_size, max_suffix_len), self.pad_token_id, dtype=torch.long)
        suffix_attention_mask = torch.zeros(batch_size, max_suffix_len, dtype=torch.long)
        labels = torch.full((batch_size, max_suffix_len), -100, dtype=torch.long)

        for i in range(batch_size):
            p_len = min(len(prefix_seqs[i]), max_prefix_len)
            prefix_input_ids[i, :p_len] = torch.tensor(prefix_seqs[i][:p_len], dtype=torch.long)
            prefix_attention_mask[i, :p_len] = 1

            s_len = min(len(suffix_seqs[i]), max_suffix_len)
            suffix_input_ids[i, :s_len] = torch.tensor(suffix_seqs[i][:s_len], dtype=torch.long)
            suffix_attention_mask[i, :s_len] = 1
            labels[i, :s_len] = suffix_input_ids[i, :s_len]

        return {
            "prefix_input_ids": prefix_input_ids,
            "prefix_attention_mask": prefix_attention_mask,
            "suffix_input_ids": suffix_input_ids,
            "suffix_attention_mask": suffix_attention_mask,
            "labels": labels,
        }
