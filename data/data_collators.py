"""
Data collators for Concept Encoder training objectives.

DataCollatorForTSDAE:
    TSDAE-style (Transformer-based Sequential Denoising Auto-Encoder) training.
    Randomly "deletes" tokens by zeroing their attention_mask, forcing the
    concept encoder to reconstruct the full sequence from surviving tokens only.

    Wang et al., "TSDAE: Using Transformer-based Sequential Denoising
    Auto-Encoder for Unsupervised Sentence Embedding Learning", EMNLP 2021.
"""

from typing import Any, Dict, List, Optional
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

        # Apply deletion to attention_mask: deleted tokens become invisible
        encoder_mask = original_mask.clone()
        encoder_mask[delete_mask] = 0

        # Labels: original token ids at every non-pad position, -100 at padding
        labels = padded_ids.clone()
        labels[original_mask == 0] = -100

        return {
            "input_ids": padded_ids,
            "attention_mask": encoder_mask,
            "labels": labels,
        }
