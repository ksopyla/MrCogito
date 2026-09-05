"""Deterministic sortish sampling over cached sequence lengths."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler
from transformers.trainer_pt_utils import get_length_grouped_indices


class CachedLengthGroupedSampler(Sampler[int]):
    """Group similar lengths inside bounded shuffled windows.

    Hugging Face/Accelerate retains ownership of distributed batch sharding. Every
    rank builds the same deterministic index stream, and the prepared dataloader
    assigns disjoint batches to ranks as it does for Trainer's native sampler.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        seed: int,
        mega_batch_mult: int = 20,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if mega_batch_mult < 1:
            raise ValueError("mega_batch_mult must be positive.")
        if len(lengths) < 1:
            raise ValueError("lengths must not be empty.")
        self.lengths = lengths
        self.batch_size = batch_size
        self.seed = seed
        self.mega_batch_mult = mega_batch_mult
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.lengths)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = get_length_grouped_indices(
            self.lengths,
            self.batch_size,
            mega_batch_mult=self.mega_batch_mult,
            generator=generator,
        )
        return iter(indices)
