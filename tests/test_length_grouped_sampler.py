import math

from data.length_grouped_sampler import CachedLengthGroupedSampler


def _pad_ratio(indices, lengths, batch_size):
    real = 0
    slots = 0
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        if not batch:
            continue
        batch_lengths = [lengths[index] for index in batch]
        real += sum(batch_lengths)
        slots += len(batch) * max(batch_lengths)
    return 1.0 - real / slots


def test_length_grouping_is_reproducible_and_reshuffles_by_epoch():
    lengths = [64 + (index * 97) % 4033 for index in range(1000)]
    sampler = CachedLengthGroupedSampler(
        lengths,
        batch_size=18,
        seed=42,
        mega_batch_mult=20,
    )

    epoch_zero = list(sampler)
    assert sorted(epoch_zero) == list(range(len(lengths)))
    assert epoch_zero == list(
        CachedLengthGroupedSampler(
            lengths,
            batch_size=18,
            seed=42,
            mega_batch_mult=20,
        )
    )

    sampler.set_epoch(1)
    assert list(sampler) != epoch_zero


def test_length_grouping_reduces_padding_for_e17c_microbatches():
    lengths = [64 + (index * 97) % 4033 for index in range(3600)]
    grouped = list(
        CachedLengthGroupedSampler(
            lengths,
            batch_size=3 * 6,
            seed=42,
            mega_batch_mult=20,
        )
    )
    baseline = list(
        CachedLengthGroupedSampler(
            lengths,
            batch_size=3 * 6,
            seed=42,
            mega_batch_mult=1,
        )
    )

    assert _pad_ratio(grouped, lengths, 3) < 0.12
    assert _pad_ratio(grouped, lengths, 3) < _pad_ratio(baseline, lengths, 3) / 3


def test_four_gpu_batch_sharding_is_disjoint_and_length_aligned():
    lengths = [64 + (index * 131) % 4033 for index in range(3600)]
    ordered = list(
        CachedLengthGroupedSampler(
            lengths,
            batch_size=3 * 6,
            seed=7,
            mega_batch_mult=20,
        )
    )
    local_batch_size = 3
    world_size = 4
    batches = [
        ordered[start : start + local_batch_size]
        for start in range(0, len(ordered), local_batch_size)
    ]
    rank_batches = [batches[rank::world_size] for rank in range(world_size)]
    rank_indices = [
        {index for batch in per_rank for index in batch}
        for per_rank in rank_batches
    ]

    for left in range(world_size):
        for right in range(left + 1, world_size):
            assert rank_indices[left].isdisjoint(rank_indices[right])
    assert set().union(*rank_indices) == set(range(len(lengths)))

    complete_steps = min(len(per_rank) for per_rank in rank_batches)
    concurrent_max_spread = []
    for step in range(complete_steps):
        rank_maxima = [
            max(lengths[index] for index in rank_batches[rank][step])
            for rank in range(world_size)
        ]
        concurrent_max_spread.append(max(rank_maxima) - min(rank_maxima))
    assert sum(concurrent_max_spread) / len(concurrent_max_spread) < 128
    assert max(map(len, rank_batches)) - min(map(len, rank_batches)) <= 1
    assert math.ceil(len(batches) / world_size) == max(map(len, rank_batches))
