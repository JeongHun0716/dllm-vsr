from __future__ import annotations

import math
from typing import Optional

import torch
from torch.utils.data import Sampler


class MaxFramesGlobalSortBatchShuffleSampler(Sampler[list[int]]):
    """Dynamic bucket sampler with a sum-of-frames budget.

    Sort globally by length, greedy-pack until sum(lengths) <= max_tokens
    (keeps per-batch lengths close → minimal padding), shuffle batch list per
    epoch, then distribute across DDP ranks. `cost_pair=True` ensures ranks at
    the same step receive batches with comparable cost.
    """

    def __init__(
        self,
        lengths: list[int],
        max_tokens: int,
        max_batch_size: Optional[int] = None,
        shuffle_batches: bool = True,
        drop_last: bool = False,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
        break_ties_with_noise: bool = True,
        cost_pair: bool = True,
    ):
        self.lengths = lengths
        self.max_tokens = int(max_tokens)
        self.shuffle_batches = shuffle_batches
        self.drop_last = drop_last
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.break_ties_with_noise = break_ties_with_noise
        self.cost_pair = bool(cost_pair)
        self.epoch = 0
        self.max_batch_size = max_batch_size
        self._batches_cache: list[list[int]] | None = None
        self._batches_cache_epoch: int = -1
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(f"Invalid rank={rank} for num_replicas={num_replicas}")

    def set_epoch(self, epoch: int):
        if int(epoch) != self.epoch:
            self._batches_cache = None
            self._batches_cache_epoch = -1
        self.epoch = int(epoch)

    def _get_batches(self) -> list[list[int]]:
        if self._batches_cache is None or self._batches_cache_epoch != self.epoch:
            self._batches_cache = self._make_batches()
            self._batches_cache_epoch = self.epoch
        return self._batches_cache

    def _sorted_indices(self) -> list[int]:
        n = len(self.lengths)
        if not self.break_ties_with_noise:
            return sorted(range(n), key=lambda i: self.lengths[i])
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        noise = torch.rand(n, generator=g).tolist()
        return sorted(range(n), key=lambda i: (self.lengths[i], noise[i]))

    def _make_batches(self) -> list[list[int]]:
        indices = self._sorted_indices()
        batches: list[list[int]] = []
        batch: list[int] = []
        cur_sum = 0

        for i in indices:
            L = int(self.lengths[i])

            # Samples that exceed the budget on their own get their own batch.
            if L > self.max_tokens:
                if batch:
                    batches.append(batch)
                    batch, cur_sum = [], 0
                batches.append([i])
                continue

            over_tokens = (cur_sum + L) > self.max_tokens
            over_bs = self.max_batch_size is not None and len(batch) >= self.max_batch_size

            if batch and (over_tokens or over_bs):
                batches.append(batch)
                batch, cur_sum = [], 0

            batch.append(i)
            cur_sum += L

        if batch and not self.drop_last:
            batches.append(batch)

        return batches

    def _batch_cost(self, batch: list[int]) -> int:
        return sum(self.lengths[i] for i in batch) if batch else 0

    def __iter__(self):
        batches = list(self._get_batches())

        if self.num_replicas == 1 or not self.cost_pair:
            if self.shuffle_batches:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                perm = torch.randperm(len(batches), generator=g).tolist()
                batches = [batches[i] for i in perm]

            if self.num_replicas > 1:
                total = int(math.ceil(len(batches) / self.num_replicas) * self.num_replicas)
                if total > len(batches):
                    batches.extend(batches[: total - len(batches)])
                batches = batches[self.rank:total:self.num_replicas]

            for b in batches:
                yield b
            return

        # Cost-paired stride: batches with similar cost go to ranks at the same step.
        sorted_batches = sorted(batches, key=self._batch_cost)
        n = len(sorted_batches)
        rem = n % self.num_replicas
        if rem != 0:
            extra = self.num_replicas - rem
            sorted_batches = sorted_batches + sorted_batches[:extra]

        n_chunks = len(sorted_batches) // self.num_replicas
        chunks = [
            sorted_batches[i * self.num_replicas: (i + 1) * self.num_replicas]
            for i in range(n_chunks)
        ]

        if self.shuffle_batches:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(n_chunks, generator=g).tolist()
            chunks = [chunks[i] for i in perm]

        for chunk in chunks:
            yield chunk[self.rank]

    def __len__(self):
        batches = self._get_batches()
        if self.num_replicas == 1 or not self.cost_pair:
            if self.num_replicas == 1:
                return len(batches)
            total = int(math.ceil(len(batches) / self.num_replicas) * self.num_replicas)
            return total // self.num_replicas

        n = len(batches)
        rem = n % self.num_replicas
        if rem != 0:
            n = n + (self.num_replicas - rem)
        return n // self.num_replicas
