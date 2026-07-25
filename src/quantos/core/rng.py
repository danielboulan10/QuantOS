"""Deterministic, hierarchically-derived random number generation.

The reproducibility contract
----------------------------
Every stochastic component in QuantOS -- every agent, every Monte Carlo path
set, every bootstrap resample, every cross-validation shuffle -- draws from a
generator obtained by *spawning* from a single root :class:`SeedSequence`. The
guarantees this buys are worth stating explicitly, because they are the
difference between a research platform and a pile of scripts:

1. **Bit-exact reproducibility.** A run identified by ``(root_seed, path)``
   produces identical numbers on any machine, any OS, any NumPy >= 1.17.
2. **Stream independence.** Two components with different paths draw from
   provably non-overlapping, statistically independent streams. Two agents in
   the same simulation cannot accidentally share a sequence.
3. **Order independence.** Component *B*'s numbers do not change when
   component *A* is added, removed, or draws a different amount. This is the
   property naive ``seed + i`` schemes lack, and its absence is why so many
   backtests cannot be reproduced after a refactor.
4. **Refactor stability.** Streams are keyed by a *semantic path* such as
   ``"sim/agents/mm_01/quote_noise"``, not by construction order, so
   reordering a loop does not silently change results.

Why not just pass ``numpy.random.Generator`` around?
----------------------------------------------------
You can, and :meth:`SeedBank.generator` hands you exactly that. What the bank
adds is the *derivation discipline*: a component asks for a stream by name and
is guaranteed to get its own. Nothing is shared by default, so there is no
global mutable state to leak between tests.

References
----------
O'Neill, M. E. (2014), "PCG: A family of simple fast space-efficient
    statistically good algorithms for random number generation", HMC-CS-2014-0905.
NumPy Enhancement Proposal 19, "Random number generator policy".
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

__all__ = ["DEFAULT_SEED", "SeedBank", "spawn_key"]

DEFAULT_SEED = 20240719
"""Repository-wide default root seed.

Fixed so that ``quantos demo`` produces byte-identical output for every reader
of the README. Research code should pass an explicit seed instead of relying
on it.
"""


def spawn_key(path: str) -> int:
    """Map a semantic stream path to a stable 128-bit spawn key.

    Purpose
        Turn a human-readable name such as ``"sim/agents/mm_01"`` into an
        integer usable as a :class:`numpy.random.SeedSequence` spawn key,
        without any dependence on call order.
    Inputs
        ``path`` -- arbitrary string; conventionally slash-delimited.
    Outputs
        ``int`` in ``[0, 2**128)``.
    Assumptions
        BLAKE2b is used purely as a fixed, portable string-to-integer map --
        no cryptographic property is relied upon, only stability across
        platforms and Python versions. (``hash()`` is unusable here: it is
        salted per-process by default.)
    Complexity
        ``O(len(path))``.

    Example
        >>> spawn_key("a") == spawn_key("a")
        True
        >>> spawn_key("a") == spawn_key("b")
        False
    """
    digest = hashlib.blake2b(path.encode("utf-8"), digest_size=16).digest()
    return int.from_bytes(digest, "big")


@dataclass(frozen=True)
class SeedBank:
    """Hierarchical factory for independent, reproducible random streams.

    A :class:`SeedBank` is immutable and cheap to copy. :meth:`child` returns a
    new bank one level down the tree; :meth:`generator` materialises the
    :class:`numpy.random.Generator` for the current node.

    Example
        >>> bank = SeedBank(root=42)
        >>> a = bank.child("agents").child("mm_01").generator()
        >>> b = bank.child("agents").child("mm_02").generator()
        >>> bool(a.normal() != b.normal())      # independent streams
        True
        >>> again = SeedBank(root=42).child("agents").child("mm_01").generator()
        >>> float(again.normal()) == float(SeedBank(root=42).child(
        ...     "agents").child("mm_01").generator().normal())
        True
    """

    root: int = DEFAULT_SEED
    path: tuple[str, ...] = field(default_factory=tuple)

    def child(self, name: str) -> SeedBank:
        """Descend one level, returning a new bank. Does not mutate ``self``."""
        return SeedBank(root=self.root, path=(*self.path, name))

    @property
    def qualified_name(self) -> str:
        """The full slash-delimited path, for logging and provenance records."""
        return "/".join(self.path) if self.path else "<root>"

    def seed_sequence(self) -> np.random.SeedSequence:
        """The :class:`numpy.random.SeedSequence` for this node.

        Derivation mixes the root seed with the spawn key of every path
        component. Because :class:`SeedSequence` entropy mixing is
        order-sensitive but *path*-determined here, siblings are independent
        and the mapping is stable under refactoring.
        """
        entropy: list[int] = [int(self.root)]
        entropy.extend(spawn_key(component) for component in self.path)
        return np.random.SeedSequence(entropy)

    def generator(self) -> np.random.Generator:
        """A fresh :class:`numpy.random.Generator` (PCG64DXSM) for this node.

        A *new* generator each call, positioned at the start of the stream --
        so calling twice reproduces the same numbers. Hold onto the returned
        object if you want to advance through a stream.
        """
        return np.random.Generator(np.random.PCG64DXSM(self.seed_sequence()))

    def spawn(self, count: int, prefix: str = "s") -> list[np.random.Generator]:
        """``count`` independent generators, for parallel or batched work.

        Example
            >>> gens = SeedBank(root=1).spawn(3, prefix="path")
            >>> len({float(g.normal()) for g in gens})
            3
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        return [self.child(f"{prefix}_{i:06d}").generator() for i in range(count)]

    def __iter__(self) -> Iterator[np.random.Generator]:
        """Endless supply of independent generators; useful for path loops."""
        index = 0
        while True:
            yield self.child(f"auto_{index:012d}").generator()
            index += 1
