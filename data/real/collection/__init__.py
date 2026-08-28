"""Offline representation of raw episodes produced by the real xArm collector.

The physical collector remains external to this repository.  This package owns
the tracked boundary that discovers and validates its recorded output.
"""

from data.real.collection.raw_episodes import RawEpisode, discover_raw_episodes

__all__ = ["RawEpisode", "discover_raw_episodes"]

