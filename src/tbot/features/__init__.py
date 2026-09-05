"""tbot.features — event and text features built on the warehouse, point-in-time.

Everything here is a pure read that answers "what could a decision at the close
of `asof` have known?". The first family is 8-K events (ruling 41): item codes,
acceptance time relative to the close, and a local-model sentiment hook. None
of it is a signal yet; a signal is a registered hypothesis, and registering one
is the search protocol's job after the gate closes.
"""

__all__ = ["events", "sentiment"]
