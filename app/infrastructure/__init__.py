"""Infrastructure adapters — concrete implementations of the domain ports.

Each subpackage owns one seam and exposes a `build_*` factory that reads
settings and returns a port implementation. The container calls those factories
once at startup; nothing else knows which implementation is in play.
"""
