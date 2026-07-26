"""A local web view for the research pipeline.

Stdlib only -- see :mod:`quantos.web.server` for why, and for what this
deliberately is not.
"""

from quantos.web.server import render_landing, render_page, serve

__all__ = ["render_landing", "render_page", "serve"]
