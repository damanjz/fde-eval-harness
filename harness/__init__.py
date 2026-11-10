"""FDE eval and monitoring harness: run feature golden sets against any model, compare, gate, monitor."""

import sys

__version__ = "0.1.0"


def harden_console() -> None:
    """Never crash on a character the console cannot print (a rupee sign in a failure reason, say)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
