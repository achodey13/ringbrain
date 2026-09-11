import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """Call once at process startup (API server, CLI, eval harness). Every
    RingBrain module logs through a `ringbrain.*` logger name so this one
    handler configuration covers all of them.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    )
    root = logging.getLogger("ringbrain")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
