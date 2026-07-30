"""PathReliability — a retraining-free reliability audit + abstention gate for
frozen medical-imaging encoders. Bring your own model and dataset."""
from .audit import audit          # noqa: F401
from .report import write_all     # noqa: F401
from .adapter import Encoder      # noqa: F401

__version__ = "0.1.0"
