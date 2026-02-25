from .dataset import get_dataset
# from .metrics import get_all_metrics
from .utils import CharVocab, StringDataset

def get_all_metrics(*args, **kwargs):
    # Lazy import to avoid importing heavy metric deps at package import time
    from .metrics import get_all_metrics as _get_all_metrics
    return _get_all_metrics(*args, **kwargs)

__version__ = '0.3.1'
__all__ = [
    "get_dataset", "get_all_metrics", "CharVocab",
    "StringDataset"]
