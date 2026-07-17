"""Feature-view adapters — shared by trainer + serving so train == serve.
Verbatim logic from UID172 model/d0_features.py."""
from .features_v2 import extract_features_v2
from .phasberg import chunk_features


def phasberg_dict(chunk):
    d = chunk_features(chunk or [])
    d["hand_count"] = float(len(chunk or []))
    return d


def v2_dict(chunk):
    return extract_features_v2(chunk or [])
