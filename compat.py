"""
compat.py — Centralized environment compatibility fixes.

This module MUST be imported before any TensorFlow, tf-keras, or DeepFace imports.
It handles:
  1. Windows console UTF-8 encoding (prevents cp1252 UnicodeEncodeError)
  2. TensorFlow 2.22 / tf-keras 2.15 API mismatch shim
  3. Suppression of non-critical TensorFlow and DeepFace log noise
"""

import os
import sys
import warnings
import logging


def apply():
    """Apply all compatibility fixes. Safe to call multiple times."""
    _fix_encoding()
    _suppress_logs()
    _fix_tf_keras_shim()


def _fix_encoding():
    """Reconfigure stdout/stderr to UTF-8 on Windows to prevent cp1252 crashes."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _suppress_logs():
    """
    Suppress non-critical warnings from TensorFlow and DeepFace.

    - DEEPFACE_LOG_LEVEL=40 silences DeepFace's deprecation banner (level ERROR).
    - TF_CPP_MIN_LOG_LEVEL=3 silences TF C++ INFO/WARNING messages.
    - TF_ENABLE_ONEDNN_OPTS=0 silences oneDNN floating-point warnings.
    """
    os.environ.setdefault("DEEPFACE_LOG_LEVEL", "40")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    warnings.filterwarnings("ignore")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)


def _fix_tf_keras_shim():
    """
    Patch missing tf.__internal__.register_load_context_function for TF 2.22+.

    tf-keras 2.15 calls tf.__internal__.register_load_context_function at import
    time, but TF 2.22 renamed it to register_call_context_function. Without this
    shim, `from deepface import DeepFace` raises AttributeError.
    """
    try:
        import tensorflow as _tf
        internal = getattr(_tf, "__internal__", None)
        if internal is not None and not hasattr(internal, "register_load_context_function"):
            internal.register_load_context_function = lambda fn: None
    except Exception:
        pass
    try:
        import tensorflow.compat.v2 as _tf2
        internal2 = getattr(_tf2, "__internal__", None)
        if internal2 is not None and not hasattr(internal2, "register_load_context_function"):
            internal2.register_load_context_function = lambda fn: None
    except Exception:
        pass
