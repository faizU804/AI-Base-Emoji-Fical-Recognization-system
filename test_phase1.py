"""Phase 1 infrastructure verification."""
import compat
compat.apply()

# Test 1: Verify DeepFace loads
from deepface import DeepFace
print("[PASS] DeepFace imported")

# Test 2: Verify emotion inference works
import numpy as np
img = np.zeros((100, 100, 3), dtype=np.uint8)
result = DeepFace.analyze(img, actions=["emotion"], detector_backend="opencv", enforce_detection=False)
if isinstance(result, list):
    result = result[0]
emotion = result["dominant_emotion"]
confidence = result["emotion"][emotion]
print(f"[PASS] DeepFace inference: {emotion} ({confidence:.1f}%)")
emotion_keys = list(result["emotion"].keys())
print(f"  All emotion classes: {emotion_keys}")

# Test 3: Verify OpenCV
import cv2
print(f"[PASS] OpenCV {cv2.__version__}, CascadeClassifier={hasattr(cv2, 'CascadeClassifier')}")

# Test 4: Verify Tkinter
import tkinter as tk
print("[PASS] Tkinter available")

# Test 5: Verify module imports
from emotion_utils import get_emoji, get_emotion_name
print(f"[PASS] emotion_utils: get_emoji('happy')={get_emoji('happy')}")

# Test 6: Verify upload_emotion loads
import upload_emotion
print("[PASS] upload_emotion module loaded")

# Test 7: Verify camera_emotion loads
import camera_emotion
print("[PASS] camera_emotion module loaded")

# Test 8: Verify chat_emotion loads
import chat_emotion
print("[PASS] chat_emotion module loaded")

print()
print("=== PHASE 1 INFRASTRUCTURE: ALL TESTS PASSED ===")
