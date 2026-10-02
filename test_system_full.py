"""
test_system_full.py — Comprehensive System Integration & Unit Test Suite.

Tests all modules:
  1. Centralized compatibility shim (TF 2.22 / tf-keras / UTF-8 encoding)
  2. emotion_utils module
  3. text_emotion_ai module (DistilRoBERTa NLP + Twitter sentiment dataset engine)
  4. chat_emotion module UI logic and preset routing
  5. camera_emotion module architecture (threading, throttling, smoothing)
  6. upload_emotion module & DeepFace facial emotion inference
  7. main.py EmotionApp integration
"""

import os
import sys
import time
import unittest
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import compat
compat.apply()


class TestSystemIntegration(unittest.TestCase):

    def test_01_environment_and_compat(self):
        """Verify TF compatibility shim and UTF-8 stream reconfigurations."""
        import tensorflow as tf
        import tf_keras
        
        # Verify internal function shim is in place
        self.assertTrue(hasattr(tf, "__internal__"))
        self.assertTrue(hasattr(tf.__internal__, "register_load_context_function"))
        
        # Verify streams are UTF-8 capable
        self.assertTrue(hasattr(sys.stdout, "encoding"))
        print("[PASS] Environment and TF compatibility verified.")

    def test_02_emotion_utils(self):
        """Verify static emotion utilities map correctly."""
        from emotion_utils import get_emoji, get_emotion_name
        
        self.assertEqual(get_emoji("happy"), "😊")
        self.assertEqual(get_emoji("sad"), "😢")
        self.assertEqual(get_emoji("angry"), "😡")
        self.assertEqual(get_emotion_name("happy"), "HAPPY")
        self.assertEqual(get_emotion_name("surprise"), "SURPRISE")
        print("[PASS] Emotion utils verified.")

    def test_03_emoji_dataset_engine(self):
        """Verify CSV datasets are parsed, scored, and emojis are ranked."""
        from text_emotion_ai import EmojiDatasetEngine
        
        engine = EmojiDatasetEngine(BASE_DIR)
        
        # Check datasets loaded
        self.assertGreater(len(engine.sentiment_records), 100, "Should load sentiment records")
        self.assertGreater(len(engine.popularity), 100, "Should load Twitter popularity records")
        
        # Check recommendations
        for emotion in ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]:
            recs = engine.get_recommendations(emotion, top_n=5)
            self.assertIsInstance(recs, list)
            self.assertEqual(len(recs), 5)
            # Verify distinct emojis
            self.assertEqual(len(set(recs)), 5, f"Recommendations for {emotion} should be unique")
            
        # Test blending with secondary emotion
        blended = engine.get_recommendations("happy", secondary_emotion="surprise", secondary_conf=35.0, top_n=5)
        self.assertEqual(len(blended), 5)
        print(f"[PASS] Dataset Engine verified ({len(engine.sentiment_records)} emojis indexed).")

    def test_04_text_emotion_ai_inference(self):
        """Verify Transformer NLP classification and nuance detection."""
        from text_emotion_ai import get_text_ai
        
        ai = get_text_ai(BASE_DIR)
        ready = ai.ensure_model_ready(timeout_sec=15.0)
        self.assertTrue(ready, "DistilRoBERTa model should load within timeout")
        
        # Test 1: High positive sentiment
        res1 = ai.analyze_text("I just got promoted at work! Best day ever!")
        self.assertEqual(res1["primary_emotion"], "happy")
        self.assertGreater(res1["primary_confidence"], 40.0)
        self.assertEqual(res1["sentiment_type"], "Positive")
        self.assertIn("😊", res1["recommended_emojis"])
        
        # Test 2: Crucial Negation Test ("not happy" -> sad/grief)
        res2 = ai.analyze_text("I am definitely not happy with this bad service.")
        self.assertIn(res2["primary_emotion"], ["sad", "angry"])
        self.assertEqual(res2["sentiment_type"], "Negative")
        self.assertIn("😢", res2["recommended_emojis"] + [res2["primary_emoji"]])
        
        # Test 3: Strong anger
        res3 = ai.analyze_text("Stop lying to me, I am absolutely furious!")
        self.assertEqual(res3["primary_emotion"], "angry")
        self.assertGreater(res3["primary_confidence"], 70.0)
        
        # Test 4: Extreme fear
        res4 = ai.analyze_text("I'm terrified of flying during thunderstorms.")
        self.assertEqual(res4["primary_emotion"], "fear")
        self.assertGreater(res4["primary_confidence"], 80.0)
        
        # Test 5: Disgust
        res5 = ai.analyze_text("This milk is completely sour and rotten.")
        self.assertEqual(res5["primary_emotion"], "disgust")
        self.assertGreater(res5["primary_confidence"], 80.0)
        
        # Test 6: Empty text edge case
        res6 = ai.analyze_text("")
        self.assertEqual(res6["primary_emotion"], "neutral")
        self.assertEqual(len(res6["recommended_emojis"]), 5)
        
        print("[PASS] Text Emotion AI verified across all standard & edge test cases.")

    def test_05_camera_emotion_architecture(self):
        """Verify CameraEmotion class configurations and constants."""
        from camera_emotion import CameraEmotion
        
        self.assertEqual(CameraEmotion.INFERENCE_INTERVAL, 0.6)
        self.assertEqual(CameraEmotion.SMOOTHING_WINDOW, 5)
        self.assertEqual(CameraEmotion.CONFIDENCE_THRESHOLD, 35.0)
        print("[PASS] CameraEmotion module architecture verified.")

    def test_06_deepface_facial_inference(self):
        """Verify DeepFace emotion inference on real test image."""
        from deepface import DeepFace
        
        test_img_path = os.path.join(BASE_DIR, "test_face_happy.jpg")
        self.assertTrue(os.path.exists(test_img_path), "test_face_happy.jpg must exist")
        
        result = DeepFace.analyze(
            img_path=test_img_path,
            actions=["emotion"],
            enforce_detection=False,
            silent=True
        )
        if isinstance(result, list):
            result = result[0]
            
        dominant = result["dominant_emotion"]
        self.assertEqual(dominant.lower(), "happy", f"Expected 'happy' for test image, got {dominant}")
        print(f"[PASS] DeepFace facial emotion inference verified (detected: {dominant.upper()}).")

    def test_07_chat_emotion_module_loads(self):
        """Verify ChatEmotion GUI module imports and has required interface."""
        import chat_emotion
        self.assertTrue(hasattr(chat_emotion, "ChatEmotion"))
        print("[PASS] ChatEmotion module verified.")

    def test_08_main_app_structure(self):
        """Verify main.py EmotionApp structure without starting GUI event loop."""
        import main
        self.assertTrue(hasattr(main, "EmotionApp"))
        self.assertTrue(hasattr(main.EmotionApp, "open_camera"))
        self.assertTrue(hasattr(main.EmotionApp, "open_upload"))
        self.assertTrue(hasattr(main.EmotionApp, "open_chat"))
        print("[PASS] Main application structure verified.")

    def test_09_ui_theme_and_3d_engine(self):
        """Verify unified design system, responsive tiers, and HUD components."""
        import ui_theme
        self.assertTrue(hasattr(ui_theme, "Theme"))
        self.assertTrue(hasattr(ui_theme, "ParticleCanvas3D"))
        self.assertTrue(hasattr(ui_theme, "CameraHUDOverlay"))
        self.assertTrue(hasattr(ui_theme, "CyberButton"))
        self.assertTrue(hasattr(ui_theme, "ResponsiveLayout"))
        
        # Test responsive breakpoints
        self.assertEqual(ui_theme.ResponsiveLayout.get_tier(1200), ui_theme.ResponsiveLayout.DESKTOP)
        self.assertEqual(ui_theme.ResponsiveLayout.get_tier(800), ui_theme.ResponsiveLayout.TABLET)
        self.assertEqual(ui_theme.ResponsiveLayout.get_tier(500), ui_theme.ResponsiveLayout.MOBILE)
        
        # Test theme emotion metadata
        self.assertIn("happy", ui_theme.Theme.EMOTION_THEMES)
        self.assertIn("color", ui_theme.Theme.EMOTION_THEMES["happy"])
        print("[PASS] UI Theme and 3D engine components verified.")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSystemIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    sys.exit(0 if res.wasSuccessful() else 1)

