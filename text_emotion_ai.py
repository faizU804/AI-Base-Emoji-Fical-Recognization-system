"""
text_emotion_ai.py — Text Emotion Analysis & Dataset-Driven Emoji Recommendation.

Combines:
  1. DistilRoBERTa-based 7-class emotion classification (j-hartmann/emotion-english-distilroberta-base)
  2. Dataset ranking using Emoji_Sentiment_Data_v1.0.csv and Emojitracker_20150604.csv
  3. Multi-emoji ranked recommendations, secondary emotion blending, and sentiment analytics.
"""

import os
import sys
import csv
import math
import threading
from typing import Dict, List, Any, Optional

import compat
compat.apply()

# 7 standard emotion categories aligned across DeepFace and NLP
EMOTION_MAPPING = {
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
    "neutral": "neutral"
}

EMOTION_DISPLAY = {
    "happy": {"name": "HAPPY / JOY", "emoji": "😊", "color": "#10B981", "sentiment": "Positive"},
    "sad": {"name": "SAD / GRIEF", "emoji": "😢", "color": "#3B82F6", "sentiment": "Negative"},
    "angry": {"name": "ANGRY / RAGE", "emoji": "😡", "color": "#EF4444", "sentiment": "Negative"},
    "surprise": {"name": "SURPRISE / AWE", "emoji": "😲", "color": "#F59E0B", "sentiment": "Neutral/Positive"},
    "fear": {"name": "FEAR / ANXIETY", "emoji": "😨", "color": "#8B5CF6", "sentiment": "Negative"},
    "disgust": {"name": "DISGUST / REVULSION", "emoji": "🤢", "color": "#84CC16", "sentiment": "Negative"},
    "neutral": {"name": "NEUTRAL / OBJECTIVE", "emoji": "😐", "color": "#6B7280", "sentiment": "Neutral"}
}

# Curated seed emojis guaranteed for high relevance and universal rendering
CURATED_SEEDS = {
    "happy": ["😊", "😂", "🥰", "😍", "🎉", "🥳", "✨", "😁", "😃", "💖", "🙌", "🤩"],
    "sad": ["😢", "😭", "💔", "😞", "😔", "🥺", "😿", "🌧️", "🥀", "😣", "🖤", "😥"],
    "angry": ["😡", "😠", "🤬", "😤", "💢", "👿", "🔥", "💥", "😾", "🖕", "🌋", "👊"],
    "surprise": ["😲", "😮", "😱", "🤯", "😳", "😯", "✨", "❗", "🙀", "🎉", "💥", "👀"],
    "fear": ["😨", "😰", "😱", "🥶", "😖", "🫣", "⚡", "🥺", "😬", "🕯️", "🫨", "🌪️"],
    "disgust": ["🤢", "🤮", "😒", "🤧", "😷", "💩", "😖", "👎", "🤐", "🙄", "🤨", "🥀"],
    "neutral": ["😐", "🤔", "😶", "🧐", "💬", "📝", "👍", "🆗", "👀", "☕", "📊", "💡"]
}


class EmojiDatasetEngine:
    """
    Parses and indexes Emoji Sentiment Data and Emojitracker CSVs.
    Computes sentiment polarity and ranks emojis for each emotion.
    """

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        self.sentiment_csv = os.path.join(base_dir, "Emoji_Sentiment_Data_v1.0.csv")
        self.tracker_csv = os.path.join(base_dir, "Emojitracker_20150604.csv")
        
        self.popularity: Dict[str, int] = {}
        self.sentiment_records: Dict[str, Dict[str, Any]] = {}
        self.ranked_emojis: Dict[str, List[str]] = {k: list(v) for k, v in CURATED_SEEDS.items()}
        
        self._load_datasets()

    def _load_datasets(self):
        # 1. Load Twitter Emojitracker frequencies
        if os.path.exists(self.tracker_csv):
            try:
                with open(self.tracker_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            self.popularity[row["Emoji"]] = int(row["Occurrences"])
                        except Exception:
                            pass
            except Exception as e:
                print(f"[EmojiDatasetEngine] Warning loading tracker CSV: {e}")

        # 2. Keywords in Unicode names for each emotion category
        EMOTION_KEYWORDS = {
            "happy": ["SMILE", "SMILING", "JOY", "GRIN", "LAUGH", "HEART", "CELEBRAT", "PARTY", "SUN", "BLUSH", "KISS", "STAR", "GIFT", "CUPID", "CHEER"],
            "sad": ["CRY", "CRYING", "SAD", "TEAR", "BROKEN HEART", "DISAPPOINTED", "PENSIVE", "DEJECTED", "WORRIED", "CONFOUNDED"],
            "angry": ["ANGRY", "RAGE", "POUTING", "IMP", "DEVIL", "FUME", "FIRE", "SKULL", "MAD"],
            "surprise": ["SURPRISE", "ASTONISHED", "SHOCKED", "SCREAM", "OPEN MOUTH", "FLUSHED", "EXCLAMATION"],
            "fear": ["FEAR", "SCARED", "ANXIOUS", "COLD SWEAT", "TERRIFIED", "SHIVER", "GHOST"],
            "disgust": ["DISGUST", "NAUSEATED", "VOMIT", "UNAMUSED", "SICK", "POOP"],
            "neutral": ["NEUTRAL", "EXPRESSIONLESS", "THINKING", "SPEECH", "BALL", "BOOK", "CALENDAR", "OK"]
        }

        # 3. Load Emoji Sentiment Data
        if os.path.exists(self.sentiment_csv):
            candidates: Dict[str, List[tuple]] = {k: [] for k in EMOTION_KEYWORDS}
            allowed_blocks = {"Emoticons", "Miscellaneous Symbols and Pictographs", "Dingbats", "Miscellaneous Symbols"}
            
            try:
                with open(self.sentiment_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        emoji = row["Emoji"]
                        block = row.get("Unicode block", "")
                        if block not in allowed_blocks:
                            continue

                        name = row.get("Unicode name", "").upper()
                        try:
                            pos = int(row["Positive"])
                            neu = int(row["Neutral"])
                            neg = int(row["Negative"])
                            total = pos + neu + neg
                            if total < 5:
                                continue

                            pop = self.popularity.get(emoji, total)
                            pop_weight = math.log10(max(10, pop))
                            pos_ratio = pos / total
                            neg_ratio = neg / total
                            neu_ratio = neu / total

                            self.sentiment_records[emoji] = {
                                "name": name,
                                "pos": pos_ratio,
                                "neu": neu_ratio,
                                "neg": neg_ratio,
                                "popularity": pop
                            }

                            for emotion, kw_list in EMOTION_KEYWORDS.items():
                                if any(kw in name for kw in kw_list):
                                    if emotion == "happy" and pos_ratio >= 0.35:
                                        score = pos_ratio * 3.0 + pop_weight * 0.4
                                        candidates[emotion].append((emoji, score))
                                    elif emotion in ["sad", "angry", "fear", "disgust"] and neg_ratio >= 0.25:
                                        score = neg_ratio * 3.0 + pop_weight * 0.4
                                        candidates[emotion].append((emoji, score))
                                    elif emotion == "neutral" and neu_ratio >= 0.30:
                                        score = neu_ratio * 2.0 + pop_weight * 0.4
                                        candidates[emotion].append((emoji, score))
                                    elif emotion == "surprise":
                                        score = (pos_ratio + neg_ratio) * 1.5 + pop_weight * 0.4
                                        candidates[emotion].append((emoji, score))
                        except Exception:
                            pass

                # Build final ranked lists blending curated top seeds + dataset emojis
                for emotion in EMOTION_KEYWORDS:
                    sorted_cand = sorted(candidates[emotion], key=lambda x: x[1], reverse=True)
                    ranked = []
                    seen = set()

                    # Prioritize top 3 curated seeds
                    for em in CURATED_SEEDS[emotion][:3]:
                        if em not in seen:
                            ranked.append(em)
                            seen.add(em)

                    # Add dataset-driven candidates
                    for em, _ in sorted_cand:
                        if em not in seen:
                            ranked.append(em)
                            seen.add(em)
                        if len(ranked) >= 12:
                            break

                    # Fill remainder from curated
                    for em in CURATED_SEEDS[emotion]:
                        if em not in seen:
                            ranked.append(em)
                            seen.add(em)

                    self.ranked_emojis[emotion] = ranked

            except Exception as e:
                print(f"[EmojiDatasetEngine] Warning parsing sentiment CSV: {e}")

    def get_recommendations(
        self,
        primary_emotion: str,
        secondary_emotion: Optional[str] = None,
        secondary_conf: float = 0.0,
        top_n: int = 5
    ) -> List[str]:
        """
        Returns ranked emoji recommendations. If secondary emotion has meaningful
        confidence (> 20%), blends secondary recommendations into the results.
        """
        primary = primary_emotion.lower()
        primary_list = self.ranked_emojis.get(primary, CURATED_SEEDS.get(primary, ["🙂"]))

        if secondary_emotion and secondary_conf >= 20.0 and secondary_emotion.lower() != primary:
            sec = secondary_emotion.lower()
            sec_list = self.ranked_emojis.get(sec, CURATED_SEEDS.get(sec, []))
            
            # Blend: top 3 primary + top 2 secondary
            blended = []
            seen = set()
            for em in primary_list[:3]:
                if em not in seen:
                    blended.append(em)
                    seen.add(em)
            for em in sec_list:
                if em not in seen:
                    blended.append(em)
                    seen.add(em)
                if len(blended) >= top_n:
                    break
            return blended[:top_n]

        return primary_list[:top_n]


class TextEmotionAI:
    """
    Singleton AI Engine for Text Emotion Classification and Emoji Recommendation.
    Loads transformer model asynchronously or on-demand.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TextEmotionAI, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, base_dir: Optional[str] = None):
        if self._initialized:
            return
        
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.dataset_engine = EmojiDatasetEngine(self.base_dir)
        self.classifier = None
        self.is_loading = False
        self.model_loaded = False
        self.load_error = None
        self._load_lock = threading.Lock()
        
        # Start async loading immediately
        self._init_model_async()
        self._initialized = True

    def _init_model_async(self):
        def _load_worker():
            with self._load_lock:
                if self.model_loaded:
                    return
                self.is_loading = True
                try:
                    from transformers import pipeline
                    model_id = "j-hartmann/emotion-english-distilroberta-base"
                    self.classifier = pipeline(
                        "text-classification",
                        model=model_id,
                        top_k=None,
                        device=-1
                    )
                    self.model_loaded = True
                    self.load_error = None
                except Exception as e:
                    self.load_error = str(e)
                    print(f"[TextEmotionAI] Error loading model: {e}")
                finally:
                    self.is_loading = False

        thread = threading.Thread(target=_load_worker, daemon=True)
        thread.start()

    def ensure_model_ready(self, timeout_sec: float = 10.0):
        """Blocks until the model is loaded or timeout expires."""
        if self.model_loaded:
            return True
        import time
        start = time.time()
        while self.is_loading:
            time.sleep(0.05)
            if (time.time() - start) > timeout_sec:
                break
        return self.model_loaded

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyzes input sentence and returns:
          - primary_emotion
          - primary_confidence (0-100)
          - primary_emoji
          - secondary_emotion & secondary_confidence
          - distribution: {emotion: percentage}
          - recommended_emojis: list of top emojis
          - sentiment_type: Positive, Negative, Neutral
          - explanation: textual breakdown
        """
        cleaned = text.strip() if text else ""
        if not cleaned:
            return {
                "text": text,
                "primary_emotion": "neutral",
                "primary_confidence": 100.0,
                "primary_emoji": "😐",
                "display_name": "NEUTRAL",
                "color": "#6B7280",
                "secondary_emotion": None,
                "secondary_confidence": 0.0,
                "distribution": {"neutral": 100.0},
                "recommended_emojis": ["😐", "🤔", "😶", "💬", "👍"],
                "sentiment_type": "Neutral",
                "explanation": "Empty text provided. Defaulting to neutral state."
            }

        # If model is ready, run transformer pipeline
        if self.model_loaded and self.classifier is not None:
            try:
                preds = self.classifier(cleaned)[0]
                # Sort descending by score
                preds.sort(key=lambda x: x["score"], reverse=True)

                distribution = {}
                for p in preds:
                    std_label = EMOTION_MAPPING.get(p["label"], p["label"])
                    distribution[std_label] = round(p["score"] * 100.0, 1)

                top_item = preds[0]
                primary_label = EMOTION_MAPPING.get(top_item["label"], top_item["label"])
                primary_conf = round(top_item["score"] * 100.0, 1)

                sec_item = preds[1] if len(preds) > 1 else None
                secondary_label = EMOTION_MAPPING.get(sec_item["label"], sec_item["label"]) if sec_item else None
                secondary_conf = round(sec_item["score"] * 100.0, 1) if sec_item else 0.0

                emojis = self.dataset_engine.get_recommendations(
                    primary_label,
                    secondary_label,
                    secondary_conf,
                    top_n=5
                )

                meta = EMOTION_DISPLAY.get(primary_label, EMOTION_DISPLAY["neutral"])

                # Generate brief explanation
                if primary_conf > 85.0:
                    exp = f"High confidence detection of {meta['name']} ({primary_conf}%)."
                elif secondary_label and secondary_conf > 25.0:
                    sec_meta = EMOTION_DISPLAY.get(secondary_label, EMOTION_DISPLAY["neutral"])
                    exp = f"Predominantly {meta['name']} ({primary_conf}%) with subtle nuance of {sec_meta['name']} ({secondary_conf}%)."
                else:
                    exp = f"Detected {meta['name']} ({primary_conf}%)."

                return {
                    "text": cleaned,
                    "primary_emotion": primary_label,
                    "primary_confidence": primary_conf,
                    "primary_emoji": meta["emoji"],
                    "display_name": meta["name"],
                    "color": meta["color"],
                    "secondary_emotion": secondary_label,
                    "secondary_confidence": secondary_conf,
                    "distribution": distribution,
                    "recommended_emojis": emojis,
                    "sentiment_type": meta["sentiment"],
                    "explanation": exp
                }

            except Exception as e:
                print(f"[TextEmotionAI] Inference error, using fallback: {e}")

        # Fallback if model not ready or inference error
        return self._rule_based_fallback(cleaned)

    def _rule_based_fallback(self, text: str) -> Dict[str, Any]:
        """Robust fallback heuristic when model is still loading."""
        lower = text.lower()
        if any(w in lower for w in ["not happy", "sad", "unhappy", "cry", "depressed", "heartbroken", "terrible", "bad"]):
            emotion = "sad"
        elif any(w in lower for w in ["angry", "mad", "furious", "hate", "rage", "pissed", "annoyed"]):
            emotion = "angry"
        elif any(w in lower for w in ["happy", "great", "love", "joy", "excited", "glad", "awesome", "wonderful"]):
            emotion = "happy"
        elif any(w in lower for w in ["wow", "shock", "surprise", "unbelievable", "omg"]):
            emotion = "surprise"
        elif any(w in lower for w in ["fear", "scared", "terrified", "panic", "afraid"]):
            emotion = "fear"
        elif any(w in lower for w in ["disgust", "gross", "sick", "nasty", "rotten"]):
            emotion = "disgust"
        else:
            emotion = "neutral"

        meta = EMOTION_DISPLAY.get(emotion, EMOTION_DISPLAY["neutral"])
        emojis = self.dataset_engine.get_recommendations(emotion, top_n=5)
        return {
            "text": text,
            "primary_emotion": emotion,
            "primary_confidence": 75.0,
            "primary_emoji": meta["emoji"],
            "display_name": meta["name"],
            "color": meta["color"],
            "secondary_emotion": None,
            "secondary_confidence": 0.0,
            "distribution": {emotion: 75.0, "neutral": 25.0},
            "recommended_emojis": emojis,
            "sentiment_type": meta["sentiment"],
            "explanation": f"Quick-heuristic matched {meta['name']} (model loading in background)."
        }


# Singleton accessor
_default_engine = None

def get_text_ai(base_dir: Optional[str] = None) -> TextEmotionAI:
    global _default_engine
    if _default_engine is None:
        _default_engine = TextEmotionAI(base_dir)
    return _default_engine
