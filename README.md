# AI-Based Emotion Recognition & Emoji Intelligence System

A production-grade, multi-modal Artificial Intelligence application that combines **Computer Vision** (facial emotion recognition via OpenCV & DeepFace CNN) and **Natural Language Processing** (text sentiment & emotion classification via HuggingFace Transformers) with **Dataset-Driven Emoji Recommendation** using real-world Twitter sentiment and frequency datasets.

---

## Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │                main.py                 │
                      │    (Tkinter Modern Desktop Portal)     │
                      └──────────────────┬─────────────────────┘
                                         │
       ┌─────────────────────────────────┼─────────────────────────────────┐
       ▼                                 ▼                                 ▼
┌───────────────┐              ┌───────────────────┐             ┌───────────────────┐
│ camera_emotion│              │   upload_emotion  │             │   chat_emotion    │
│  (Live Webcam │              │   (Image Upload   │             │   (Text NLP &     │
│   Recognition)│              │     Analysis)     │             │ Emoji Intelligence│
└───────┬───────┘              └─────────┬─────────┘             └─────────┬─────────┘
        │                                │                                 │
        ▼                                ▼                                 ▼
┌──────────────────────────────────────────────────┐             ┌───────────────────┐
│                    DeepFace                      │             │  text_emotion_ai  │
│         (Pre-trained Facial Expression CNN)      │             │  (DistilRoBERTa   │
│             7 Facial Emotion Classes             │             │   Emotion Model)  │
└───────────────────────┬──────────────────────────┘             └─────────┬─────────┘
                        │                                                  │
                        ▼                                                  ▼
             ┌─────────────────────┐                            ┌─────────────────────┐
             │      compat.py      │                            │ EmojiDatasetEngine  │
             │ (TF 2.22 / tf-keras │                            │ (Twitter Sentiment  │
             │    & UTF-8 Shim)    │                            │ & Frequency Ranking)│
             └─────────────────────┘                            └─────────────────────┘
```

---

## Core Capabilities

### 1. Module A — Text Emotion & Emoji Intelligence (`chat_emotion.py` & `text_emotion_ai.py`)
- **Deep Emotion Classification**: Utilizes a fine-tuned DistilRoBERTa model (`j-hartmann/emotion-english-distilroberta-base`) to classify text into 7 emotion categories (`happy`, `sad`, `angry`, `surprise`, `fear`, `disgust`, `neutral`).
- **Nuance & Negation Awareness**: Correctly resolves complex linguistic constructs such as negations (*"I am definitely not happy with this service"* $\rightarrow$ `SAD / GRIEF` $97.6\%$) and mixed emotions (*"I got promoted today!"* $\rightarrow$ `HAPPY / JOY` $55.0\%$ + `SURPRISE / AWE` $38.8\%$).
- **Dataset-Driven Emoji Ranking**: Ranks candidate emojis using:
  - `Emoji_Sentiment_Data_v1.0.csv`: 969 emojis scored for positive, neutral, and negative sentiment ratios from Twitter annotations.
  - `Emojitracker_20150604.csv`: Global Twitter usage counts across 845 emojis.
- **Emotion Blending**: When secondary emotion confidence exceeds 20%, candidate emojis from both primary and secondary emotions are combined.
- **Non-Blocking Asynchronous Processing**: Inference runs on background worker threads with smooth UI progress feedback.
- **Interactive UI**: Includes 8 one-click presets, clipboard copy for recommended emojis, and real-time probability distribution bars.

### 2. Module B — Facial Emotion Recognition (`camera_emotion.py` & `upload_emotion.py`)
- **Live Webcam Stream**:
  - Real-time face detection with OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`).
  - DeepFace CNN emotion inference throttled to $0.6\text{s}$ intervals to maintain smooth video rendering ($\ge 30\text{ FPS}$).
  - 5-frame temporal smoothing (majority-vote rolling window) to eliminate flickering.
  - Confidence threshold ($< 35\%$ displays *Uncertain* state).
  - Handles no-face, single-face, and multi-face states cleanly.
- **Static Image Upload**:
  - Supports PNG, JPG, and JPEG images.
  - Displays bounding boxes, dominant emotion badge, and confidence percentage.

### 3. Centralized Environment Compatibility (`compat.py`)
- **Python 3.14 + TensorFlow 2.22-rc0 + tf-keras 2.15**:
  - In TF 2.22, `tf.__internal__.register_load_context_function` was deprecated/renamed to `register_call_context_function`.
  - `compat.py` provides a centralized runtime monkey-patch that resolves this without requiring downgrading Python or TensorFlow.
- **Windows Console UTF-8 Guard**:
  - Automatically reconfigures `sys.stdout` and `sys.stderr` to UTF-8 with character replacement to prevent `cp1252` encoding crashes on emoji output.
- **Logging & Warning Suppression**:
  - Silences non-critical TensorFlow C++ and oneDNN floating-point warnings (`TF_CPP_MIN_LOG_LEVEL=3`).

---

## Dataset Description

1. **`Emoji_Sentiment_Data_v1.0.csv`**:
   - 969 annotated emojis from Twitter sentiment evaluations.
   - Fields: `Emoji`, `Unicode codepoint`, `Occurrences`, `Position`, `Negative`, `Neutral`, `Positive`, `Unicode name`, `Unicode block`.
   - Used to score emojis based on sentiment polarity and emotional category.
2. **`Emojitracker_20150604.csv`**:
   - 845 emojis with aggregate Twitter usage volume.
   - Fields: `Emoji`, `Unicode codepoint`, `Occurrences`.
   - Used as frequency weights to prioritize widely recognized and supported emojis.

---

## Installation & Setup

### 1. Requirements
- Python 3.10+ (tested on Python 3.14.5)
- Windows / macOS / Linux

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

*Installed dependencies include:*
- `opencv-python`
- `deepface`
- `tensorflow`
- `tf-keras`
- `torch`
- `transformers`
- `pillow`

---

## Running the Application

### Launch the GUI
```bash
python main.py
```

### Run the Comprehensive Test Suite
```bash
python test_system_full.py
```

The automated test suite verifies:
1. Environment and TensorFlow compatibility shims
2. Emotion utilities mapping
3. Emoji dataset engine parsing and ranking
4. Text Emotion AI transformer inference and negation handling
5. Camera emotion architecture parameters
6. DeepFace facial emotion inference on real images
7. ChatEmotion module interface
8. Main application routing and window lifecycle

---

## File Structure

```
d:/saood/New folder/
├── main.py                             # 3D interactive desktop portal entry point
├── ui_theme.py                         # Unified design system, 3D particle canvas & HUD
├── camera_emotion.py                   # Live webcam emotion detector with Sci-Fi HUD
├── upload_emotion.py                   # Image upload emotion detector
├── chat_emotion.py                     # Modern Text Emotion & Emoji Intelligence GUI
├── text_emotion_ai.py                  # Core NLP classifier & dataset ranking engine
├── emotion_utils.py                    # Static emotion and emoji mapping utilities
├── compat.py                           # Centralized compatibility shim (TF / UTF-8)
├── Emoji_Sentiment_Data_v1.0.csv       # Twitter emoji sentiment dataset
├── Emojitracker_20150604.csv           # Twitter emoji usage frequency dataset
├── haarcascade_frontalface_default.xml # OpenCV face detection cascade
├── test_face_happy.jpg                 # Test image for automated validation
├── test_system_full.py                 # Full unit & integration test suite
├── test_phase1.py                      # Phase 1 infrastructure test script
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation
```
