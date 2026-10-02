"""
camera_emotion.py — Live webcam emotion recognition.

Features:
  - Multi-backend camera auto-discovery (DirectShow, Default, Media Foundation)
  - OpenCV Haar Cascade for real-time face detection with green bounding boxes
  - Sci-Fi HUD overlay: holographic scanning beam, corner brackets, telemetry
  - Non-blocking background thread for DeepFace CNN emotion inference
  - Inference throttling (every 0.6s) to ensure smooth 30+ FPS video rendering
  - 5-frame rolling window majority-vote temporal smoothing (eliminates flickering)
  - Emotion-reactive visual theming (HUD accents update with detected emotion)
  - Responsive layout fitting 1000x750 window without clipping controls
"""

import os
import sys
import time
import threading
from collections import deque
import tkinter as tk
from tkinter import messagebox

import compat
compat.apply()

import cv2
from PIL import Image, ImageTk
from deepface import DeepFace

from ui_theme import Theme, CyberButton, CameraHUDOverlay
from emotion_utils import get_emoji, get_emotion_name


class CameraEmotion:
    """
    Live Webcam Facial Emotion Recognition GUI Screen.
    """

    # Minimum seconds between DeepFace inferences to prevent CPU overload
    INFERENCE_INTERVAL = 0.6

    # Temporal smoothing: majority vote over recent predictions
    SMOOTHING_WINDOW = 5

    # Confidence threshold below which emotion is flagged as Uncertain
    CONFIDENCE_THRESHOLD = 35.0

    # Display dimensions for camera preview
    PREVIEW_WIDTH = 580
    PREVIEW_HEIGHT = 435

    def __init__(self, parent, back_callback):
        self.parent = parent
        self.back_callback = back_callback

        # Camera state
        self.cap = None
        self.running = False
        self.is_paused = False
        self.after_id = None
        self.camera_index = None

        # Emotion inference state
        self.last_inference_time = 0
        self.inference_running = False
        self.current_emotion = None
        self.current_confidence = 0.0
        self.current_emoji = ""
        self.prediction_history = deque(maxlen=self.SMOOTHING_WINDOW)

        # Sci-Fi HUD Overlay
        self.hud_overlay = CameraHUDOverlay()

        # Load Face Cascade for fast real-time face detection
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cascade_path = os.path.join(base_dir, "haarcascade_frontalface_default.xml")
        self.face_cascade = None
        if os.path.exists(cascade_path):
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                self.face_cascade = cascade

        # Clear parent of previous screen widgets
        for widget in self.parent.winfo_children():
            widget.destroy()

        self.parent.configure(bg=Theme.BG_ROOT)

        # Container Frame
        self.window = tk.Frame(self.parent, bg=Theme.BG_ROOT)
        self.window.pack(fill="both", expand=True)

        # Loading / Searching Screen
        self.loading_frame = tk.Frame(self.window, bg=Theme.BG_ROOT)
        self.loading_frame.pack(expand=True, fill="both")

        self.loading_label = tk.Label(
            self.loading_frame,
            text="🎥 Initializing Optical Sensor...",
            font=(Theme.FONT_FAMILY, 18, "bold"),
            bg=Theme.BG_ROOT,
            fg=Theme.TEXT_HIGHLIGHT
        )
        self.loading_label.pack(expand=True, pady=(150, 10))

        self.loading_sub = tk.Label(
            self.loading_frame,
            text="Probing video capture devices (DirectShow / MSMF)...",
            font=(Theme.FONT_CODE, 9),
            bg=Theme.BG_ROOT,
            fg=Theme.TEXT_MUTED
        )
        self.loading_sub.pack(pady=(0, 20))

        self.parent.update_idletasks()

        # Begin camera auto-discovery
        self.open_available_camera()

    # ==================================================
    # MULTI-BACKEND CAMERA DISCOVERY
    # ==================================================

    def open_available_camera(self):
        """
        Attempts to open available camera across indices and backends.
        Tries DirectShow first (recommended for Windows), then Default/MSMF.
        """
        candidate_indices = [0, 1, 2]
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_ANY, "Default"),
            (cv2.CAP_MSMF, "Media Foundation")
        ]

        found_cap = None
        found_index = None

        for index in candidate_indices:
            for backend_flag, backend_name in backends:
                cap = None
                try:
                    cap = cv2.VideoCapture(index, backend_flag)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                        ret, test_frame = cap.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            found_cap = cap
                            found_index = index
                            break
                    if cap is not None:
                        cap.release()
                except Exception:
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass

            if found_cap is not None:
                break

        if found_cap is not None:
            self.cap = found_cap
            self.camera_index = found_index
            self.show_camera_screen()
            return

        # No Camera Found Screen
        for widget in self.window.winfo_children():
            widget.destroy()

        err_box = tk.Frame(
            self.window,
            bg=Theme.BG_SURFACE,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=Theme.BORDER_SUBTLE,
            padx=40,
            pady=35
        )
        err_box.pack(expand=True)

        tk.Label(
            err_box,
            text="⚠️",
            font=(Theme.FONT_EMOJI, 44),
            bg=Theme.BG_SURFACE
        ).pack(pady=(0, 10))

        tk.Label(
            err_box,
            text="Optical Sensor Offline",
            font=(Theme.FONT_FAMILY, 16, "bold"),
            bg=Theme.BG_SURFACE,
            fg=Theme.DANGER
        ).pack(pady=(0, 8))

        tk.Label(
            err_box,
            text="Could not initialize a webcam stream.\n"
                 "Please check:\n"
                 "1. Camera is plugged in or built-in webcam is enabled.\n"
                 "2. Windows Camera Privacy Settings allow desktop app access.\n"
                 "3. No other application is locking the camera stream.",
            font=(Theme.FONT_FAMILY, 10),
            bg=Theme.BG_SURFACE,
            fg=Theme.TEXT_SECONDARY,
            justify="center"
        ).pack(pady=(0, 25))

        btn_row = tk.Frame(err_box, bg=Theme.BG_SURFACE)
        btn_row.pack()

        CyberButton(
            btn_row,
            text="Retry Connection",
            icon="🔄",
            bg_color=Theme.PRIMARY,
            command=self.retry_camera,
            padx=16,
            pady=8
        ).pack(side="left", padx=8)

        CyberButton(
            btn_row,
            text="Back to Home",
            icon="←",
            bg_color=Theme.BG_SURFACE_ALT,
            command=self.go_back,
            padx=16,
            pady=8
        ).pack(side="left", padx=8)

    def retry_camera(self):
        for widget in self.window.winfo_children():
            widget.destroy()

        self.loading_frame = tk.Frame(self.window, bg=Theme.BG_ROOT)
        self.loading_frame.pack(expand=True, fill="both")

        self.loading_label = tk.Label(
            self.loading_frame,
            text="🎥 Retrying webcam connection...",
            font=(Theme.FONT_FAMILY, 18, "bold"),
            bg=Theme.BG_ROOT,
            fg=Theme.TEXT_HIGHLIGHT
        )
        self.loading_label.pack(expand=True, pady=(150, 10))

        self.parent.update_idletasks()
        self.open_available_camera()

    # ==================================================
    # MAIN CAMERA INTERFACE
    # ==================================================

    def show_camera_screen(self):
        for widget in self.window.winfo_children():
            widget.destroy()

        # 1. Top Header Bar
        header = tk.Frame(self.window, bg=Theme.BG_SURFACE_ALT, height=65)
        header.pack(fill="x")
        header.pack_propagate(False)

        header_content = tk.Frame(header, bg=Theme.BG_SURFACE_ALT)
        header_content.pack(fill="both", expand=True, padx=22)

        # Back Button
        back_btn = CyberButton(
            header_content,
            text="Back to Home",
            icon="←",
            bg_color=Theme.BG_SURFACE,
            hover_color=Theme.BG_CARD_HOVER,
            command=self.go_back,
            padx=12,
            pady=5
        )
        back_btn.pack(side="left", pady=14)

        # Title & Subtitle
        title_box = tk.Frame(header_content, bg=Theme.BG_SURFACE_ALT)
        title_box.pack(side="left", padx=20, pady=10)

        tk.Label(
            title_box,
            text="🎥 FACIAL VISION AI :: REAL-TIME TRACKING",
            font=(Theme.FONT_FAMILY, 13, "bold"),
            bg=Theme.BG_SURFACE_ALT,
            fg=Theme.TEXT_PRIMARY
        ).pack(anchor="w")

        tk.Label(
            title_box,
            text="OpenCV Face Tracking • DeepFace Expression CNN • 5-Frame Temporal Smoothing",
            font=(Theme.FONT_FAMILY, 8),
            bg=Theme.BG_SURFACE_ALT,
            fg=Theme.TEXT_SECONDARY
        ).pack(anchor="w")

        # Live Status Telemetry Badge
        self.live_badge = tk.Label(
            header_content,
            text=f"● STREAM_0 ACTIVE",
            font=(Theme.FONT_CODE, 9, "bold"),
            bg=Theme.BG_SURFACE_ALT,
            fg=Theme.SUCCESS
        )
        self.live_badge.pack(side="right", pady=18)

        # 2. Main Body Container
        body = tk.Frame(self.window, bg=Theme.BG_ROOT)
        body.pack(fill="both", expand=True, padx=20, pady=14)

        # Left Column: Video Viewport
        video_col = tk.Frame(body, bg=Theme.BG_ROOT)
        video_col.pack(side="left", padx=(0, 16))

        # Sci-Fi Video Canvas with HUD Overlay
        self.video_container = tk.Frame(
            video_col,
            bg=Theme.BG_INPUT,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=Theme.BORDER_LIGHT
        )
        self.video_container.pack()

        # Canvas for compositing video frame + HUD scanning overlay
        self.video_canvas = tk.Canvas(
            self.video_container,
            bg="black",
            width=self.PREVIEW_WIDTH,
            height=self.PREVIEW_HEIGHT,
            bd=0,
            highlightthickness=0
        )
        self.video_canvas.pack()

        # Video Controls Row
        ctrl_row = tk.Frame(video_col, bg=Theme.BG_ROOT)
        ctrl_row.pack(fill="x", pady=(10, 0))

        self.pause_btn = CyberButton(
            ctrl_row,
            text="Pause Stream",
            icon="⏸",
            bg_color=Theme.BG_SURFACE_ALT,
            command=self.toggle_pause,
            padx=12,
            pady=5
        )
        self.pause_btn.pack(side="left")

        self.status_label = tk.Label(
            ctrl_row,
            text="Initializing stream...",
            font=(Theme.FONT_CODE, 8),
            bg=Theme.BG_ROOT,
            fg=Theme.TEXT_SECONDARY
        )
        self.status_label.pack(side="left", padx=12)

        # Right Column: Real-Time Emotion Analytics Card
        analytics_card = tk.Frame(
            body,
            bg=Theme.BG_SURFACE,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=Theme.BORDER_SUBTLE
        )
        analytics_card.pack(side="left", fill="both", expand=True)

        card_inner = tk.Frame(analytics_card, bg=Theme.BG_SURFACE)
        card_inner.pack(fill="both", expand=True, padx=20, pady=20)

        tk.Label(
            card_inner,
            text="Live Facial Telemetry",
            font=(Theme.FONT_FAMILY, 12, "bold"),
            bg=Theme.BG_SURFACE,
            fg=Theme.TEXT_PRIMARY
        ).pack(anchor="w")

        tk.Label(
            card_inner,
            text="DeepFace CNN infers 7 facial expression classes with majority smoothing.",
            font=(Theme.FONT_FAMILY, 8),
            bg=Theme.BG_SURFACE,
            fg=Theme.TEXT_MUTED
        ).pack(anchor="w", pady=(2, 16))

        # Big Hero Emotion Display
        self.hero_box = tk.Frame(
            card_inner,
            bg=Theme.BG_INPUT,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=Theme.BORDER_LIGHT,
            padx=16,
            pady=20
        )
        self.hero_box.pack(fill="x", pady=(0, 16))

        self.emoji_label = tk.Label(
            self.hero_box,
            text="👤",
            font=(Theme.FONT_EMOJI, 48),
            bg=Theme.BG_INPUT,
            fg=Theme.TEXT_HIGHLIGHT
        )
        self.emoji_label.pack()

        self.emotion_name_label = tk.Label(
            self.hero_box,
            text="Looking for face...",
            font=(Theme.FONT_FAMILY, 17, "bold"),
            bg=Theme.BG_INPUT,
            fg=Theme.TEXT_HIGHLIGHT
        )
        self.emotion_name_label.pack(pady=(5, 3))

        self.confidence_label = tk.Label(
            self.hero_box,
            text="Confidence: --%",
            font=(Theme.FONT_CODE, 9),
            bg=Theme.BG_INPUT,
            fg=Theme.TEXT_SECONDARY
        )
        self.confidence_label.pack()

        # Telemetry Pills Row
        telem_row = tk.Frame(card_inner, bg=Theme.BG_SURFACE)
        telem_row.pack(fill="x", pady=(0, 16))

        self.face_badge = tk.Label(
            telem_row,
            text="FACES: 0",
            font=(Theme.FONT_CODE, 8, "bold"),
            bg=Theme.BG_INPUT,
            fg=Theme.TEXT_SECONDARY,
            padx=8,
            pady=3
        )
        self.face_badge.pack(side="left", padx=(0, 8))

        self.smooth_badge = tk.Label(
            telem_row,
            text="SMOOTHING: 5-FRAME",
            font=(Theme.FONT_CODE, 8, "bold"),
            bg=Theme.BG_INPUT,
            fg=Theme.TEXT_HIGHLIGHT,
            padx=8,
            pady=3
        )
        self.smooth_badge.pack(side="left")

        # Pro Tips Card
        tips_box = tk.Frame(
            card_inner,
            bg=Theme.BG_INPUT,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=Theme.BORDER_SUBTLE,
            padx=12,
            pady=10
        )
        tips_box.pack(fill="x", pady=(0, 15))

        tk.Label(
            tips_box,
            text="💡 Best Tracking Conditions:",
            font=(Theme.FONT_FAMILY, 9, "bold"),
            bg=Theme.BG_INPUT,
            fg=Theme.TEXT_HIGHLIGHT
        ).pack(anchor="w")

        tk.Label(
            tips_box,
            text="• Face the webcam with clear front lighting.\n"
                 "• Maintain standard distance (1.5 - 3 feet).\n"
                 "• Expressions are smoothed across a 5-frame window.",
            font=(Theme.FONT_FAMILY, 8),
            bg=Theme.BG_INPUT,
            fg=Theme.TEXT_MUTED,
            justify="left"
        ).pack(anchor="w", pady=(3, 0))

        # Start Camera Loop
        self.running = True
        self.is_paused = False
        self.update_camera()

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.set_text("Resume Stream")
            self.status_label.config(text="Stream paused by user.")
            self.live_badge.config(text="● STREAM PAUSED", fg=Theme.EMOTION_THEMES["surprise"]["color"])
        else:
            self.pause_btn.set_text("Pause Stream")
            self.status_label.config(text="Stream resumed.")
            self.live_badge.config(text=f"● STREAM_{self.camera_index} ACTIVE", fg=Theme.SUCCESS)
            self.update_camera()

    # ==================================================
    # CAMERA FRAME CAPTURE & DETECTION LOOP
    # ==================================================

    def update_camera(self):
        if not self.running or self.is_paused or self.cap is None:
            return

        if not hasattr(self, "video_canvas") or not self.video_canvas.winfo_exists():
            return

        try:
            ret, frame = self.cap.read()
        except Exception:
            ret, frame = False, None

        if not ret or frame is None:
            try:
                self.status_label.config(text="Waiting for camera frame...")
            except Exception:
                return
            if self.running and not self.is_paused:
                self.after_id = self.parent.after(200, self.update_camera)
            return

        # Mirror video preview horizontally
        frame = cv2.flip(frame, 1)

        # Fast Face Detection with OpenCV Cascade
        face_count = 0
        faces = []
        if self.face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )
            face_count = len(faces)
            for (x, y, w, h) in faces:
                # Cyan sci-fi target bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (248, 189, 56), 2)
                # Corner accents on bounding box
                d = 10
                cv2.line(frame, (x, y), (x + d, y), (255, 255, 255), 3)
                cv2.line(frame, (x, y), (x, y + d), (255, 255, 255), 3)
                cv2.line(frame, (x + w, y + h), (x + w - d, y + h), (255, 255, 255), 3)
                cv2.line(frame, (x + w, y + h), (x + w, y + h - d), (255, 255, 255), 3)

        # Asynchronous Throttled Emotion Inference
        now = time.time()
        if (
            face_count > 0
            and not self.inference_running
            and (now - self.last_inference_time) >= self.INFERENCE_INTERVAL
        ):
            self.last_inference_time = now
            frame_copy = frame.copy()
            self.inference_running = True
            thread = threading.Thread(
                target=self._run_inference,
                args=(frame_copy,),
                daemon=True
            )
            thread.start()

        # Update Analytics Cards safely
        try:
            self.face_badge.config(
                text=f"FACES: {face_count}",
                fg=Theme.SUCCESS if face_count == 1 else (Theme.EMOTION_THEMES["surprise"]["color"] if face_count > 1 else Theme.TEXT_MUTED)
            )

            emotion_meta = Theme.EMOTION_THEMES.get(self.current_emotion, Theme.EMOTION_THEMES["neutral"])

            if face_count == 0:
                self.status_label.config(text="Looking for face in frame...")
                self.emoji_label.config(text="👤")
                self.emotion_name_label.config(text="Looking for face...", fg=Theme.TEXT_MUTED)
                self.confidence_label.config(text="Position your face inside the frame")
                self.prediction_history.clear()
                self.current_emotion = None
                self.hero_box.config(highlightbackground=Theme.BORDER_LIGHT)
            else:
                self.status_label.config(text="Tracking active • Analyzing expression")
                self._render_current_emotion()
        except (tk.TclError, Exception):
            return

        # Render Frame to Video Canvas
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            pil_img = pil_img.resize(
                (self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT),
                Image.Resampling.BILINEAR
            )
            photo = ImageTk.PhotoImage(pil_img)

            # Draw image onto canvas
            self.video_canvas.delete("img")
            self.video_canvas.create_image(0, 0, anchor="nw", image=photo, tags="img")
            self.video_canvas.image = photo

            # Draw Sci-Fi HUD overlay (brackets, laser scan, telemetry)
            self.hud_overlay.draw_hud(
                self.video_canvas,
                self.PREVIEW_WIDTH,
                self.PREVIEW_HEIGHT,
                face_count,
                emotion_meta
            )
        except (tk.TclError, Exception):
            return

        # Schedule Next Frame (~33 FPS)
        if self.running and not self.is_paused:
            self.after_id = self.parent.after(33, self.update_camera)

    def _run_inference(self, frame):
        """Runs DeepFace emotion model in background thread."""
        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                detector_backend="opencv",
                enforce_detection=False,
                silent=True
            )
            if isinstance(result, list):
                result = result[0]

            emotion = result["dominant_emotion"]
            confidence = result["emotion"][emotion]

            self.prediction_history.append((emotion, confidence))
            smoothed_emotion, smoothed_conf = self._smooth_predictions()

            self.current_emotion = smoothed_emotion
            self.current_confidence = smoothed_conf
            self.current_emoji = get_emoji(smoothed_emotion)
        except Exception:
            pass
        finally:
            self.inference_running = False

    def _smooth_predictions(self):
        """Majority-vote temporal smoothing over recent inference window."""
        if not self.prediction_history:
            return None, 0.0

        counts = {}
        conf_sums = {}
        for em, conf in self.prediction_history:
            counts[em] = counts.get(em, 0) + 1
            conf_sums[em] = conf_sums.get(em, 0.0) + conf

        best_em = max(counts, key=counts.get)
        avg_conf = conf_sums[best_em] / counts[best_em]
        return best_em, avg_conf

    def _render_current_emotion(self):
        """Updates UI with smoothed emotion prediction and reactive theme."""
        if not self.current_emotion:
            return

        meta = Theme.EMOTION_THEMES.get(self.current_emotion, Theme.EMOTION_THEMES["neutral"])

        if self.current_confidence < self.CONFIDENCE_THRESHOLD:
            self.emoji_label.config(text="🤔")
            self.emotion_name_label.config(text="Uncertain", fg=Theme.EMOTION_THEMES["surprise"]["color"])
            self.confidence_label.config(
                text=f"Confidence too low ({self.current_confidence:.0f}%)"
            )
            self.hero_box.config(highlightbackground=Theme.BORDER_LIGHT)
        else:
            self.emoji_label.config(text=self.current_emoji)
            self.emotion_name_label.config(
                text=get_emotion_name(self.current_emotion),
                fg=meta["color"]
            )
            self.confidence_label.config(
                text=f"Confidence: {self.current_confidence:.0f}%"
            )
            # Emotion-reactive glowing border on hero box
            self.hero_box.config(highlightbackground=meta["color"])

    # ==================================================
    # CLEAN SHUTDOWN & NAVIGATION
    # ==================================================

    def stop_camera(self):
        self.running = False
        if self.after_id is not None:
            try:
                self.parent.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def go_back(self):
        self.stop_camera()
        try:
            self.window.destroy()
        except Exception:
            pass
        self.back_callback()