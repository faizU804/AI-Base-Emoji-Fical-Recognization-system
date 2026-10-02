"""
upload_emotion.py — High-resolution portrait photo emotion detection.

Features:
  - DeepFace CNN emotion inference on uploaded image files
  - OpenCV Haar Cascade facial detection with sci-fi target bounding boxes
  - Futuristic dark cosmic UI matching the unified design system
  - Emotion-reactive telemetry card with confidence scoring
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

import compat
compat.apply()

import cv2
from PIL import Image, ImageTk
from deepface import DeepFace

from ui_theme import Theme, CyberButton
from emotion_utils import get_emoji, get_emotion_name


class UploadEmotion:
    """
    Image Upload & Facial Emotion Analysis GUI Screen.
    """

    def __init__(self, parent, go_back):
        self.parent = parent
        self.go_back = go_back

        base_dir = os.path.dirname(os.path.abspath(__file__))
        cascade_path = os.path.join(base_dir, "haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self.build_ui()

    def build_ui(self):
        for widget in self.parent.winfo_children():
            widget.destroy()

        self.parent.configure(bg=Theme.BG_ROOT)

        # 1. Top Header Bar
        header = tk.Frame(self.parent, bg=Theme.BG_SURFACE_ALT, height=65)
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
            text="🖼️ PORTRAIT VISION AI :: IMAGE ANALYSIS",
            font=(Theme.FONT_FAMILY, 13, "bold"),
            bg=Theme.BG_SURFACE_ALT,
            fg=Theme.TEXT_PRIMARY
        ).pack(anchor="w")

        tk.Label(
            title_box,
            text="High-Resolution Facial Tracking • DeepFace Expression CNN • Bounding Reticles",
            font=(Theme.FONT_FAMILY, 8),
            bg=Theme.BG_SURFACE_ALT,
            fg=Theme.TEXT_SECONDARY
        ).pack(anchor="w")

        # Telemetry Status Badge
        tk.Label(
            header_content,
            text="● ENGINE READY",
            font=(Theme.FONT_CODE, 9, "bold"),
            bg=Theme.BG_SURFACE_ALT,
            fg=Theme.SUCCESS
        ).pack(side="right", pady=18)

        # 2. Main 2-Column Container
        container = tk.Frame(self.parent, bg=Theme.BG_ROOT)
        container.pack(fill="both", expand=True, padx=22, pady=16)
        container.grid_columnconfigure(0, weight=5)  # Left: Image Viewport
        container.grid_columnconfigure(1, weight=4)  # Right: Detection Analytics
        container.grid_rowconfigure(0, weight=1)

        # Left Column: Image Display Card
        left_card = tk.Frame(
            container,
            bg=Theme.BG_SURFACE,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=Theme.BORDER_SUBTLE
        )
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        left_inner = tk.Frame(left_card, bg=Theme.BG_SURFACE)
        left_inner.pack(fill="both", expand=True, padx=18, pady=18)

        tk.Label(
            left_inner,
            text="Portrait Photo Viewport",
            font=(Theme.FONT_FAMILY, 12, "bold"),
            bg=Theme.BG_SURFACE,
            fg=Theme.TEXT_PRIMARY
        ).pack(anchor="w")

        tk.Label(
            left_inner,
            text="Upload any portrait photo (JPG, PNG, WEBP). Faces will be detected with bounding targets.",
            font=(Theme.FONT_FAMILY, 8),
            bg=Theme.BG_SURFACE,
            fg=Theme.TEXT_MUTED
        ).pack(anchor="w", pady=(2, 10))

        # Viewport Frame
        viewport_frame = tk.Frame(
            left_inner,
            bg=Theme.BG_INPUT,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=Theme.BORDER_LIGHT
        )
        viewport_frame.pack(fill="both", expand=True, pady=(0, 12))

        self.image_label = tk.Label(
            viewport_frame,
            text="📸\n\nNo Image Loaded\nClick 'Select Image' below to analyze a portrait",
            font=(Theme.FONT_FAMILY, 11),
            bg=Theme.BG_INPUT,
            fg=Theme.TEXT_SECONDARY,
            justify="center"
        )
        self.image_label.pack(fill="both", expand=True, padx=10, pady=10)

        # Controls Row
        ctrl_row = tk.Frame(left_inner, bg=Theme.BG_SURFACE)
        ctrl_row.pack(fill="x")

        CyberButton(
            ctrl_row,
            text="Select Image File",
            icon="📁",
            bg_color=Theme.SECONDARY,
            hover_color=Theme.SECONDARY_HOVER,
            command=self.select_image,
            padx=16,
            pady=8
        ).pack(side="left")

        # Right Column: Analytics Card
        right_card = tk.Frame(
            container,
            bg=Theme.BG_SURFACE,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=Theme.BORDER_SUBTLE
        )
        right_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        right_inner = tk.Frame(right_card, bg=Theme.BG_SURFACE)
        right_inner.pack(fill="both", expand=True, padx=20, pady=20)

        tk.Label(
            right_inner,
            text="Inference Telemetry",
            font=(Theme.FONT_FAMILY, 12, "bold"),
            bg=Theme.BG_SURFACE,
            fg=Theme.TEXT_PRIMARY
        ).pack(anchor="w")

        tk.Label(
            right_inner,
            text="Dominant facial expression detected by DeepFace neural network.",
            font=(Theme.FONT_FAMILY, 8),
            bg=Theme.BG_SURFACE,
            fg=Theme.TEXT_MUTED
        ).pack(anchor="w", pady=(2, 16))

        # Hero Emotion Box
        self.hero_box = tk.Frame(
            right_inner,
            bg=Theme.BG_INPUT,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=Theme.BORDER_LIGHT,
            padx=16,
            pady=22
        )
        self.hero_box.pack(fill="x", pady=(0, 16))

        self.emoji_label = tk.Label(
            self.hero_box,
            text="🖼️",
            font=(Theme.FONT_EMOJI, 48),
            bg=Theme.BG_INPUT,
            fg=Theme.SECONDARY
        )
        self.emoji_label.pack()

        self.emotion_label = tk.Label(
            self.hero_box,
            text="NO IMAGE LOADED",
            font=(Theme.FONT_FAMILY, 16, "bold"),
            bg=Theme.BG_INPUT,
            fg=Theme.TEXT_PRIMARY
        )
        self.emotion_label.pack(pady=(6, 3))

        self.confidence_label = tk.Label(
            self.hero_box,
            text="Confidence: --%",
            font=(Theme.FONT_CODE, 9),
            bg=Theme.BG_INPUT,
            fg=Theme.TEXT_SECONDARY
        )
        self.confidence_label.pack()

        # Telemetry Pill
        self.status_pill = tk.Label(
            right_inner,
            text="STATUS: AWAITING INPUT",
            font=(Theme.FONT_CODE, 8, "bold"),
            bg=Theme.BG_INPUT,
            fg=Theme.TEXT_SECONDARY,
            padx=10,
            pady=5
        )
        self.status_pill.pack(anchor="w", pady=(0, 16))

        # Instructions Box
        info_box = tk.Frame(
            right_inner,
            bg=Theme.BG_INPUT,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=Theme.BORDER_SUBTLE,
            padx=12,
            pady=12
        )
        info_box.pack(fill="x")

        tk.Label(
            info_box,
            text="ℹ️ Analysis Details:",
            font=(Theme.FONT_FAMILY, 9, "bold"),
            bg=Theme.BG_INPUT,
            fg=Theme.TEXT_HIGHLIGHT
        ).pack(anchor="w")

        tk.Label(
            info_box,
            text="• Evaluates 7 distinct facial expressions.\n"
                 "• Calculates confidence probability scores.\n"
                 "• Outlines detected faces with target reticles.",
            font=(Theme.FONT_FAMILY, 8),
            bg=Theme.BG_INPUT,
            fg=Theme.TEXT_MUTED,
            justify="left"
        ).pack(anchor="w", pady=(4, 0))

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select Portrait Photo",
            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp *.webp"
                )
            ]
        )

        if not path:
            return

        try:
            image = cv2.imread(path)
            if image is None:
                messagebox.showerror("Error", "Could not open selected image.")
                return

            # Face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                1.1,
                5,
                minSize=(60, 60)
            )

            # Draw sci-fi bounding targets
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (248, 189, 56), 2)
                d = 12
                cv2.line(image, (x, y), (x + d, y), (255, 255, 255), 3)
                cv2.line(image, (x, y), (x, y + d), (255, 255, 255), 3)
                cv2.line(image, (x + w, y + h), (x + w - d, y + h), (255, 255, 255), 3)
                cv2.line(image, (x + w, y + h), (x + w, y + h - d), (255, 255, 255), 3)

            # AI emotion inference
            result = DeepFace.analyze(
                image,
                actions=["emotion"],
                detector_backend="opencv",
                enforce_detection=False,
                silent=True
            )

            if isinstance(result, list):
                result = result[0]

            emotion = result["dominant_emotion"]
            confidence = result["emotion"][emotion]
            emoji = get_emoji(emotion)
            name = get_emotion_name(emotion)

            meta = Theme.EMOTION_THEMES.get(emotion.lower(), Theme.EMOTION_THEMES["neutral"])

            # Convert to RGB & resize for preview viewport
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            pil_image.thumbnail((540, 380))
            photo = ImageTk.PhotoImage(pil_image)

            self.image_label.config(image=photo, text="")
            self.image_label.image = photo

            # Update Analytics
            self.emoji_label.config(text=emoji)
            self.emotion_label.config(text=name, fg=meta["color"])
            self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
            self.hero_box.config(highlightbackground=meta["color"])

            face_str = f"FACES DETECTED: {len(faces)}"
            self.status_pill.config(text=f"STATUS: SUCCESS • {face_str}", fg=Theme.SUCCESS)

        except Exception as error:
            messagebox.showerror("Detection Error", str(error))