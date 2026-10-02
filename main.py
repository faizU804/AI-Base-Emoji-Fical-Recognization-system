"""
main.py — Next-Gen Multi-Modal AI Emotion Intelligence Platform.

Modernized 3D interactive interface featuring:
  - 3D perspective particle mesh & floating emotion nodes with mouse parallax
  - Clean, prominent, reliable functional feature cards
  - Direct 1-click access to Live Camera, Image Upload, and Text Emotion AI
  - Dark cosmic obsidian aesthetics with electric indigo accents
  - Reduced Motion toggle and system telemetry status bar
"""

import sys
import tkinter as tk
from tkinter import messagebox

import compat
compat.apply()

from ui_theme import Theme, ParticleCanvas3D, ResponsiveLayout
from camera_emotion import CameraEmotion
from upload_emotion import UploadEmotion
from chat_emotion import ChatEmotion


class EmotionApp:
    """
    Main Desktop Application Window and Navigation Manager.
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("EMOTION MATRIX — Advanced AI Emotion Intelligence")
        self.root.geometry("1100x780")
        self.root.minsize(800, 650)
        self.root.configure(bg=Theme.BG_ROOT)

        # Active background particle canvas
        self.particle_canvas = None

        # Setup protocol & home view
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.show_home()
        self.root.mainloop()

    def clear(self):
        """Cleans all widgets from the root window."""
        if hasattr(self, "active_screen") and self.active_screen is not None:
            if hasattr(self.active_screen, "stop_camera"):
                try:
                    self.active_screen.stop_camera()
                except Exception:
                    pass
            self.active_screen = None

        if self.particle_canvas is not None:
            self.particle_canvas.stop_animation()
            self.particle_canvas = None

        for widget in self.root.winfo_children():
            widget.destroy()

    def show_home(self):
        """Builds the futuristic 3D hero landing interface with prominent functional cards."""
        self.clear()
        self.root.configure(bg=Theme.BG_ROOT)

        # ==========================================
        # 1. TOP BRANDING & TELEMETRY HEADER
        # ==========================================
        header = tk.Frame(self.root, bg=Theme.BG_SURFACE_ALT, height=58, bd=0)
        header.pack(fill="x")
        header.pack_propagate(False)

        header_inner = tk.Frame(header, bg=Theme.BG_SURFACE_ALT)
        header_inner.pack(fill="both", expand=True, padx=24)

        # Left: App Logo & System Telemetry
        brand_box = tk.Frame(header_inner, bg=Theme.BG_SURFACE_ALT)
        brand_box.pack(side="left", pady=12)

        tk.Label(
            brand_box,
            text="⚡ EMOTION MATRIX",
            font=(Theme.FONT_FAMILY, 13, "bold"),
            bg=Theme.BG_SURFACE_ALT,
            fg=Theme.TEXT_HIGHLIGHT
        ).pack(side="left")

        tk.Label(
            brand_box,
            text=" :: AI INTELLIGENCE PLATFORM",
            font=(Theme.FONT_CODE, 10),
            bg=Theme.BG_SURFACE_ALT,
            fg=Theme.TEXT_SECONDARY
        ).pack(side="left")

        # Right: Reduced Motion Toggle & Status Indicator
        controls_box = tk.Frame(header_inner, bg=Theme.BG_SURFACE_ALT)
        controls_box.pack(side="right", pady=12)

        self.motion_btn = tk.Button(
            controls_box,
            text="✨ Motion: ON" if not Theme.REDUCED_MOTION else "🛑 Motion: OFF",
            font=(Theme.FONT_FAMILY, 9),
            bg=Theme.BG_SURFACE,
            fg=Theme.TEXT_SECONDARY,
            activebackground=Theme.BG_CARD_HOVER,
            activeforeground=Theme.TEXT_PRIMARY,
            bd=1,
            relief="solid",
            padx=10,
            pady=3,
            cursor="hand2",
            command=self.toggle_motion
        )
        self.motion_btn.pack(side="left", padx=12)

        tk.Label(
            controls_box,
            text="● ALL CORES ONLINE",
            font=(Theme.FONT_CODE, 9, "bold"),
            bg=Theme.BG_SURFACE_ALT,
            fg=Theme.SUCCESS
        ).pack(side="left")

        # ==========================================
        # 2. 3D HERO BANNER WITH PARTICLES
        # ==========================================
        hero_banner = tk.Frame(self.root, bg=Theme.BG_ROOT, height=145)
        hero_banner.pack(fill="x", padx=20, pady=(10, 5))
        hero_banner.pack_propagate(False)

        self.particle_canvas = ParticleCanvas3D(hero_banner, bg=Theme.BG_ROOT, height=145)
        self.particle_canvas.pack(fill="both", expand=True)
        self.particle_canvas.start_animation()

        # Hero titles overlaid centered on the 3D canvas
        title_box = tk.Frame(self.particle_canvas, bg=Theme.BG_ROOT)
        self.particle_canvas.create_window(540, 72, window=title_box, anchor="center")

        tk.Label(
            title_box,
            text="NEXT-GEN MULTI-MODAL EMOTION AI",
            font=(Theme.FONT_FAMILY, 21, "bold"),
            bg=Theme.BG_ROOT,
            fg=Theme.TEXT_PRIMARY
        ).pack(pady=(0, 4))

        tk.Label(
            title_box,
            text="Real-time Computer Vision facial expression tracking • DistilRoBERTa NLP • Twitter sentiment dataset emoji ranking",
            font=(Theme.FONT_FAMILY, 10),
            bg=Theme.BG_ROOT,
            fg=Theme.TEXT_SECONDARY
        ).pack()

        # ==========================================
        # 3. PRIMARY FUNCTIONAL CARDS (100% Visible & Clickable)
        # ==========================================
        cards_section = tk.Frame(self.root, bg=Theme.BG_ROOT)
        cards_section.pack(fill="both", expand=True, padx=25, pady=(5, 15))

        # Card 1: Live Webcam Vision
        self._build_card(
            parent=cards_section,
            icon="🎥",
            tag="REAL-TIME VISION",
            tag_color=Theme.TEXT_HIGHLIGHT,
            title="LIVE WEBCAM AI",
            desc="Optical face tracking with DeepFace CNN, 5-frame temporal smoothing, confidence thresholding, and Sci-Fi HUD reticles.",
            features=["• 30+ FPS optical stream", "• Facial bounding targets", "• Throttled DeepFace inference"],
            btn_text="🎥  Launch Live Camera",
            btn_bg=Theme.PRIMARY,
            btn_hover=Theme.PRIMARY_HOVER,
            command=self.open_camera
        )

        # Card 2: Image Portrait Analysis
        self._build_card(
            parent=cards_section,
            icon="🖼️",
            tag="STATIC PORTRAIT",
            tag_color="#38BDF8",
            title="PORTRAIT ANALYSIS",
            desc="Upload portrait photos (JPG, PNG, WEBP) to detect facial bounding boxes and dominant emotional expression breakdown.",
            features=["• High-resolution analysis", "• Face target reticles", "• Dominant emotion breakdown"],
            btn_text="📁  Upload & Analyze Image",
            btn_bg=Theme.SECONDARY,
            btn_hover=Theme.SECONDARY_HOVER,
            command=self.open_upload
        )

        # Card 3: NLP Text & Emoji Intelligence
        self._build_card(
            parent=cards_section,
            icon="💬",
            tag="ROBERTA TRANSFORMER",
            tag_color="#A855F7",
            title="TEXT & EMOJI AI",
            desc="Deep semantic NLP emotion classification, negation intelligence ('not happy'), and Twitter sentiment-ranked emojis.",
            features=["• 7 emotion distributions", "• Negation understanding", "• Ranked emoji recommendations"],
            btn_text="⚡  Launch Text Emotion AI",
            btn_bg="#7C3AED",
            btn_hover="#8B5CF6",
            command=self.open_chat
        )

        # ==========================================
        # 4. BOTTOM SYSTEM TELEMETRY BAR
        # ==========================================
        footer = tk.Frame(self.root, bg=Theme.BG_SURFACE, height=44, bd=0)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)

        footer_inner = tk.Frame(footer, bg=Theme.BG_SURFACE)
        footer_inner.pack(fill="both", expand=True, padx=24)

        tk.Label(
            footer_inner,
            text="[ PYTHON 3.14 ]   [ OPENCV 4.14 ]   [ DEEPFACE CNN ]   [ ROBERTA TRANSFORMER ]   [ 969 TWITTER EMOJIS ]",
            font=(Theme.FONT_CODE, 8),
            bg=Theme.BG_SURFACE,
            fg=Theme.TEXT_MUTED
        ).pack(side="left", pady=12)

        tk.Button(
            footer_inner,
            text="✕ Exit",
            font=(Theme.FONT_FAMILY, 9, "bold"),
            bg="#991B1B",
            fg="white",
            activebackground="#B91C1C",
            activeforeground="white",
            relief="flat",
            padx=14,
            pady=3,
            cursor="hand2",
            command=self.close
        ).pack(side="right", pady=8)

    def _build_card(self, parent, icon, tag, tag_color, title, desc, features, btn_text, btn_bg, btn_hover, command):
        """Constructs an elevated functional product card."""
        card = tk.Frame(
            parent,
            bg=Theme.BG_SURFACE,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=Theme.BORDER_SUBTLE,
            padx=22,
            pady=20
        )
        card.pack(side="left", fill="both", expand=True, padx=8)

        # Top tag row
        top_row = tk.Frame(card, bg=Theme.BG_SURFACE)
        top_row.pack(fill="x", pady=(0, 10))

        tk.Label(
            top_row,
            text=icon,
            font=(Theme.FONT_EMOJI, 22),
            bg=Theme.BG_SURFACE,
            fg=tag_color
        ).pack(side="left")

        tk.Label(
            top_row,
            text=tag,
            font=(Theme.FONT_CODE, 8, "bold"),
            bg=Theme.BG_INPUT,
            fg=tag_color,
            padx=8,
            pady=3
        ).pack(side="right")

        # Title
        tk.Label(
            card,
            text=title,
            font=(Theme.FONT_FAMILY, 15, "bold"),
            bg=Theme.BG_SURFACE,
            fg=Theme.TEXT_PRIMARY
        ).pack(anchor="w", pady=(0, 6))

        # Description
        tk.Label(
            card,
            text=desc,
            font=(Theme.FONT_FAMILY, 9),
            bg=Theme.BG_SURFACE,
            fg=Theme.TEXT_SECONDARY,
            wraplength=250,
            justify="left"
        ).pack(anchor="w", pady=(0, 14))

        # Features bullet list
        feat_box = tk.Frame(card, bg=Theme.BG_SURFACE)
        feat_box.pack(anchor="w", fill="x", pady=(0, 20))
        for feat in features:
            tk.Label(
                feat_box,
                text=feat,
                font=(Theme.FONT_FAMILY, 8),
                bg=Theme.BG_SURFACE,
                fg=Theme.TEXT_MUTED,
                anchor="w"
            ).pack(anchor="w", pady=1)

        # Big Action Button
        action_btn = tk.Button(
            card,
            text=btn_text,
            font=(Theme.FONT_FAMILY, 11, "bold"),
            bg=btn_bg,
            fg="white",
            activebackground=btn_hover,
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=16,
            pady=11,
            cursor="hand2",
            command=command
        )
        action_btn.pack(side="bottom", fill="x")

        # Hover micro-interaction for button
        action_btn.bind("<Enter>", lambda e, b=action_btn, c=btn_hover: b.config(bg=c))
        action_btn.bind("<Leave>", lambda e, b=action_btn, c=btn_bg: b.config(bg=c))

    def toggle_motion(self):
        """Toggles reduced motion on/off."""
        Theme.REDUCED_MOTION = not Theme.REDUCED_MOTION
        if hasattr(self, "motion_btn"):
            self.motion_btn.config(
                text="🛑 Motion: OFF" if Theme.REDUCED_MOTION else "✨ Motion: ON"
            )

    def open_camera(self):
        try:
            self.clear()
            self.active_screen = CameraEmotion(self.root, self.show_home)
        except Exception as error:
            messagebox.showerror("Camera Error", str(error))
            self.show_home()

    def open_upload(self):
        try:
            self.clear()
            self.active_screen = UploadEmotion(self.root, self.show_home)
        except Exception as error:
            messagebox.showerror("Upload Error", str(error))
            self.show_home()

    def open_chat(self):
        try:
            self.clear()
            self.active_screen = ChatEmotion(self.root, self.show_home)
        except Exception as error:
            messagebox.showerror("Text AI Error", str(error))
            self.show_home()

    def close(self):
        if self.particle_canvas is not None:
            self.particle_canvas.stop_animation()
        self.root.destroy()


if __name__ == "__main__":
    EmotionApp()