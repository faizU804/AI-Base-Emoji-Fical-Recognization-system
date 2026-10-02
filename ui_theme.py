"""
ui_theme.py — Unified Design System & 3D Interactive Component Library.

Provides:
  - Futuristic Dark AI Palette (Obsidian, Electric Indigo, Emotion Accent Tokens)
  - Responsive Breakpoint Utilities (Desktop, Tablet, Mobile)
  - 3D Interactive Particle Mesh Canvas with perspective projection & mouse parallax
  - Sci-Fi HUD Camera Overlay (Corner brackets, scanning lines, telemetry)
  - Animated State Badges & Emotion-Aware Theme Mapping
  - Reduced Motion support
"""

import math
import random
import time
import tkinter as tk
from typing import Dict, Tuple, Optional, Callable


# ==============================================================================
# DESIGN SYSTEM COLOR PALETTE (Futuristic Dark AI Aesthetic)
# ==============================================================================
class Theme:
    # Deep backgrounds
    BG_ROOT = "#090D16"         # Deepest cosmic obsidian
    BG_SURFACE = "#0F172A"      # Slate 900 (primary cards)
    BG_SURFACE_ALT = "#1E293B"  # Slate 800 (elevated headers, toolbars)
    BG_CARD_HOVER = "#243247"   # Subtle hover brightness
    BG_INPUT = "#0D1322"        # Inset text inputs
    
    # Borders & Dividers
    BORDER_SUBTLE = "#1E293B"
    BORDER_LIGHT = "#334155"
    BORDER_GLOW = "#6366F1"     # Electric indigo highlight
    
    # Typography
    TEXT_PRIMARY = "#F8FAFC"    # Pure crisp white
    TEXT_SECONDARY = "#94A3B8"  # Cool slate muted
    TEXT_MUTED = "#64748B"      # Darker hint
    TEXT_HIGHLIGHT = "#38BDF8"  # Cyan accent
    
    # Core Accent
    PRIMARY = "#4F46E5"         # Electric Indigo
    PRIMARY_HOVER = "#6366F1"
    PRIMARY_ACTIVE = "#4338CA"
    PRIMARY_GLOW = "#818CF8"

    SECONDARY = "#0EA5E9"       # Cyber Sky Blue
    SECONDARY_HOVER = "#38BDF8"

    DANGER = "#EF4444"          # Crimson Red
    DANGER_HOVER = "#F87171"

    SUCCESS = "#10B981"         # Emerald Green
    SUCCESS_HOVER = "#34D399"

    # Emotion-Reactive Dynamic Theme Accents
    EMOTION_THEMES = {
        "happy": {
            "name": "JOY / HAPPY",
            "emoji": "😊",
            "color": "#10B981",       # Emerald
            "glow": "#34D399",
            "bg_tint": "#064E3B",
            "badge_bg": "#065F46",
            "sentiment": "Positive"
        },
        "sad": {
            "name": "SAD / GRIEF",
            "emoji": "😢",
            "color": "#3B82F6",       # Cobalt Blue
            "glow": "#60A5FA",
            "bg_tint": "#1E3A8A",
            "badge_bg": "#1E40AF",
            "sentiment": "Negative"
        },
        "angry": {
            "name": "ANGRY / RAGE",
            "emoji": "😡",
            "color": "#EF4444",       # Crimson
            "glow": "#F87171",
            "bg_tint": "#7F1D1D",
            "badge_bg": "#991B1B",
            "sentiment": "Negative"
        },
        "surprise": {
            "name": "SURPRISE / AWE",
            "emoji": "😲",
            "color": "#F59E0B",       # Amber Gold
            "glow": "#FBBF24",
            "bg_tint": "#78350F",
            "badge_bg": "#92400E",
            "sentiment": "Neutral/Pos"
        },
        "fear": {
            "name": "FEAR / ANXIOUS",
            "emoji": "😨",
            "color": "#8B5CF6",       # Purple
            "glow": "#A78BFA",
            "bg_tint": "#4C1D95",
            "badge_bg": "#5B21B6",
            "sentiment": "Negative"
        },
        "disgust": {
            "name": "DISGUST",
            "emoji": "🤢",
            "color": "#84CC16",       # Toxic Lime
            "glow": "#A3E635",
            "bg_tint": "#365314",
            "badge_bg": "#3F6212",
            "sentiment": "Negative"
        },
        "neutral": {
            "name": "NEUTRAL",
            "emoji": "😐",
            "color": "#64748B",       # Slate
            "glow": "#94A3B8",
            "bg_tint": "#1E293B",
            "badge_bg": "#334155",
            "sentiment": "Neutral"
        }
    }

    # Typography Font Families
    FONT_FAMILY = "Segoe UI"
    FONT_CODE = "Consolas"
    FONT_EMOJI = "Segoe UI Emoji"

    # Global preference for reduced motion (can be toggled by user)
    REDUCED_MOTION = False


# ==============================================================================
# RESPONSIVE BREAKPOINT MANAGER
# ==============================================================================
class ResponsiveLayout:
    """Classifies window width into responsive layouts."""
    DESKTOP = "desktop"  # width >= 950
    TABLET = "tablet"    # 650 <= width < 950
    MOBILE = "mobile"    # width < 650

    @staticmethod
    def get_tier(width: int) -> str:
        if width >= 950:
            return ResponsiveLayout.DESKTOP
        elif width >= 650:
            return ResponsiveLayout.TABLET
        return ResponsiveLayout.MOBILE


# ==============================================================================
# INTERACTIVE 3D PARTICLE CONSTELLATION CANVAS
# ==============================================================================
class ParticleCanvas3D(tk.Canvas):
    """
    A 3D perspective particle mesh with depth layers, floating emotion nodes,
    and responsive mouse parallax interaction.
    """

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("bg", Theme.BG_ROOT)
        kwargs.setdefault("highlightthickness", 0)
        kwargs.setdefault("bd", 0)
        super().__init__(parent, **kwargs)

        self.parent = parent
        self.particles = []
        self.num_particles = 32
        self.running = False
        self.after_id = None

        # 3D Camera / Perspective parameters
        self.cam_fov = 280.0
        self.cam_dist = 400.0

        # Mouse parallax coordinates
        self.mouse_target_x = 0.0
        self.mouse_target_y = 0.0
        self.parallax_x = 0.0
        self.parallax_y = 0.0

        # Floating emoji symbols in 3D space
        self.emojis_3d = ["✨", "🧠", "💡", "⚡", "🔮", "💎"]
        self.nodes_3d = []

        self._init_particles()
        self._init_floating_nodes()

        self.bind("<Configure>", self._on_resize)
        self.bind("<Motion>", self._on_mouse_move)

    def _init_particles(self):
        self.particles = []
        for _ in range(self.num_particles):
            # Coordinates in 3D cube: x in [-300, 300], y in [-200, 200], z in [50, 450]
            self.particles.append({
                "x": random.uniform(-350, 350),
                "y": random.uniform(-250, 250),
                "z": random.uniform(50, 450),
                "vx": random.uniform(-0.4, 0.4),
                "vy": random.uniform(-0.3, 0.3),
                "vz": random.uniform(-0.3, 0.3),
                "radius": random.uniform(2.0, 4.0),
                "color": random.choice(["#6366F1", "#818CF8", "#38BDF8", "#A855F7", "#34D399"])
            })

    def _init_floating_nodes(self):
        self.nodes_3d = []
        for i, symbol in enumerate(self.emojis_3d):
            self.nodes_3d.append({
                "symbol": symbol,
                "x": random.uniform(-300, 300),
                "y": random.uniform(-200, 200),
                "z": random.uniform(100, 400),
                "vx": random.uniform(-0.25, 0.25),
                "vy": random.uniform(-0.2, 0.2),
                "vz": random.uniform(-0.2, 0.2),
                "base_size": random.randint(14, 20)
            })

    def _on_resize(self, event):
        pass

    def _on_mouse_move(self, event):
        w = max(1, self.winfo_width())
        h = max(1, self.winfo_height())
        # Normalized mouse offset [-1, 1]
        self.mouse_target_x = (event.x - (w / 2.0)) / (w / 2.0)
        self.mouse_target_y = (event.y - (h / 2.0)) / (h / 2.0)

    def start_animation(self):
        if not self.running:
            self.running = True
            self._render_frame()

    def stop_animation(self):
        self.running = False
        if self.after_id is not None:
            try:
                self.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

    def _render_frame(self):
        if not self.running:
            return

        w = self.winfo_width()
        h = self.winfo_height()
        if w < 10 or h < 10:
            self.after_id = self.after(50, self._render_frame)
            return

        cx = w / 2.0
        cy = h / 2.0

        # Smooth mouse parallax damping
        if not Theme.REDUCED_MOTION:
            self.parallax_x += (self.mouse_target_x * 40.0 - self.parallax_x) * 0.08
            self.parallax_y += (self.mouse_target_y * 30.0 - self.parallax_y) * 0.08

        self.delete("particle")

        projected_pts = []

        # 1. Update and Project 3D Particles
        for p in self.particles:
            if not Theme.REDUCED_MOTION:
                p["x"] += p["vx"]
                p["y"] += p["vy"]
                p["z"] += p["vz"]

                # Boundary bounce
                if p["x"] < -400 or p["x"] > 400:
                    p["vx"] *= -1
                if p["y"] < -300 or p["y"] > 300:
                    p["vy"] *= -1
                if p["z"] < 50 or p["z"] > 450:
                    p["vz"] *= -1

            # Apply perspective projection
            eff_x = p["x"] + self.parallax_x
            eff_y = p["y"] + self.parallax_y
            eff_z = max(20.0, p["z"])

            scale = self.cam_fov / (self.cam_fov + eff_z)
            sx = cx + eff_x * scale
            sy = cy + eff_y * scale
            r = max(1.2, p["radius"] * scale)

            projected_pts.append((sx, sy, eff_z, r, p["color"]))

        # 2. Draw 3D Connection Lines between nearby particles
        num_pts = len(projected_pts)
        for i in range(num_pts):
            x1, y1, z1, _, _ = projected_pts[i]
            for j in range(i + 1, num_pts):
                x2, y2, z2, _, _ = projected_pts[j]
                dist_sq = (x1 - x2) ** 2 + (y1 - y2) ** 2
                if dist_sq < 9000:  # Distance < ~95px
                    dist = math.sqrt(dist_sq)
                    alpha_factor = 1.0 - (dist / 95.0)
                    depth_factor = max(0.2, 1.0 - (z1 / 500.0))
                    # Pick line color based on depth
                    line_col = "#24324D" if depth_factor < 0.6 else "#374A72"
                    self.create_line(x1, y1, x2, y2, fill=line_col, width=1, tags="particle")

        # 3. Draw Projected Particles
        for sx, sy, z, r, color in projected_pts:
            self.create_oval(
                sx - r, sy - r, sx + r, sy + r,
                fill=color, outline="", tags="particle"
            )

        # 4. Project and Draw 3D Floating Emoji Nodes
        for node in self.nodes_3d:
            if not Theme.REDUCED_MOTION:
                node["x"] += node["vx"]
                node["y"] += node["vy"]
                node["z"] += node["vz"]

                if node["x"] < -350 or node["x"] > 350:
                    node["vx"] *= -1
                if node["y"] < -250 or node["y"] > 250:
                    node["vy"] *= -1
                if node["z"] < 100 or node["z"] > 400:
                    node["vz"] *= -1

            eff_x = node["x"] + self.parallax_x * 1.5
            eff_y = node["y"] + self.parallax_y * 1.5
            eff_z = max(50.0, node["z"])

            scale = self.cam_fov / (self.cam_fov + eff_z)
            sx = cx + eff_x * scale
            sy = cy + eff_y * scale
            font_size = max(10, int(node["base_size"] * scale))

            self.create_text(
                sx, sy,
                text=node["symbol"],
                font=(Theme.FONT_EMOJI, font_size),
                tags="particle"
            )

        # Schedule next animation frame (~33ms = 30 FPS for minimal CPU load)
        delay = 100 if Theme.REDUCED_MOTION else 33
        self.after_id = self.after(delay, self._render_frame)


# ==============================================================================
# SCI-FI HUD CAMERA SCANNER OVERLAY
# ==============================================================================
class CameraHUDOverlay:
    """
    Renders cybernetic corner brackets, animated scanning beam, and telemetry
    directly onto preview frames or canvas.
    """

    def __init__(self):
        self.scan_y = 0.0
        self.scan_speed = 4.0
        self.scan_dir = 1

    def draw_hud(self, canvas: tk.Canvas, width: int, height: int, face_count: int, emotion_meta: dict):
        canvas.delete("hud")

        accent_color = emotion_meta.get("color", Theme.PRIMARY_GLOW)
        bracket_len = 24
        bracket_pad = 12

        # 1. Four Cyber Corner Brackets
        # Top-Left
        canvas.create_line(bracket_pad, bracket_pad, bracket_pad + bracket_len, bracket_pad, fill=accent_color, width=3, tags="hud")
        canvas.create_line(bracket_pad, bracket_pad, bracket_pad, bracket_pad + bracket_len, fill=accent_color, width=3, tags="hud")
        # Top-Right
        canvas.create_line(width - bracket_pad, bracket_pad, width - bracket_pad - bracket_len, bracket_pad, fill=accent_color, width=3, tags="hud")
        canvas.create_line(width - bracket_pad, bracket_pad, width - bracket_pad, bracket_pad + bracket_len, fill=accent_color, width=3, tags="hud")
        # Bottom-Left
        canvas.create_line(bracket_pad, height - bracket_pad, bracket_pad + bracket_len, height - bracket_pad, fill=accent_color, width=3, tags="hud")
        canvas.create_line(bracket_pad, height - bracket_pad, bracket_pad, height - bracket_pad - bracket_len, fill=accent_color, width=3, tags="hud")
        # Bottom-Right
        canvas.create_line(width - bracket_pad, height - bracket_pad, width - bracket_pad - bracket_len, height - bracket_pad, fill=accent_color, width=3, tags="hud")
        canvas.create_line(width - bracket_pad, height - bracket_pad, width - bracket_pad, height - bracket_pad - bracket_len, fill=accent_color, width=3, tags="hud")

        # 2. Animated Holographic Scanning Beam
        if not Theme.REDUCED_MOTION:
            self.scan_y += self.scan_speed * self.scan_dir
            if self.scan_y > (height - 30):
                self.scan_dir = -1
            elif self.scan_y < 30:
                self.scan_dir = 1

            # Main laser beam
            canvas.create_line(
                bracket_pad + 5, self.scan_y,
                width - bracket_pad - 5, self.scan_y,
                fill=accent_color, width=2, tags="hud"
            )
            # Subtle glow trail
            canvas.create_line(
                bracket_pad + 15, self.scan_y - (2 * self.scan_dir),
                width - bracket_pad - 15, self.scan_y - (2 * self.scan_dir),
                fill=Theme.SECONDARY, width=1, tags="hud"
            )

        # 3. Telemetry Overlay Badges
        # Top Header Telemetry
        canvas.create_text(
            bracket_pad + 10, bracket_pad + 12,
            text="[ VISION_AI :: STREAM ACTIVE ]",
            font=(Theme.FONT_CODE, 8, "bold"),
            fill=Theme.TEXT_SECONDARY, anchor="w", tags="hud"
        )
        canvas.create_text(
            width - bracket_pad - 10, bracket_pad + 12,
            text=f"FACES: {face_count} | 30 FPS",
            font=(Theme.FONT_CODE, 8, "bold"),
            fill=accent_color, anchor="e", tags="hud"
        )


# ==============================================================================
# MODERN ELEVATED INTERACTIVE BUTTON
# ==============================================================================
class CyberButton(tk.Frame):
    """
    A sleek, styled interactive button with hover elevation, active compression,
    and rounded neon aesthetics.
    """

    def __init__(
        self,
        parent,
        text: str,
        command: Optional[Callable] = None,
        bg_color: str = Theme.PRIMARY,
        hover_color: str = Theme.PRIMARY_HOVER,
        fg_color: str = Theme.TEXT_PRIMARY,
        font: tuple = (Theme.FONT_FAMILY, 10, "bold"),
        padx: int = 16,
        pady: int = 8,
        icon: str = "",
        cursor: str = "hand2",
        **kwargs
    ):
        super().__init__(parent, bg=bg_color, bd=0, highlightthickness=1, highlightbackground=Theme.BORDER_LIGHT, **kwargs)
        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.fg_color = fg_color

        full_text = f"{icon}  {text}" if icon else text
        self.lbl = tk.Label(
            self,
            text=full_text,
            font=font,
            bg=bg_color,
            fg=fg_color,
            padx=padx,
            pady=pady,
            cursor=cursor
        )
        self.lbl.pack(fill="both", expand=True)

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self.bind("<ButtonRelease-1>", self._on_release)

        self.lbl.bind("<Enter>", self._on_enter)
        self.lbl.bind("<Leave>", self._on_leave)
        self.lbl.bind("<Button-1>", self._on_click)
        self.lbl.bind("<ButtonRelease-1>", self._on_release)

    def _on_enter(self, event):
        self.config(bg=self.hover_color, highlightbackground=self.hover_color)
        self.lbl.config(bg=self.hover_color)

    def _on_leave(self, event):
        self.config(bg=self.bg_color, highlightbackground=Theme.BORDER_LIGHT)
        self.lbl.config(bg=self.bg_color)

    def _on_click(self, event):
        # Visual compression
        self.config(highlightbackground=Theme.TEXT_HIGHLIGHT)
        if self.command:
            self.command()

    def _on_release(self, event):
        self.config(highlightbackground=self.hover_color)

    def set_text(self, text: str):
        self.lbl.config(text=text)

    def set_enabled(self, enabled: bool):
        if enabled:
            self.config(bg=self.bg_color)
            self.lbl.config(bg=self.bg_color, fg=self.fg_color, cursor="hand2")
        else:
            self.config(bg=Theme.BG_SURFACE_ALT)
            self.lbl.config(bg=Theme.BG_SURFACE_ALT, fg=Theme.TEXT_MUTED, cursor="arrow")
