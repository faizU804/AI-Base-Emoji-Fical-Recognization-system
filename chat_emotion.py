"""
chat_emotion.py — AI-Powered Text Emotion Recognition & Emoji Intelligence.

Features:
  - DistilRoBERTa Transformer NLP for deep emotion classification (7 classes)
  - Dataset-driven emoji ranking using Twitter sentiment & frequency CSVs
  - Emotion-reactive visual theming (card border & glows change with detected emotion)
  - Responsive 2-column layout with dark cosmic AI aesthetics
  - Micro-interactions (hover elevation, copy toast, animated probability bars)
  - 1-click test presets (including negation tests) and clipboard copy
  - LIVE auto-detection while typing with debounced inference (750ms idle trigger)
"""

import threading
import tkinter as tk
from tkinter import messagebox
from typing import Dict, Any, Optional

import compat
compat.apply()

from ui_theme import Theme, CyberButton, ResponsiveLayout
from text_emotion_ai import get_text_ai, EMOTION_DISPLAY


class ChatEmotion:
    """
    GUI Screen for Text Emotion & Emoji Recommendation.
    """

    def __init__(self, parent, go_back):
        self.parent = parent
        self.go_back = go_back

        # Access singleton text emotion AI engine
        self.ai = get_text_ai()
        self.is_analyzing = False

        # Live auto-detection debounce state
        self._debounce_id = None           # Tkinter after() timer ID
        self._last_analyzed_text = ""      # Prevents re-analyzing identical text
        self._live_enabled = True          # Master toggle for live mode
        self.DEBOUNCE_MS = 750             # Idle time before auto-triggering (ms)
        self.MIN_CHARS = 3                 # Minimum chars to trigger live analysis

        self.build_ui()

    def build_ui(self):
        # Clear existing widgets from parent
        for widget in self.parent.winfo_children():
            widget.destroy()

        self.parent.configure(bg=Theme.BG_ROOT)

        # ==========================================
        # TOP HEADER BAR
        # ==========================================
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
            command=self.handle_back,
            padx=12,
            pady=5
        )
        back_btn.pack(side="left", pady=14)

        # Title & Subtitle
        title_box = tk.Frame(header_content, bg=Theme.BG_SURFACE_ALT)
        title_box.pack(side="left", padx=20, pady=10)

        tk.Label(
            title_box,
            text="💬 NLP TEXT EMOTION & EMOJI INTELLIGENCE",
            font=(Theme.FONT_FAMILY, 13, "bold"),
            bg=Theme.BG_SURFACE_ALT,
            fg=Theme.TEXT_PRIMARY
        ).pack(anchor="w")

        tk.Label(
            title_box,
            text="Transformer Language Model • Twitter Dataset Ranking • Sentiment Distribution",
            font=(Theme.FONT_FAMILY, 8),
            bg=Theme.BG_SURFACE_ALT,
            fg=Theme.TEXT_SECONDARY
        ).pack(anchor="w")

        # Status Indicator Badge
        self.status_badge = tk.Label(
            header_content,
            text="● RoBERTa AI Active" if self.ai.model_loaded else "◌ Initializing Core...",
            font=(Theme.FONT_CODE, 9, "bold"),
            bg=Theme.BG_SURFACE_ALT,
            fg=Theme.SUCCESS if self.ai.model_loaded else Theme.EMOTION_THEMES["surprise"]["color"]
        )
        self.status_badge.pack(side="right", pady=18)

        # ==========================================
        # MAIN CONTENT CONTAINER (2-column layout)
        # ==========================================
        self.container = tk.Frame(self.parent, bg=Theme.BG_ROOT)
        self.container.pack(fill="both", expand=True, padx=22, pady=16)
        self.container.grid_columnconfigure(0, weight=4)  # Left: Input & Presets
        self.container.grid_columnconfigure(1, weight=5)  # Right: Results & Analytics
        self.container.grid_rowconfigure(0, weight=1)

        # ==========================================
        # LEFT PANEL: INPUT & PROMPT PRESETS
        # ==========================================
        left_card = tk.Frame(
            self.container,
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
            text="Enter Natural Language Text",
            font=(Theme.FONT_FAMILY, 12, "bold"),
            bg=Theme.BG_SURFACE,
            fg=Theme.TEXT_PRIMARY
        ).pack(anchor="w")

        tk.Label(
            left_inner,
            text="The AI evaluates deep contextual semantics, negation nuance,\nand maps outputs to Twitter sentiment dataset rankings.",
            font=(Theme.FONT_FAMILY, 8),
            bg=Theme.BG_SURFACE,
            fg=Theme.TEXT_MUTED,
            justify="left"
        ).pack(anchor="w", pady=(2, 10))

        # Text input box with dark inset aesthetic
        self.text_input = tk.Text(
            left_inner,
            font=(Theme.FONT_FAMILY, 11),
            height=4,
            bg=Theme.BG_INPUT,
            fg=Theme.TEXT_PRIMARY,
            insertbackground=Theme.TEXT_HIGHLIGHT,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=Theme.BORDER_LIGHT,
            padx=12,
            pady=10,
            wrap="word"
        )
        self.text_input.pack(fill="x", pady=(0, 10))
        self.text_input.bind("<Return>", self._on_enter_press)
        # Live auto-detection: trigger debounced analysis on every keystroke
        self.text_input.bind("<KeyRelease>", self._on_text_modified)

        # Action Buttons Row
        action_row = tk.Frame(left_inner, bg=Theme.BG_SURFACE)
        action_row.pack(fill="x", pady=(0, 10))

        self.analyze_btn = CyberButton(
            action_row,
            text="Analyze Emotion",
            icon="⚡",
            bg_color=Theme.PRIMARY,
            hover_color=Theme.PRIMARY_HOVER,
            command=self.start_analysis,
            padx=16,
            pady=6
        )
        self.analyze_btn.pack(side="left")

        clear_btn = CyberButton(
            action_row,
            text="Clear",
            bg_color=Theme.BG_SURFACE_ALT,
            hover_color=Theme.BG_CARD_HOVER,
            fg_color=Theme.TEXT_SECONDARY,
            command=self.clear_input,
            padx=12,
            pady=6
        )
        clear_btn.pack(side="left", padx=8)

        # Live Mode Indicator (subtle pulsing dot)
        self.live_indicator = tk.Label(
            action_row,
            text="● LIVE",
            font=(Theme.FONT_CODE, 8, "bold"),
            bg=Theme.BG_SURFACE,
            fg=Theme.SUCCESS,
            padx=8
        )
        self.live_indicator.pack(side="left", padx=(8, 0))

        # Progress / Loading Indicator
        self.loading_label = tk.Label(
            left_inner,
            text="",
            font=(Theme.FONT_FAMILY, 9, "italic"),
            bg=Theme.BG_SURFACE,
            fg=Theme.TEXT_HIGHLIGHT
        )
        self.loading_label.pack(anchor="w", pady=(0, 8))

        # Quick Test Presets Section
        tk.Label(
            left_inner,
            text="Curated Evaluation Presets (Click to run):",
            font=(Theme.FONT_FAMILY, 10, "bold"),
            bg=Theme.BG_SURFACE,
            fg=Theme.TEXT_SECONDARY
        ).pack(anchor="w", pady=(6, 6))

        presets_frame = tk.Frame(left_inner, bg=Theme.BG_SURFACE)
        presets_frame.pack(fill="both", expand=True)

        presets = [
            ("🎉 Joy / Promotion", "I just got promoted at work! Best day of my life!"),
            ("💔 Sad / Heartbroken", "I failed the exam and feel completely heartbroken."),
            ("🚫 Negation Test", "I am definitely not happy with this poor service."),
            ("😡 Angry / Furious", "Stop lying to me, I am absolutely furious!"),
            ("😲 Surprise / Shock", "What?! That is completely unexpected and unbelievable!"),
            ("😨 Fear / Terrified", "I'm terrified of walking alone in this dark alley."),
            ("🤢 Disgust / Rotten", "This meat smells rotten and makes me sick."),
            ("📅 Neutral Statement", "The weekly team meeting is scheduled for 3 PM tomorrow.")
        ]

        for label, text in presets:
            btn = tk.Button(
                presets_frame,
                text=label,
                font=(Theme.FONT_FAMILY, 8),
                bg=Theme.BG_SURFACE_ALT,
                fg=Theme.TEXT_PRIMARY,
                activebackground=Theme.BG_CARD_HOVER,
                activeforeground=Theme.TEXT_HIGHLIGHT,
                bd=1,
                relief="solid",
                anchor="w",
                padx=10,
                pady=3,
                cursor="hand2",
                command=lambda t=text: self.load_preset(t)
            )
            btn.pack(fill="x", pady=2)

        # ==========================================
        # RIGHT PANEL: EMOTION ANALYTICS & EMOJIS
        # ==========================================
        right_card = tk.Frame(
            self.container,
            bg=Theme.BG_SURFACE,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=Theme.BORDER_SUBTLE
        )
        right_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        right_inner = tk.Frame(right_card, bg=Theme.BG_SURFACE)
        right_inner.pack(fill="both", expand=True, padx=20, pady=18)

        # Top Section: Detected Emotion Hero Card
        self.emotion_card = tk.Frame(
            right_inner,
            bg=Theme.BG_INPUT,
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=Theme.BORDER_LIGHT,
            padx=16,
            pady=14
        )
        self.emotion_card.pack(fill="x", pady=(0, 12))

        top_info_row = tk.Frame(self.emotion_card, bg=Theme.BG_INPUT)
        top_info_row.pack(fill="x")

        # Large Hero Emoji
        self.hero_emoji = tk.Label(
            top_info_row,
            text="💬",
            font=(Theme.FONT_EMOJI, 40),
            bg=Theme.BG_INPUT,
            fg=Theme.TEXT_HIGHLIGHT
        )
        self.hero_emoji.pack(side="left", padx=(0, 14))

        # Emotion Name & Badges
        name_col = tk.Frame(top_info_row, bg=Theme.BG_INPUT)
        name_col.pack(side="left", fill="both", expand=True)

        self.emotion_title = tk.Label(
            name_col,
            text="READY FOR ANALYSIS",
            font=(Theme.FONT_FAMILY, 15, "bold"),
            bg=Theme.BG_INPUT,
            fg=Theme.TEXT_PRIMARY
        )
        self.emotion_title.pack(anchor="w")

        badge_row = tk.Frame(name_col, bg=Theme.BG_INPUT)
        badge_row.pack(anchor="w", pady=(3, 0))

        self.sentiment_badge = tk.Label(
            badge_row,
            text="Sentiment: Pending",
            font=(Theme.FONT_CODE, 8, "bold"),
            bg=Theme.BG_SURFACE_ALT,
            fg=Theme.TEXT_SECONDARY,
            padx=8,
            pady=2
        )
        self.sentiment_badge.pack(side="left", padx=(0, 8))

        self.confidence_badge = tk.Label(
            badge_row,
            text="Confidence: --%",
            font=(Theme.FONT_CODE, 8, "bold"),
            bg="#1E3A8A",
            fg="#60A5FA",
            padx=8,
            pady=2
        )
        self.confidence_badge.pack(side="left")

        # Nuance Explanation Label
        self.explanation_label = tk.Label(
            self.emotion_card,
            text="Type any thought above or select a preset to analyze.",
            font=(Theme.FONT_FAMILY, 9, "italic"),
            bg=Theme.BG_INPUT,
            fg=Theme.TEXT_MUTED,
            anchor="w",
            justify="left",
            wraplength=400
        )
        self.explanation_label.pack(fill="x", pady=(8, 0))

        # Recommendations Section
        recs_header = tk.Frame(right_inner, bg=Theme.BG_SURFACE)
        recs_header.pack(fill="x", pady=(2, 6))

        tk.Label(
            recs_header,
            text="AI-Recommended Emojis (Dataset-Ranked):",
            font=(Theme.FONT_FAMILY, 10, "bold"),
            bg=Theme.BG_SURFACE,
            fg=Theme.TEXT_PRIMARY
        ).pack(side="left")

        self.copy_feedback = tk.Label(
            recs_header,
            text="",
            font=(Theme.FONT_FAMILY, 8, "bold"),
            bg=Theme.BG_SURFACE,
            fg=Theme.SUCCESS
        )
        self.copy_feedback.pack(side="right")

        self.recs_frame = tk.Frame(right_inner, bg=Theme.BG_SURFACE)
        self.recs_frame.pack(fill="x", pady=(0, 12))

        # Default placeholder emojis
        self._display_recommended_emojis(["😊", "👍", "💬", "✨", "🎉"])

        # Emotion Probability Distribution Bars
        tk.Label(
            right_inner,
            text="Emotion Probability Distribution:",
            font=(Theme.FONT_FAMILY, 10, "bold"),
            bg=Theme.BG_SURFACE,
            fg=Theme.TEXT_PRIMARY
        ).pack(anchor="w", pady=(0, 6))

        self.bars_container = tk.Frame(right_inner, bg=Theme.BG_SURFACE)
        self.bars_container.pack(fill="both", expand=True)

        self.bar_rows = {}
        for emotion_key in ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]:
            meta = Theme.EMOTION_THEMES[emotion_key]
            row = tk.Frame(self.bars_container, bg=Theme.BG_SURFACE)
            row.pack(fill="x", pady=2)

            name_lbl = tk.Label(
                row,
                text=f"{meta['emoji']} {emotion_key.capitalize()}",
                font=(Theme.FONT_FAMILY, 8, "bold"),
                width=11,
                anchor="w",
                bg=Theme.BG_SURFACE,
                fg=Theme.TEXT_SECONDARY
            )
            name_lbl.pack(side="left")

            # Progress bar canvas
            canvas = tk.Canvas(row, height=12, bg=Theme.BG_INPUT, bd=0, highlightthickness=0)
            canvas.pack(side="left", fill="x", expand=True, padx=8)

            pct_lbl = tk.Label(
                row,
                text="0.0%",
                font=(Theme.FONT_CODE, 8),
                width=6,
                anchor="e",
                bg=Theme.BG_SURFACE,
                fg=Theme.TEXT_MUTED
            )
            pct_lbl.pack(side="right")

            self.bar_rows[emotion_key] = {
                "canvas": canvas,
                "pct_lbl": pct_lbl,
                "color": meta["color"]
            }

        # Check periodic model status
        self._check_model_status_loop()

    def _check_model_status_loop(self):
        if self.ai.model_loaded:
            self.status_badge.config(text="● RoBERTa AI Active", fg=Theme.SUCCESS)
        else:
            self.parent.after(500, self._check_model_status_loop)

    def load_preset(self, text: str):
        self.text_input.delete("1.0", "end")
        self.text_input.insert("1.0", text)
        self.start_analysis()

    def clear_input(self):
        self.text_input.delete("1.0", "end")
        self.copy_feedback.config(text="")

    def _on_enter_press(self, event):
        if not event.state & 0x0001:  # Not Shift
            self._cancel_debounce()  # Cancel any pending live timer
            self.start_analysis()
            return "break"
        return None

    # ==================================================
    # LIVE AUTO-DETECTION (Debounced Inference)
    # ==================================================

    def _cancel_debounce(self):
        """Cancels any pending debounce timer."""
        if self._debounce_id is not None:
            try:
                self.parent.after_cancel(self._debounce_id)
            except Exception:
                pass
            self._debounce_id = None

    def _on_text_modified(self, event=None):
        """
        Called on every <KeyRelease> in the text widget.
        Resets the debounce timer so inference only fires
        after the user stops typing for DEBOUNCE_MS.
        """
        if not self._live_enabled:
            return

        # Ignore modifier-only keys that don't change content
        if event and event.keysym in (
            "Shift_L", "Shift_R", "Control_L", "Control_R",
            "Alt_L", "Alt_R", "Caps_Lock", "Tab", "Escape",
            "Up", "Down", "Left", "Right", "Home", "End",
            "Prior", "Next",  # Page Up / Page Down
        ):
            return

        # Cancel previous pending timer
        self._cancel_debounce()

        text = self.text_input.get("1.0", "end").strip()

        if len(text) < self.MIN_CHARS:
            # Text too short — update indicator but don't analyze
            self.live_indicator.config(text="● LIVE", fg=Theme.SUCCESS)
            return

        # Skip if identical to last analyzed text (no change)
        if text == self._last_analyzed_text:
            return

        # Show "waiting" state in live indicator
        self.live_indicator.config(text="◌ TYPING...", fg=Theme.EMOTION_THEMES["surprise"]["color"])

        # Schedule analysis after debounce delay
        self._debounce_id = self.parent.after(self.DEBOUNCE_MS, self._run_live_analysis)

    def _run_live_analysis(self):
        """
        Fires after the debounce period elapses. Runs inference
        in a background thread to keep the UI completely responsive.
        """
        self._debounce_id = None

        text = self.text_input.get("1.0", "end").strip()

        # Guard: text disappeared, too short, or already analyzed
        if len(text) < self.MIN_CHARS or text == self._last_analyzed_text:
            self.live_indicator.config(text="● LIVE", fg=Theme.SUCCESS)
            return

        # Guard: manual analysis already in flight
        if self.is_analyzing:
            return

        self.is_analyzing = True
        self.live_indicator.config(text="⚡ ANALYZING...", fg=Theme.TEXT_HIGHLIGHT)
        self.loading_label.config(text="")

        captured_text = text  # Snapshot to compare after inference

        def _worker():
            result = self.ai.analyze_text(captured_text)
            self.parent.after(0, self._on_live_result, captured_text, result)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def _on_live_result(self, analyzed_text: str, result: Dict[str, Any]):
        """
        Callback on the main thread after live inference completes.
        Only renders if the text hasn't changed while inference was running.
        """
        self.is_analyzing = False
        self._last_analyzed_text = analyzed_text
        self.live_indicator.config(text="● LIVE", fg=Theme.SUCCESS)

        # If user typed more while we were analyzing, don't render stale results
        current_text = self.text_input.get("1.0", "end").strip()
        if current_text != analyzed_text:
            # Text changed during inference — re-trigger debounce for new text
            self._on_text_modified()
            return

        # Render the fresh results
        self._render_results(result)

    def start_analysis(self):
        if self.is_analyzing:
            return

        text = self.text_input.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Input Required", "Please enter a sentence or choose a preset to analyze.")
            return

        self._cancel_debounce()  # Cancel any pending live timer
        self.is_analyzing = True
        self.analyze_btn.set_enabled(False)
        self.loading_label.config(text="⚡ Running DistilRoBERTa emotion inference...")
        self.live_indicator.config(text="⚡ ANALYZING...", fg=Theme.TEXT_HIGHLIGHT)

        def _worker():
            result = self.ai.analyze_text(text)
            self._last_analyzed_text = text  # Track to avoid re-analyzing same text
            self.parent.after(0, self._render_results, result)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def _render_results(self, result: Dict[str, Any]):
        self.is_analyzing = False
        self.analyze_btn.set_enabled(True)
        self.live_indicator.config(text="● LIVE", fg=Theme.SUCCESS)
        self.loading_label.config(text="")

        primary = result["primary_emotion"]
        theme_meta = Theme.EMOTION_THEMES.get(primary, Theme.EMOTION_THEMES["neutral"])

        # Emotion-reactive theme styling for result card
        self.emotion_card.config(highlightbackground=theme_meta["color"])
        self.hero_emoji.config(text=result["primary_emoji"])
        self.emotion_title.config(text=result["display_name"], fg=theme_meta["color"])

        sentiment_type = result["sentiment_type"]
        self.sentiment_badge.config(
            text=f"Sentiment: {sentiment_type}",
            bg=theme_meta["badge_bg"],
            fg=Theme.TEXT_PRIMARY
        )

        conf = result["primary_confidence"]
        self.confidence_badge.config(text=f"Confidence: {conf}%")
        self.explanation_label.config(text=result["explanation"])

        # Update Emoji Recommendations
        self._display_recommended_emojis(result["recommended_emojis"])

        # Update Probability Distribution Bars
        dist = result.get("distribution", {})
        for emotion_key, bar in self.bar_rows.items():
            pct = dist.get(emotion_key, 0.0)
            bar["pct_lbl"].config(text=f"{pct:.1f}%")

            canvas = bar["canvas"]
            canvas.delete("all")
            width = canvas.winfo_width()
            if width <= 1:
                width = 180
            fill_w = max(2, int((pct / 100.0) * width))
            canvas.create_rectangle(0, 0, fill_w, 12, fill=bar["color"], width=0)

    def _display_recommended_emojis(self, emojis: list):
        for widget in self.recs_frame.winfo_children():
            widget.destroy()

        for emoji in emojis:
            btn = tk.Button(
                self.recs_frame,
                text=emoji,
                font=(Theme.FONT_EMOJI, 20),
                bg=Theme.BG_INPUT,
                fg=Theme.TEXT_PRIMARY,
                activebackground=Theme.BG_CARD_HOVER,
                activeforeground=Theme.TEXT_HIGHLIGHT,
                relief="solid",
                bd=1,
                highlightthickness=1,
                highlightbackground=Theme.BORDER_LIGHT,
                padx=8,
                pady=4,
                cursor="hand2",
                command=lambda e=emoji: self.copy_to_clipboard(e)
            )
            btn.pack(side="left", padx=4)

    def copy_to_clipboard(self, emoji: str):
        try:
            self.parent.clipboard_clear()
            self.parent.clipboard_append(emoji)
            self.copy_feedback.config(text=f"Copied {emoji} ✓")
            self.parent.after(2200, lambda: self.copy_feedback.config(text=""))
        except Exception:
            pass

    def handle_back(self):
        self.go_back()