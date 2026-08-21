"""
Tunisian Sign Language Recognition - Web App (Gradio)
Apple-inspired UI: soft glass cards, SF-style type, animated ring countdown,
snappy button interactions.

Features:
  - Upload / webcam image -> predicted sign as text + confidence.
  - Animated 3-second circular countdown before webcam capture.
  - Hand cropping (MediaPipe) so the model focuses only on the hand.
  - Voice spelling of the predicted sign (gTTS, Arabic).
  - Sentence prediction mode: predict a sequence of images (3s apart) and combine
    into a sentence.

Run with:
    python app_gradio.py
"""
import os
import math
import time
import threading
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from hand_crop import crop_hand_pil, hand_available
import word_map

# =========================
# CONFIGURATION
# =========================
MODEL_PATH = "models/tsl_model.h5"
DATA_PATH = "Data/raw"
IMG_SIZE = 224
TOP_K = 3         # show top 3 predictions
COUNTDOWN = 3      # seconds before webcam capture
COUNTDOWN_STEPS_PER_SEC = 20   # animation smoothness (higher = smoother ring)
SENTENCE_GAP = 3   # seconds between each image in sentence mode

# =========================
# DISCOVER CLASSES
# =========================
def get_class_names(data_path):
    """Discover classes from training data folders (sorted for consistency)."""
    return sorted(
        [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    )

# =========================
# LOAD MODEL (once at startup)
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}\nTrain first using: python main.py")

model = load_model(MODEL_PATH)
print("[OK] Model loaded.")

class_names = get_class_names(DATA_PATH)
if not class_names:
    raise FileNotFoundError(f"No class folders found in: {DATA_PATH}")
print(f"[OK] {len(class_names)} classes loaded.")
print(f"[OK] Hand crop available: {hand_available()}")

# =========================
# PREPROCESS WITH HAND CROP
# =========================
def preprocess_pil(img):
    """Convert PIL image -> cropped hand -> preprocessed 224x224 array."""
    img = img.convert("RGB")
    # Crop tight around the hand so the model ignores background.
    img = crop_hand_pil(img)
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# =========================
# PREDICTION
# =========================
def predict_pil(img):
    """Predict a PIL image, returning probabilities vector."""
    arr = preprocess_pil(img)
    return model.predict(arr, verbose=0)[0]

def format_result(probs):
    """Build display outputs from a probability vector, Apple-card styled."""
    top_indices = probs.argsort()[-TOP_K:][::-1]
    top_class = class_names[top_indices[0]]
    top_confidence = probs[top_indices[0]]

    confidences = {}
    rows = []
    for i, idx in enumerate(top_indices):
        label = class_names[idx]
        conf = float(probs[idx])
        confidences[label] = conf
        pct = conf * 100
        rank_class = "rank-1" if i == 0 else "rank-other"
        rows.append(
            f"""
            <div class="result-row {rank_class}" style="animation-delay:{i * 90}ms">
                <span class="result-rank">{i+1}</span>
                <span class="result-label">{label}</span>
                <div class="result-bar-track">
                    <div class="result-bar-fill" style="width:{pct:.1f}%"></div>
                </div>
                <span class="result-pct">{conf:.1%}</span>
            </div>
            """
        )

    summary = f"""
    <div class="hero-result fade-in-up">
        <div class="hero-badge">✓ Detected</div>
        <div class="hero-label">{top_class}</div>
        <div class="hero-confidence">{top_confidence:.1%} confidence</div>
    </div>
    <div class="result-list">
        {''.join(rows)}
    </div>
    """
    return summary, top_class, confidences

def predict_sign(img):
    """Single image prediction -> display outputs."""
    img = img.convert("RGB")
    probs = predict_pil(img)
    summary, top_class, confidences = format_result(probs)
    return summary, top_class, confidences, img

# =========================
# VOICE SPELLING
# =========================
def speak_sign(label):
    """Speak the Arabic phrase for the predicted sign (in a background thread)."""
    try:
        t = threading.Thread(target=word_map.speak_sign, args=(label,), daemon=True)
        t.start()
    except Exception as e:
        print("[WARN] Voice failed:", e)

# =========================
# ANIMATED CIRCULAR COUNTDOWN
# =========================
RING_RADIUS = 54
RING_CIRCUMFERENCE = 2 * math.pi * RING_RADIUS

def build_countdown_ring(remaining_label, frac, capturing=False, message="Get ready..."):
    """Return an HTML/SVG ring that fills smoothly as the countdown progresses."""
    frac = max(0.0, min(1.0, frac))
    offset = RING_CIRCUMFERENCE * (1 - frac)
    center_content = (
        '<span class="ring-flash">📸</span>' if capturing
        else f'<span class="ring-number">{remaining_label}</span>'
    )
    ring_class = "countdown-ring capturing" if capturing else "countdown-ring"
    return f"""
    <div class="countdown-wrap fade-in-up">
        <div class="{ring_class}">
            <svg width="140" height="140" viewBox="0 0 140 140">
                <circle class="ring-track" cx="70" cy="70" r="{RING_RADIUS}" />
                <circle class="ring-fill" cx="70" cy="70" r="{RING_RADIUS}"
                    style="stroke-dasharray:{RING_CIRCUMFERENCE:.2f};
                           stroke-dashoffset:{offset:.2f};" />
            </svg>
            <div class="ring-center">{center_content}</div>
        </div>
        <div class="countdown-caption">{message}</div>
    </div>
    """

def build_countdown_idle_message(text):
    return f"""<div class="countdown-wrap fade-in-up">
        <div class="empty-state">{text}</div>
    </div>"""

# =========================
# COUNTDOWN + PREDICT (generator: animates the ring live)
# =========================
def capture_and_predict(img):
    """
    Generator: streams an animated countdown ring for COUNTDOWN seconds,
    then runs the prediction. Also toggles button interactivity so people
    can't double-fire captures mid-countdown.
    """
    no_change = gr.update()
    disable_btns = gr.update(interactive=False)
    enable_btns = gr.update(interactive=True)

    if img is None:
        yield (
            build_countdown_idle_message("Please add an image first (upload or webcam)."),
            no_change, no_change, no_change, no_change,
            enable_btns, enable_btns,
        )
        return

    total_ticks = COUNTDOWN * COUNTDOWN_STEPS_PER_SEC
    for tick in range(total_ticks):
        elapsed = tick / COUNTDOWN_STEPS_PER_SEC
        frac = elapsed / COUNTDOWN
        remaining = max(1, COUNTDOWN - int(elapsed))
        ring_html = build_countdown_ring(remaining, frac, message="Hold your sign steady...")
        yield (
            ring_html, no_change, no_change, no_change, no_change,
            disable_btns, disable_btns,
        )
        time.sleep(1 / COUNTDOWN_STEPS_PER_SEC)

    # Flash "capturing" state briefly
    yield (
        build_countdown_ring("", 1.0, capturing=True, message="Capturing..."),
        no_change, no_change, no_change, no_change,
        disable_btns, disable_btns,
    )
    time.sleep(0.35)

    summary, top_class, confidences, out_img = predict_sign(img)
    yield (
        build_countdown_idle_message("Ready when you are — press Capture again anytime."),
        summary, top_class, confidences, out_img,
        enable_btns, enable_btns,
    )

# =========================
# SENTENCE PREDICTION MODE
# =========================
def predict_sentence(images, speak_each, progress=gr.Progress()):
    """
    images: list of PIL images.
    Predicts each (3s apart), then combines the top signs into a sentence.
    """
    if not images:
        return "<div class='empty-state'>No images provided.</div>", ""

    words = []
    chips = []
    total = len(images)
    for idx, img in enumerate(images):
        progress(idx / total, desc=f"Predicting image {idx+1}/{total}...")
        probs = predict_pil(img)
        _, top_class, confidences = format_result(probs)
        words.append(top_class)
        chips.append(
            f"""<span class="word-chip fade-in-up" style="animation-delay:{idx * 100}ms">
                    {top_class} <em>{confidences.get(top_class, 0):.0%}</em>
                </span>"""
        )
        if speak_each:
            speak_sign(top_class)
        if idx < len(images) - 1:
            time.sleep(SENTENCE_GAP)  # 3s between each image

    sentence = " ".join(words)
    build = f"""
    <div class="hero-result fade-in-up">
        <div class="hero-badge">✓ Sentence built</div>
        <div class="hero-label">{sentence}</div>
        <div class="hero-confidence">{len(words)} signs combined</div>
    </div>
    <div class="chip-row">{''.join(chips)}</div>
    """
    return build, sentence

# =========================
# APPLE-STYLE THEME + CSS
# =========================
apple_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.gray,
    neutral_hue=gr.themes.colors.gray,
    font=[
        gr.themes.GoogleFont("Inter"),
        "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif",
    ],
    font_mono=[
        gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace",
    ],
).set(
    body_background_fill="#f5f5f7",
    body_background_fill_dark="#000000",
    block_background_fill="rgba(255,255,255,0.72)",
    block_background_fill_dark="rgba(28,28,30,0.72)",
    block_border_width="1px",
    block_border_color="rgba(0,0,0,0.06)",
    block_border_color_dark="rgba(255,255,255,0.08)",
    block_radius="22px",
    block_shadow="0 8px 30px rgba(0,0,0,0.06)",
    block_label_text_weight="600",
    block_title_text_weight="700",
    button_primary_background_fill="#0071e3",
    button_primary_background_fill_hover="#0077ed",
    button_primary_text_color="#ffffff",
    button_primary_border_color="transparent",
    button_secondary_background_fill="rgba(0,0,0,0.05)",
    button_secondary_background_fill_hover="rgba(0,0,0,0.09)",
    button_secondary_text_color="#1d1d1f",
    input_background_fill="rgba(255,255,255,0.6)",
    input_radius="16px",
    layout_gap="20px",
)

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    background: radial-gradient(circle at 20% -10%, #eaf3ff 0%, #f5f5f7 45%, #f5f5f7 100%) !important;
}

.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }

/* ---- Hero header ---- */
#app-header {
    text-align: center;
    padding: 48px 12px 8px 12px;
    animation: fadeInDown 0.7s ease both;
}
#app-header h1 {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #0071e3, #42a5f5 60%, #7dd3fc);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
#app-header p {
    font-size: 1.05rem;
    color: #6e6e73;
    font-weight: 500;
    max-width: 640px;
    margin: 0 auto;
}
#app-header .header-badges {
    display: flex; justify-content: center; gap: 8px; margin-top: 14px; flex-wrap: wrap;
}
.header-chip {
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.02em;
    padding: 6px 12px; border-radius: 999px;
    background: rgba(0,113,227,0.09); color: #0071e3;
    border: 1px solid rgba(0,113,227,0.18);
}

/* ---- Tabs ---- */
.tabs { animation: fadeInUp 0.6s ease both; animation-delay: 0.1s; }
.tab-nav button {
    font-weight: 600 !important;
    border-radius: 999px !important;
    transition: transform 0.12s cubic-bezier(.4,0,.2,1), background 0.12s ease !important;
}

/* ---- Cards / blocks ---- */
.block {
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    transition: box-shadow 0.3s ease;
}
.block:hover { box-shadow: 0 14px 40px rgba(0,0,0,0.09) !important; }

/* ---- Buttons: snappy, high-performance feel ---- */
button {
    transition: transform 0.1s cubic-bezier(.4,0,.2,1),
                box-shadow 0.14s ease,
                background 0.14s ease,
                opacity 0.14s ease !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
    will-change: transform;
    transform: translateZ(0);
}
button:hover { transform: translateY(-1px); }
button:active { transform: scale(0.95) translateY(0); }
button:disabled { opacity: 0.45; cursor: not-allowed; transform: none !important; }

.primary:hover, button[variant="primary"]:hover {
    box-shadow: 0 8px 20px rgba(0,113,227,0.35) !important;
}

/* ---- Capture button: unmistakable, high-visibility ---- */
#capture-btn {
    background: linear-gradient(135deg, #ff453a, #ff8a3d) !important;
    color: #fff !important;
    font-size: 1.05rem !important;
    padding: 14px 22px !important;
    border-radius: 18px !important;
    box-shadow: 0 6px 18px rgba(255, 69, 58, 0.35) !important;
    position: relative;
    overflow: visible;
}
#capture-btn::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 18px;
    box-shadow: 0 0 0 0 rgba(255, 69, 58, 0.55);
    animation: capturePulse 2.2s ease-out infinite;
    pointer-events: none;
}
#capture-btn:hover {
    box-shadow: 0 10px 26px rgba(255, 69, 58, 0.45) !important;
    transform: translateY(-2px);
}
#capture-btn:active { transform: scale(0.95); }
#capture-btn:disabled::before { animation: none; box-shadow: none; }

@keyframes capturePulse {
    0%   { box-shadow: 0 0 0 0 rgba(255, 69, 58, 0.45); }
    70%  { box-shadow: 0 0 0 16px rgba(255, 69, 58, 0); }
    100% { box-shadow: 0 0 0 0 rgba(255, 69, 58, 0); }
}

#predict-btn {
    box-shadow: 0 6px 16px rgba(0,113,227,0.25) !important;
}

#voice-btn {
    border-radius: 999px !important;
}

/* ---- Image / gallery frames ---- */
.image-container, .gr-image, .thumbnail-item {
    border-radius: 20px !important;
    overflow: hidden;
    transition: transform 0.3s ease;
}

/* ---- Animated countdown ring ---- */
.countdown-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 14px;
    padding: 26px 20px;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(0,113,227,0.06), rgba(255,138,61,0.06));
    border: 1px solid rgba(0,0,0,0.05);
    margin-bottom: 14px;
}
.countdown-ring { position: relative; width: 140px; height: 140px; }
.countdown-ring svg { transform: rotate(-90deg); }
.ring-track {
    fill: none;
    stroke: rgba(0,0,0,0.08);
    stroke-width: 10;
}
.ring-fill {
    fill: none;
    stroke: #0071e3;
    stroke-width: 10;
    stroke-linecap: round;
    transition: stroke-dashoffset 0.05s linear;
}
.countdown-ring.capturing .ring-fill { stroke: #ff453a; }
.ring-center {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
}
.ring-number {
    font-size: 2.4rem;
    font-weight: 800;
    color: #1d1d1f;
    animation: ringTick 0.35s ease both;
}
.ring-flash {
    font-size: 2.6rem;
    animation: flashPop 0.4s cubic-bezier(.4,0,.2,1) both;
}
.countdown-caption {
    font-weight: 600;
    color: #6e6e73;
    font-size: 0.92rem;
    letter-spacing: 0.01em;
}
@keyframes ringTick {
    from { opacity: 0; transform: scale(1.3); }
    to { opacity: 1; transform: scale(1); }
}
@keyframes flashPop {
    0% { transform: scale(0.6); opacity: 0; }
    60% { transform: scale(1.15); opacity: 1; }
    100% { transform: scale(1); opacity: 1; }
}

/* ---- Result markdown (custom HTML we render) ---- */
.hero-result {
    text-align: center;
    padding: 28px 20px 22px 20px;
    margin-bottom: 14px;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(0,113,227,0.08), rgba(125,211,252,0.10));
    border: 1px solid rgba(0,113,227,0.12);
}
.hero-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #1a9e4e;
    background: rgba(52, 199, 89, 0.12);
    border: 1px solid rgba(52, 199, 89, 0.3);
    padding: 4px 10px;
    border-radius: 999px;
    margin-bottom: 10px;
}
.hero-label {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #1d1d1f;
}
.hero-confidence {
    margin-top: 4px;
    font-size: 0.95rem;
    font-weight: 600;
    color: #0071e3;
}

.result-list { display: flex; flex-direction: column; gap: 10px; }
.result-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    border-radius: 14px;
    background: rgba(255,255,255,0.55);
    border: 1px solid rgba(0,0,0,0.05);
    opacity: 0;
    animation: fadeInUp 0.5s ease forwards;
}
.result-row.rank-1 { border-color: rgba(0,113,227,0.35); background: rgba(0,113,227,0.06); }
.result-rank {
    width: 22px; height: 22px;
    border-radius: 50%;
    background: #0071e3;
    color: #fff;
    font-size: 0.75rem;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.result-row.rank-other .result-rank { background: #c7c7cc; }
.result-label { font-weight: 600; min-width: 110px; color: #1d1d1f; }
.result-bar-track {
    flex: 1;
    height: 8px;
    border-radius: 999px;
    background: rgba(0,0,0,0.06);
    overflow: hidden;
}
.result-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #0071e3, #64b5f6);
    width: 0%;
    animation: growBar 0.9s cubic-bezier(.4,0,.2,1) forwards;
    animation-delay: 0.15s;
}
.result-pct { font-weight: 700; font-size: 0.85rem; color: #6e6e73; min-width: 48px; text-align: right; }

.chip-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 6px; }
.word-chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 8px 14px;
    border-radius: 999px;
    background: rgba(0,113,227,0.08);
    border: 1px solid rgba(0,113,227,0.2);
    font-weight: 600;
    color: #1d1d1f;
    opacity: 0;
    animation: fadeInUp 0.5s ease forwards;
}
.word-chip em { font-style: normal; color: #0071e3; font-weight: 700; font-size: 0.85em; }

.empty-state {
    text-align: center;
    padding: 30px;
    color: #6e6e73;
    font-weight: 500;
}

/* ---- Instructions strip ---- */
.how-it-works {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    justify-content: center;
    margin: 4px 0 22px 0;
}
.how-step {
    flex: 1 1 200px;
    max-width: 260px;
    padding: 14px 16px;
    border-radius: 16px;
    background: rgba(255,255,255,0.55);
    border: 1px solid rgba(0,0,0,0.05);
    text-align: center;
}
.how-step .num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 24px; height: 24px; border-radius: 50%;
    background: #0071e3; color: #fff; font-weight: 800; font-size: 0.8rem;
    margin-bottom: 8px;
}
.how-step .txt { font-size: 0.85rem; font-weight: 600; color: #3a3a3c; }

/* ---- Keyframes ---- */
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-14px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(14px); }
    to { opacity: 1; transform: translateY(0); }
}
.fade-in-up { animation: fadeInUp 0.5s ease both; }
@keyframes growBar { from { width: 0%; } }

/* ---- Scrollbar polish ---- */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.15); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,0,0,0.25); }
"""

# =========================
# GRADIO INTERFACE (Blocks)
# =========================
examples = [os.path.join("Data/test", f) for f in os.listdir("Data/test")]

with gr.Blocks(
    title="Tunisian Sign Language Recognition",
    theme=apple_theme,
    css=CUSTOM_CSS,
) as demo:

    gr.HTML(
        f"""
        <div id="app-header">
            <h1>Tunisian Sign Language</h1>
            <p>Upload or capture a gesture and let the model translate it instantly — with live
            confidence scoring{" and automatic hand cropping" if hand_available() else ""}.</p>
            <div class="header-badges">
                <span class="header-chip">{len(class_names)} signs supported</span>
                <span class="header-chip">Live webcam capture</span>
                <span class="header-chip">Voice playback</span>
            </div>
        </div>
        """
    )

    gr.HTML(
        """
        <div class="how-it-works">
            <div class="how-step"><div class="num">1</div><div class="txt">Upload a photo or open your webcam</div></div>
            <div class="how-step"><div class="num">2</div><div class="txt">Hit Capture and hold your sign through the countdown</div></div>
            <div class="how-step"><div class="num">3</div><div class="txt">Get the sign, confidence score, and hear it spoken aloud</div></div>
        </div>
        """
    )

    # ---- Single image tab ----
    with gr.Tab("Single Image"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                img_input = gr.Image(
                    type="pil",
                    sources=["upload", "webcam"],
                    label="Upload or take a photo (webcam)",
                    height=380,
                )
                with gr.Row():
                    take_btn = gr.Button(
                        f"📸 Capture & Predict ({COUNTDOWN}s)",
                        elem_id="capture-btn",
                        scale=3,
                    )
                    predict_btn = gr.Button(
                        "Predict Now", elem_id="predict-btn", variant="primary", scale=2
                    )
                    voice_btn = gr.Button("🔊 Speak", elem_id="voice-btn", scale=1)
                voice_state = gr.State("")

                countdown_html = gr.HTML(
                    value="",
                    visible=True,
                )
            with gr.Column(scale=5):
                output_md = gr.HTML(
                    value="<div class='empty-state'>Your prediction will appear here.</div>",
                    label="Prediction",
                )
                output_text = gr.Textbox(label="Predicted Sign (text)", interactive=False)
                output_label = gr.Label(num_top_classes=TOP_K, label="Confidence", visible=False)
                output_img = gr.Image(type="pil", label="Input Image", height=220)

        gr.Examples(examples=examples, inputs=img_input)

        predict_btn.click(
            predict_sign,
            inputs=img_input,
            outputs=[output_md, output_text, output_label, output_img],
        )
        take_btn.click(
            capture_and_predict,
            inputs=img_input,
            outputs=[
                countdown_html, output_md, output_text, output_label, output_img,
                take_btn, predict_btn,
            ],
        )
        # After prediction, store top class for voice button.
        predict_btn.click(
            lambda x: x, inputs=output_text, outputs=voice_state
        )
        take_btn.click(
            lambda x: x, inputs=output_text, outputs=voice_state
        )
        voice_btn.click(
            speak_sign, inputs=voice_state, outputs=None
        )

    # ---- Sentence prediction tab ----
    with gr.Tab("Sentence Prediction"):
        gr.Markdown(
            "Upload **multiple images** (one per sign). The model predicts each "
            f"one (about {SENTENCE_GAP}s apart) and combines the signs into a sentence."
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                sent_input = gr.Gallery(
                    label="Upload sign images (in order)",
                    columns=4,
                    height="auto",
                )
                speak_each = gr.Checkbox(label="Speak each sign", value=False)
                sent_btn = gr.Button("Build Sentence", elem_id="predict-btn", variant="primary")
            with gr.Column(scale=5):
                sent_md = gr.HTML(
                    value="<div class='empty-state'>Your sentence will appear here.</div>",
                    label="Sentence Result",
                )
                sent_text = gr.Textbox(label="Sentence (text)", interactive=False)

        sent_btn.click(
            predict_sentence,
            inputs=[sent_input, speak_each],
            outputs=[sent_md, sent_text],
        )

if __name__ == "__main__":
    demo.launch(share=False)