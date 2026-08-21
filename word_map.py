"""
Maps Tunisian sign-language class labels (folder names) to spoken Arabic words/phrases.
Voice spelling uses gTTS (Google TTS) which supports Arabic (ar).
"""
import hashlib
import os

# Dictionary of known labels -> Arabic phrase for speech.
# If a label is not in here, we fall back to a transliteration guess.
WORD_MAP = {
    "3aslema": "عسلمة",
    "3ayla": "عائلة",
    "5adamet": "خدمة",
    "5al-3am": "خال عم",
    "5mis": "خميس",
    "5ou": "خو",
    "a7ad": "أحد",
    "assam": "آسم",
    "baladya": "بلدية",
    "banka": "بنكة",
    "barnamjk": "برامج",
    "bent": "بنت",
    "bou": "بو",
    "bousta": "بوسته",
    "car": "كار",
    "chabeb": "شباب",
    "cv": "سي في",
    "dar": "دار",
    "demande": "ديموند",
    "eben": "ابن",
    "enti": "أنتي",
    "erb3a": "أربعة",
    "jad": "جد",
    "jadda": "جدة",
    "jom3a": "جمعة",
    "karhba": "كرهبة",
    "labes": "لاباس",
    "louage": "لواج",
    "lyoum": "اليوم",
    "ma7kma": "محكمة",
    "mar2a": "مرأة",
    "mar7ba": "مرحبا",
    "metro": "مترو",
    "mostawsaf": "مستوصف",
    "n3awnek": "نعاونك",
    "nekteblk": "نكتبلك",
    "non": "لا",
    "o5t": "أخت",
    "om": "أم",
    "oui": "نعم",
    "radio": "راديو",
    "sbitar": "سبيطار",
    "se7a": "صحة",
    "sebt": "سبت",
    "siye7a": "سياحة",
    "t7eb": "تحب",
    "ta3lim": "تعليم",
    "ta3raf": "تعرف",
    "ta9ra": "تقرأ",
    "taxi": "تاكسي",
    "telvza": "تلفزة",
    "tfol": "طفل",
    "tha9afa": "ثقافة",
    "thleth": "ثلاثة",
    "thnin": "اثنين",
    "train": "تران",
    "wzara": "وزارة",
}

# Default fallback for uncleared labels.
DEFAULT_ARABIC = "إشارة"


def to_arabic(label):
    """Return the Arabic phrase for a class label (fallback to DEFAULT_ARABIC)."""
    return WORD_MAP.get(str(label).strip().lower(), DEFAULT_ARABIC)


def speak(text, lang="ar"):
    """
    Speak a string using gTTS (Google Text-to-Speech), streaming to a temp file.
    Raises ImportError if gtts is not installed.
    """
    try:
        from gtts import gTTS
    except ImportError as e:
        raise ImportError("gTTS not installed. Run: pip install gTTS") from e

    try:
        import pygame
    except ImportError:
        pygame = None

    # Cache audio by text+lang hash to avoid hitting the network each time.
    token = hashlib.md5(f"{text}|{lang}".encode("utf-8")).hexdigest()
    path = os.path.join("audio_cache", f"{token}.mp3")
    os.makedirs("audio_cache", exist_ok=True)

    if not os.path.exists(path):
        tts = gTTS(text=text, lang=lang)  # lang='ar' for Arabic
        tts.save(path)

    # Play the audio.
    if pygame:
        import time
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)
        pygame.mixer.quit()
    else:
        # Fallback: open with OS default player.
        import subprocess
        import sys
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa
        else:
            subprocess.Popen(["xdg-open", path])


def speak_sign(label):
    """Speak the Arabic phrase for a predicted sign label."""
    speak(to_arabic(label))


if __name__ == "__main__":
    # Quick test: speak a few labels.
    for lbl in ["3aslema", "enti", "car", "non"]:
        print(lbl, "->", to_arabic(lbl))
        speak_sign(lbl)
