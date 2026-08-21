import importlib
for mod in ["mediapipe", "pyttsx3"]:
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        print(f"{mod}: INSTALLED ({v})", flush=True)
    except Exception as e:
        print(f"{mod}: MISSING -> {e}", flush=True)
