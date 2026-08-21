import pyttsx3
try:
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    print("Available voices:")
    for v in voices:
        print(f"  - {v.name} | id={v.id} | lang={getattr(v, 'languages', None)}", flush=True)
except Exception as e:
    print("Error initializing TTS:", e, flush=True)
