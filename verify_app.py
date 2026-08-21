"""Verify all new modules import and basic functions work."""
print("=== VERIFY START ===", flush=True)

# 1. hand_crop
try:
    import hand_crop
    print("hand_crop: OK, hand_available =", hand_crop.hand_available(), flush=True)
except Exception as e:
    print("hand_crop FAILED:", e, flush=True)

# 2. word_map
try:
    import word_map
    print("word_map: OK, 3aslema ->", word_map.to_arabic("3aslema"), flush=True)
except Exception as e:
    print("word_map FAILED:", e, flush=True)

# 3. app_gradio imports (model load)
try:
    import app_gradio as app
    print("app_gradio: OK, classes =", len(app.class_names), flush=True)
    print("has car:", "car" in app.class_names, flush=True)
except Exception as e:
    print("app_gradio FAILED:", e, flush=True)

print("=== VERIFY END ===", flush=True)
