import os
os.environ["OPEN3D_GUI_BACKEND"] = "GLFW"

import cv2
import numpy as np
from gesture_engine import GestureEngine
from ai_narrator    import AINarrator

# --- Init ---
cap      = cv2.VideoCapture(0)
gesture_engine = GestureEngine()
ai       = AINarrator(model="llama3")

# Map gestures to AI questions
GESTURE_PROMPTS = {
    "interior_view" : "Describe the interior of a modern car cabin.",
    "pinch_open"    : "What happens to a car's aerodynamics at larger scale?",
    "reset"         : "Give me one fun fact about car design.",
}

last_gesture = "none"

def blend_3d_over_frame(bg_bgr, fg_rgba):
    """Alpha-blend Panda3D RGBA onto webcam BGR frame."""
    fg_bgr = cv2.cvtColor(fg_rgba[:,:,:3], cv2.COLOR_RGB2BGR)
    alpha  = fg_rgba[:,:,3:4].astype(np.float32) / 255.0
    bg_f   = bg_bgr.astype(np.float32)
    fg_f   = fg_bgr.astype(np.float32)
    out    = fg_f * alpha + bg_f * (1 - alpha)
    return out.astype(np.uint8)

def draw_hud(frame, gesture, scale, ai_text):
    h, w = frame.shape[:2]
    cv2.putText(frame, f"Gesture: {gesture}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,180), 2)
    cv2.putText(frame, f"Scale: {scale:.2f}x",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
    # AI caption at bottom
    if ai_text:
        lines = [ai_text[i:i+70] for i in range(0, min(len(ai_text),140), 70)]
        for j, line in enumerate(lines):
            cv2.putText(frame, line,
                        (10, h - 40 + j*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,100), 1)

# NOTE: Panda3D requires its own main loop.
# For a standalone demo without Panda3D integration,
# replace car_renderer with a placeholder color rect.
scale = 1.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    gesture, scale_delta, rotate_delta = gesture_engine.detect(frame)
    scale = max(0.3, min(3.0, scale + scale_delta))

    # Trigger AI on new gesture
    if gesture != last_gesture and gesture in GESTURE_PROMPTS:
        ai.ask(GESTURE_PROMPTS[gesture])
    last_gesture = gesture

    # (Blend 3D car here if using Panda3D)
    draw_hud(frame, gesture, scale, ai.get_reply())

    cv2.imshow("AR Car — Hand Gesture AI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()