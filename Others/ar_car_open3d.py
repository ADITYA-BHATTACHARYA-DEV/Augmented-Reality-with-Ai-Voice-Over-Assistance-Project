# """
# AR Car — Open3D Edition (lighter, same quality)
# ================================================
# Features:
#   • Open3D offscreen 3D car rendering (CPU/GPU, fraction of Panda3D memory)
#   • MediaPipe hand gesture control (scale, orbit, interior view, reset)
#   • ArUco marker anchor (car stays glued to a physical card)
#   • Voice input  → Ollama local LLM narration
#   • Keyboard     → type questions to the AI
#   • AI captions overlaid on live webcam feed
#
# Requirements
# ------------
#   pip install opencv-python mediapipe open3d ollama numpy \
#               speechrecognition pyttsx3 pyaudio
#
# Ollama must be running:
#   ollama serve          (in a separate terminal)
#   ollama pull llama3    (once)
#
# 3D model:
#   Place any .obj or .ply car model as  car.obj  next to this file.
#   Free sources: Poly Haven, Sketchfab (free/CC), TurboSquid free.
#   If no model found, a coloured box placeholder is used automatically.
#
# Controls
# --------
#   Pinch close          → shrink car
#   Pinch open/spread    → enlarge car
#   2 fingers up         → look inside (camera enters cabin)
#   Open palm (4+ fin.)  → reset view & scale
#   Wrist left / right   → orbit car left / right
#   V key                → voice query (speak after beep)
#   Q key                → quit
#   Any other key        → type question (terminal prompt)
#
# ArUco:
#   Print or display  aruco_marker.png  (generated on first run).
#   Hold it up to camera — the car will anchor to it.
# """
# import os
# os.environ["OPEN3D_GUI_BACKEND"] = "GLFW"
# import cv2
# import numpy as np
# import mediapipe as mp
# import open3d as o3d
# import ollama
# import threading
# import math
# import os
# import time
# import sys
#
# # ── optional voice ──────────────────────────────────────────────────────────
# try:
#     import speech_recognition as sr
#     import pyttsx3
#     VOICE_OK = True
# except ImportError:
#     VOICE_OK = False
#     print("[WARN] speech_recognition / pyttsx3 not installed — voice disabled")
#
# # ═══════════════════════════════════════════════════════════════════════════
# #  CONFIG
# # ═══════════════════════════════════════════════════════════════════════════
# CAM_W, CAM_H   = 1280, 720          # webcam resolution
# RENDER_W       = 640                 # Open3D render width  (half-res for speed)
# RENDER_H       = 480                 # Open3D render height
# MODEL_PATH     = "car.obj"           # your 3D model file
# OLLAMA_MODEL   = "llava-phi3"            # ollama model name
# ARUCO_DICT     = cv2.aruco.DICT_6X6_250
# ARUCO_ID       = 0                   # which marker ID to track
# MARKER_SIZE_M  = 0.08                # physical marker size in metres (8 cm)
#
# # ═══════════════════════════════════════════════════════════════════════════
# #  GESTURE ENGINE
# # ═══════════════════════════════════════════════════════════════════════════
# class GestureEngine:
#     def __init__(self):
#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(
#             max_num_hands=2,
#             min_detection_confidence=0.75,
#             min_tracking_confidence=0.75,
#         )
#         self.draw = mp.solutions.drawing_utils
#         self.prev_pinch = None
#
#     @staticmethod
#     def _dist(a, b):
#         return math.hypot(a.x - b.x, a.y - b.y)
#
#     @staticmethod
#     def _fingers_up(lm):
#         tips   = [8, 12, 16, 20]
#         joints = [6, 10, 14, 18]
#         return sum(lm[t].y < lm[j].y for t, j in zip(tips, joints))
#
#     def process(self, bgr_frame):
#         """
#         Returns (gesture: str, scale_delta: float, rotate_delta: float)
#         Draws landmarks on bgr_frame in-place.
#         """
#         rgb    = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
#         result = self.hands.process(rgb)
#
#         gesture      = "none"
#         scale_delta  = 0.0
#         rotate_delta = 0.0
#
#         if not result.multi_hand_landmarks:
#             self.prev_pinch = None
#             return gesture, scale_delta, rotate_delta
#
#         for hand_lm in result.multi_hand_landmarks:
#             lm = hand_lm.landmark
#             self.draw.draw_landmarks(
#                 bgr_frame, hand_lm, self.mp_hands.HAND_CONNECTIONS,
#                 self.draw.DrawingSpec(color=(0, 200, 150), thickness=2),
#                 self.draw.DrawingSpec(color=(0, 255, 100), thickness=1),
#             )
#
#             pinch    = self._dist(lm[4], lm[8])
#             fingers  = self._fingers_up(lm)
#             wrist_x  = lm[0].x
#
#             # ── scale via pinch ──
#             if self.prev_pinch is not None:
#                 delta = pinch - self.prev_pinch
#                 if abs(delta) > 0.005:
#                     scale_delta = delta * 1.8   # sensitivity
#             self.prev_pinch = pinch
#
#             # ── gesture classify ──
#             if fingers >= 4:
#                 gesture = "reset"
#             elif fingers == 2:
#                 gesture = "interior_view"
#             elif pinch < 0.06:
#                 gesture = "pinch_close"
#             elif pinch > 0.20:
#                 gesture = "pinch_open"
#
#             # ── orbit via wrist x-position ──
#             if wrist_x < 0.30:
#                 rotate_delta = -4.0
#             elif wrist_x > 0.70:
#                 rotate_delta = +4.0
#
#         return gesture, scale_delta, rotate_delta
#
#
# # ═══════════════════════════════════════════════════════════════════════════
# #  OPEN3D CAR RENDERER  (offscreen — lightweight!)
# # ═══════════════════════════════════════════════════════════════════════════
# class CarRenderer:
#     """
#     Renders a 3D car model offscreen with Open3D and returns
#     an RGBA NumPy array ready for alpha-blending onto the webcam frame.
#
#     Why Open3D instead of Panda3D?
#       • No full game-engine overhead  → ~4× less RAM, ~2× less CPU
#       • Pure offscreen — no window needed
#       • Simple pip install, no binary dependencies
#       • Same visual quality for static/rotated meshes
#     """
#
#     def __init__(self, width=RENDER_W, height=RENDER_H, model_path=MODEL_PATH):
#         self.w = width
#         self.h = height
#         self.yaw   = 0.0
#         self.scale = 1.0
#         self.interior = False
#
#         # ── renderer ──
#         self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
#         scene = self.renderer.scene
#         scene.set_background([0, 0, 0, 0])   # transparent bg
#
#         # ── lighting ──
#         scene.scene.set_sun_light(
#             [0.5, -1.0, -0.8],   # direction
#             [1.2, 1.2, 1.2],     # colour
#             80000                 # intensity
#         )
#         scene.scene.enable_sun_light(True)
#         scene.scene.enable_indirect_lighting(True)
#
#         # ── material ──
#         mat = o3d.visualization.rendering.MaterialRecord()
#         mat.shader = "defaultLit"
#         mat.base_color = [0.85, 0.12, 0.12, 1.0]   # red car default
#         mat.base_roughness = 0.35
#         mat.base_metallic  = 0.6
#
#         # ── load model or create placeholder ──
#         if os.path.exists(model_path):
#             mesh = o3d.io.read_triangle_mesh(model_path)
#             mesh.compute_vertex_normals()
#             print(f"[3D] Loaded {model_path}")
#         else:
#             print(f"[3D] {model_path} not found — using placeholder box")
#             mesh = self._make_car_placeholder()
#
#         # Normalise to unit size
#         bbox   = mesh.get_axis_aligned_bounding_box()
#         extent = bbox.get_max_bound() - bbox.get_min_bound()
#         mesh.translate(-bbox.get_center())
#         mesh.scale(2.0 / max(extent), center=[0, 0, 0])
#
#         scene.add_geometry("car", mesh, mat)
#         self._mesh_name = "car"
#
#         # ── camera defaults ──
#         self._set_camera_exterior()
#
#     # ── placeholder geometry ─────────────────────────────────────────────
#     @staticmethod
#     def _make_car_placeholder():
#         """Simple box-based car silhouette when no .obj is provided."""
#         body    = o3d.geometry.TriangleMesh.create_box(2.0, 1.0, 0.6)
#         roof    = o3d.geometry.TriangleMesh.create_box(1.2, 0.95, 0.5)
#         w1      = o3d.geometry.TriangleMesh.create_sphere(0.25)
#         w2      = o3d.geometry.TriangleMesh.create_sphere(0.25)
#         w3      = o3d.geometry.TriangleMesh.create_sphere(0.25)
#         w4      = o3d.geometry.TriangleMesh.create_sphere(0.25)
#
#         body.translate([-1.0, -0.5, 0.0])
#         roof.translate([-0.6, -0.475, 0.6])
#         w1.translate([-0.7, -0.6,  -0.25])
#         w2.translate([ 0.7, -0.6,  -0.25])
#         w3.translate([-0.7,  0.6,  -0.25])
#         w4.translate([ 0.7,  0.6,  -0.25])
#
#         mesh = body + roof + w1 + w2 + w3 + w4
#         mesh.paint_uniform_color([0.85, 0.12, 0.12])
#         mesh.compute_vertex_normals()
#         return mesh
#
#     # ── camera positions ─────────────────────────────────────────────────
#     def _set_camera_exterior(self):
#         self.renderer.setup_camera(
#             60,                          # fov degrees
#             [0, 0, 0],                   # look-at
#             [0, -4.5, 1.5],              # eye position
#             [0, 0, 1],                   # up vector
#         )
#
#     def _set_camera_interior(self):
#         self.renderer.setup_camera(
#             90,
#             [0.3, 0.1, 0.3],
#             [0.0, -0.3, 0.5],
#             [0, 0, 1],
#         )
#
#     # ── update state & render ─────────────────────────────────────────────
#     def update(self, gesture, scale_delta, rotate_delta):
#         self.scale = max(0.3, min(3.5, self.scale + scale_delta))
#         self.yaw  += rotate_delta
#
#         # Apply transform to geometry via scene update
#         T = np.eye(4)
#         # rotation around Z
#         rad = math.radians(self.yaw)
#         T[0, 0] =  math.cos(rad)
#         T[0, 1] = -math.sin(rad)
#         T[1, 0] =  math.sin(rad)
#         T[1, 1] =  math.cos(rad)
#         # scale
#         T[0, 0] *= self.scale
#         T[1, 1] *= self.scale
#         T[2, 2]  = self.scale
#
#         self.renderer.scene.set_geometry_transform(self._mesh_name, T)
#
#         if gesture == "interior_view" and not self.interior:
#             self._set_camera_interior()
#             self.interior = True
#         elif gesture == "reset":
#             self._set_camera_exterior()
#             self.interior = False
#             self.scale    = 1.0
#             self.yaw      = 0.0
#
#     def render_rgba(self):
#         """Return H×W×4 uint8 RGBA array."""
#         img = self.renderer.render_to_image()
#         arr = np.asarray(img)                 # RGB float32 or uint8
#         if arr.dtype != np.uint8:
#             arr = (arr * 255).clip(0, 255).astype(np.uint8)
#         # Add alpha channel: black pixels → transparent
#         grey  = arr.mean(axis=2)
#         alpha = np.where(grey < 8, 0, 255).astype(np.uint8)
#         rgba  = np.dstack([arr, alpha])
#         return rgba
#
#
# # ═══════════════════════════════════════════════════════════════════════════
# #  OLLAMA AI NARRATOR
# # ═══════════════════════════════════════════════════════════════════════════
# class AINarrator:
#     SYSTEM = (
#         "You are a knowledgeable AR car guide. "
#         "Answer in exactly 1-2 short sentences. "
#         "Be specific, vivid, and enthusiastic. "
#         "No bullet points. No markdown."
#     )
#     GESTURE_PROMPTS = {
#         "interior_view" : "Describe what I see looking inside a modern sports car cabin.",
#         "pinch_open"    : "What aerodynamic changes matter most at large car scale?",
#         "reset"         : "Give me one surprising fact about automotive design history.",
#         "reset_alt"     : "What makes a car's exterior silhouette iconic?",
#     }
#
#     def __init__(self, model=OLLAMA_MODEL):
#         self.model  = model
#         self.reply  = "👋  Show hand gestures to control the car. Press V for voice."
#         self.busy   = False
#         self._lock  = threading.Lock()
#
#     def ask_async(self, prompt: str):
#         if self.busy:
#             return
#         self.busy = True
#         threading.Thread(target=self._query, args=(prompt,), daemon=True).start()
#
#     def _query(self, prompt):
#         try:
#             res = ollama.chat(
#                 model=self.model,
#                 messages=[
#                     {"role": "system",  "content": self.SYSTEM},
#                     {"role": "user",    "content": prompt},
#                 ],
#             )
#             with self._lock:
#                 self.reply = res["message"]["content"].strip()
#         except Exception as e:
#             with self._lock:
#                 self.reply = f"[AI unavailable — is ollama running? {e}]"
#         finally:
#             self.busy = False
#
#     def get_reply(self):
#         with self._lock:
#             return self.reply
#
#
# # ═══════════════════════════════════════════════════════════════════════════
# #  VOICE INPUT
# # ═══════════════════════════════════════════════════════════════════════════
# class VoiceInput:
#     def __init__(self):
#         if not VOICE_OK:
#             return
#         self.rec = sr.Recognizer()
#         self.mic = sr.Microphone()
#         self.tts = pyttsx3.init()
#         self.tts.setProperty("rate", 165)
#         with self.mic as src:
#             self.rec.adjust_for_ambient_noise(src, duration=0.5)
#
#     def listen(self):
#         """Blocking — call from thread. Returns transcribed text or None."""
#         if not VOICE_OK:
#             return None
#         try:
#             with self.mic as src:
#                 print("[Voice] Listening…")
#                 audio = self.rec.listen(src, timeout=5, phrase_time_limit=8)
#             text = self.rec.recognize_google(audio)
#             print(f"[Voice] Heard: {text}")
#             return text
#         except Exception as e:
#             print(f"[Voice] Error: {e}")
#             return None
#
#     def speak(self, text):
#         if not VOICE_OK:
#             return
#         threading.Thread(
#             target=lambda: (self.tts.say(text), self.tts.runAndWait()),
#             daemon=True,
#         ).start()
#
#
# # ═══════════════════════════════════════════════════════════════════════════
# #  ARUCO TRACKER
# # ═══════════════════════════════════════════════════════════════════════════
# class ArucoTracker:
#     def __init__(self):
#         self.adict  = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
#         self.params = cv2.aruco.DetectorParameters()
#         self.detector = cv2.aruco.ArucoDetector(self.adict, self.params)
#
#         # Camera intrinsics (approximate for 720p; calibrate for accuracy)
#         fx = fy = CAM_W * 0.85
#         cx, cy   = CAM_W / 2, CAM_H / 2
#         self.K   = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
#         self.D   = np.zeros((5, 1))
#
#         self.anchor_pos  = None   # (x, y) pixel anchor for car placement
#         self.anchor_size = None   # pixel size of marker → drives scale
#
#         self._generate_marker_image()
#
#     @staticmethod
#     def _generate_marker_image():
#         path = "aruco_marker.png"
#         if not os.path.exists(path):
#             adict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
#             img   = cv2.aruco.generateImageMarker(adict, ARUCO_ID, 300)
#             cv2.imwrite(path, img)
#             print(f"[ArUco] Marker saved → {path}  (print or display it)")
#
#     def detect(self, frame):
#         """Returns (anchor_px: tuple|None, marker_scale_hint: float)"""
#         grey   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         corners, ids, _ = self.detector.detectMarkers(grey)
#
#         if ids is None:
#             return None, 1.0
#
#         for i, mid in enumerate(ids.flatten()):
#             if mid != ARUCO_ID:
#                 continue
#             c = corners[i][0]
#             cx = int(c[:, 0].mean())
#             cy = int(c[:, 1].mean())
#             side = np.linalg.norm(c[0] - c[1])  # marker pixel size
#             cv2.aruco.drawDetectedMarkers(frame, corners, ids)
#             return (cx, cy), max(0.3, min(3.0, side / 100))
#
#         return None, 1.0
#
#
# # ═══════════════════════════════════════════════════════════════════════════
# #  COMPOSITOR — blend 3D render onto webcam frame
# # ═══════════════════════════════════════════════════════════════════════════
# def composite(bg_bgr: np.ndarray, car_rgba: np.ndarray,
#               anchor=None, scale=1.0) -> np.ndarray:
#     """
#     Alpha-blend the car render onto the webcam frame.
#     If anchor is given, the car is centred at that pixel position.
#     """
#     h_bg, w_bg = bg_bgr.shape[:2]
#     h_cr, w_cr = car_rgba.shape[:2]
#
#     # Resize car render based on current scale
#     new_w = int(w_cr * scale)
#     new_h = int(h_cr * scale)
#     new_w = max(10, min(new_w, w_bg))
#     new_h = max(10, min(new_h, h_bg))
#     car_scaled = cv2.resize(car_rgba, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#
#     # Determine placement
#     if anchor:
#         ax, ay = anchor
#         x0 = ax - new_w // 2
#         y0 = ay - new_h // 2
#     else:
#         x0 = (w_bg - new_w) // 2
#         y0 = (h_bg - new_h) // 2
#
#     # Clamp to frame
#     x0 = max(0, min(x0, w_bg - new_w))
#     y0 = max(0, min(y0, h_bg - new_h))
#
#     roi = bg_bgr[y0:y0+new_h, x0:x0+new_w].astype(np.float32)
#     fg  = cv2.cvtColor(car_scaled[:, :, :3], cv2.COLOR_RGB2BGR).astype(np.float32)
#     a   = (car_scaled[:, :, 3:4] / 255.0)
#
#     blended = fg * a + roi * (1 - a)
#     out = bg_bgr.copy()
#     out[y0:y0+new_h, x0:x0+new_w] = blended.astype(np.uint8)
#     return out
#
#
# # ═══════════════════════════════════════════════════════════════════════════
# #  HUD OVERLAY
# # ═══════════════════════════════════════════════════════════════════════════
# GESTURE_ICONS = {
#     "none"          : "",
#     "pinch_close"   : "🤏 Shrinking",
#     "pinch_open"    : "👐 Scaling up",
#     "interior_view" : "✌️  Interior view",
#     "reset"         : "🖐  Reset",
# }
#
# def draw_hud(frame, gesture, scale, ai_text, aruco_found, fps, voice_active):
#     h, w = frame.shape[:2]
#     overlay = frame.copy()
#
#     # top-left panel
#     cv2.rectangle(overlay, (0, 0), (320, 110), (0, 0, 0), -1)
#     cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
#
#     def put(text, y, color=(220, 255, 180), scale_f=0.62, thickness=1):
#         cv2.putText(frame, text, (10, y),
#                     cv2.FONT_HERSHEY_SIMPLEX, scale_f, color, thickness, cv2.LINE_AA)
#
#     put(f"Gesture : {GESTURE_ICONS.get(gesture, gesture)}", 26)
#     put(f"Scale   : {scale:.2f}x", 50)
#     put(f"ArUco   : {'✓ anchored' if aruco_found else '✗ free float'}", 74,
#         (100, 255, 100) if aruco_found else (80, 80, 220))
#     put(f"FPS {fps:.0f}  |  V=voice  Q=quit", 98, (160, 160, 160))
#
#     # Voice indicator
#     if voice_active:
#         cv2.circle(frame, (w - 20, 20), 10, (0, 80, 255), -1)
#         put("● REC", h - 16, (0, 100, 255), 0.55)
#
#     # AI caption — bottom bar
#     if ai_text:
#         words  = ai_text.split()
#         lines, cur = [], ""
#         for word in words:
#             test = (cur + " " + word).strip()
#             if len(test) * 9 < w - 20:
#                 cur = test
#             else:
#                 lines.append(cur)
#                 cur = word
#         if cur:
#             lines.append(cur)
#         lines = lines[:3]
#
#         bar_h = len(lines) * 28 + 16
#         bar   = frame.copy()
#         cv2.rectangle(bar, (0, h - bar_h), (w, h), (10, 10, 10), -1)
#         cv2.addWeighted(bar, 0.6, frame, 0.4, 0, frame)
#         for i, line in enumerate(lines):
#             cv2.putText(frame, line,
#                         (12, h - bar_h + 24 + i * 28),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6,
#                         (255, 240, 140), 1, cv2.LINE_AA)
#
#
# # ═══════════════════════════════════════════════════════════════════════════
# #  MAIN
# # ═══════════════════════════════════════════════════════════════════════════
# def main():
#     print("=" * 60)
#     print("  AR Car  |  Open3D + MediaPipe + Ollama")
#     print("=" * 60)
#     print("  Gesture controls:")
#     print("    Pinch      → scale car")
#     print("    2 fingers  → interior view")
#     print("    Open palm  → reset")
#     print("    Wrist L/R  → orbit")
#     print("  Keys: V = voice query | Q = quit")
#     print("  ArUco: print aruco_marker.png and hold to camera")
#     print("=" * 60)
#
#     # ── init components ──
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
#     cap.set(cv2.CAP_PROP_FPS, 30)
#     if not cap.isOpened():
#         sys.exit("[ERROR] Cannot open webcam")
#
#     gesture_engine = GestureEngine()
#     car_renderer   = CarRenderer()
#     ai             = AINarrator()
#     aruco          = ArucoTracker()
#     voice          = VoiceInput()
#
#     scale         = 1.0
#     last_gesture  = "none"
#     voice_active  = False
#     aruco_anchor  = None
#     fps_timer     = time.time()
#     fps           = 0.0
#     frame_count   = 0
#
#     # Pre-warm AI
#     ai.ask_async("Say hello in one sentence as an AR car guide.")
#
#     print("\n[READY] AR window opening…\n")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("[ERROR] Frame capture failed")
#             break
#         frame = cv2.flip(frame, 1)   # mirror
#
#         # ── gesture ──
#         gesture, scale_delta, rotate_delta = gesture_engine.process(frame)
#         scale = max(0.3, min(3.5, scale + scale_delta))
#
#         # ── aruco anchor ──
#         aruco_anchor, aruco_scale_hint = aruco.detect(frame)
#         effective_scale = scale * aruco_scale_hint if aruco_anchor else scale
#
#         # ── update 3D car ──
#         car_renderer.update(gesture, scale_delta, rotate_delta)
#         car_rgba = car_renderer.render_rgba()
#
#         # ── composite ──
#         frame = composite(frame, car_rgba, anchor=aruco_anchor,
#                           scale=effective_scale * 0.7)
#
#         # ── AI trigger on gesture change ──
#         if gesture != last_gesture:
#             prompt = AINarrator.GESTURE_PROMPTS.get(gesture)
#             if prompt:
#                 ai.ask_async(prompt)
#         last_gesture = gesture
#
#         # ── FPS ──
#         frame_count += 1
#         now = time.time()
#         if now - fps_timer >= 1.0:
#             fps        = frame_count / (now - fps_timer)
#             frame_count = 0
#             fps_timer  = now
#
#         # ── HUD ──
#         draw_hud(frame, gesture, scale, ai.get_reply(),
#                  aruco_anchor is not None, fps, voice_active)
#
#         cv2.imshow("AR Car — Hand Gesture + AI  (Q to quit)", frame)
#
#         # ── key handling ──
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('v') and VOICE_OK and not voice_active:
#             def voice_thread():
#                 nonlocal voice_active
#                 voice_active = True
#                 text = voice.listen()
#                 if text:
#                     ai.ask_async(text)
#                     # optionally speak reply back
#                     # threading.Timer(2, lambda: voice.speak(ai.get_reply())).start()
#                 voice_active = False
#             threading.Thread(target=voice_thread, daemon=True).start()
#
#     cap.release()
#     cv2.destroyAllWindows()
#     print("[EXIT] AR Car closed.")
#
#
# if __name__ == "__main__":
#     main()





"""
AR Car — Windows Compatible Edition
=====================================
Fixes: Open3D EGL headless error on Windows.
Renderer replaced with PyOpenGL + pygame (works on Windows/Mac/Linux,
any Python version including 3.13).

Everything else is identical:
  • MediaPipe hand gesture control
  • ArUco marker anchor
  • Voice input (V key) → Ollama LLM narration
  • AI captions on live webcam feed

Install
-------
  pip install opencv-python mediapipe pygame PyOpenGL PyOpenGL_accelerate
              ollama numpy speechrecognition pyttsx3

  For pyaudio on Windows:
      pip install pipwin && pipwin install pyaudio

Ollama:
  ollama serve        ← keep open in another terminal
  ollama pull llama3  ← once

3D model:
  Place car.obj next to this file (any free .obj from Sketchfab/PolyHaven).
  If missing, a red box placeholder is used automatically.

Controls
--------
  Pinch close       → shrink car
  Pinch open        → enlarge car
  2 fingers up      → interior view
  Open palm         → reset
  Wrist left/right  → orbit
  V                 → voice query
  Q                 → quit
"""

import cv2
import numpy as np
import mediapipe as mp
import ollama
import threading
import math
import os
import time
import sys
import ctypes

# ── PyOpenGL + pygame offscreen ──────────────────────────────────────────
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, NOFRAME
from OpenGL.GL import *
from OpenGL.GLU import *

# ── optional voice ───────────────────────────────────────────────────────
try:
    import speech_recognition as sr
    import pyttsx3
    VOICE_OK = True
except ImportError:
    VOICE_OK = False
    print("[WARN] speech_recognition / pyttsx3 not installed — voice disabled")

# ═══════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════
CAM_W, CAM_H  = 1280, 720
RENDER_W      = 640
RENDER_H      = 480
MODEL_PATH    = "../car.obj"
OLLAMA_MODEL  = "llava-phi3"
ARUCO_DICT    = cv2.aruco.DICT_6X6_250
ARUCO_ID      = 0

# ═══════════════════════════════════════════════════════════════════════
#  OBJ LOADER  (no extra deps — pure Python)
# ═══════════════════════════════════════════════════════════════════════
def load_obj(path):
    """
    Minimal .obj loader. Returns list of triangles:
      [ ((x,y,z),(x,y,z),(x,y,z)), ... ]
    Handles triangles and quads (splits quads into 2 triangles).
    """
    verts, tris = [], []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "v":
                    verts.append((float(parts[1]),
                                  float(parts[2]),
                                  float(parts[3])))
                elif parts[0] == "f":
                    # face indices (1-based, may have v/vt/vn format)
                    idxs = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                    # fan triangulation for polygons
                    for i in range(1, len(idxs) - 1):
                        tris.append((verts[idxs[0]],
                                     verts[idxs[i]],
                                     verts[idxs[i + 1]]))
        print(f"[OBJ] Loaded {path}: {len(verts)} verts, {len(tris)} tris")
    except Exception as e:
        print(f"[OBJ] Load error: {e}")
    return tris


def compute_normals(tris):
    """Return per-triangle flat normals as list of (nx,ny,nz)."""
    normals = []
    for (a, b, c) in tris:
        ab = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
        ac = (c[0]-a[0], c[1]-a[1], c[2]-a[2])
        nx = ab[1]*ac[2] - ab[2]*ac[1]
        ny = ab[2]*ac[0] - ab[0]*ac[2]
        nz = ab[0]*ac[1] - ab[1]*ac[0]
        L  = math.sqrt(nx*nx + ny*ny + nz*nz) or 1e-9
        normals.append((nx/L, ny/L, nz/L))
    return normals


def normalise_mesh(tris):
    """Centre and scale mesh to fit in a unit cube."""
    if not tris:
        return tris
    all_v = [v for tri in tris for v in tri]
    xs = [v[0] for v in all_v]
    ys = [v[1] for v in all_v]
    zs = [v[2] for v in all_v]
    cx = (max(xs)+min(xs))/2
    cy = (max(ys)+min(ys))/2
    cz = (max(zs)+min(zs))/2
    scale = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)) or 1
    out = []
    for (a, b, c) in tris:
        def s(v): return ((v[0]-cx)/scale, (v[1]-cy)/scale, (v[2]-cz)/scale)
        out.append((s(a), s(b), s(c)))
    return out


def make_placeholder_tris():
    """Box + roof + 4 sphere-approximated wheels as triangle soup."""
    tris = []

    def box(x0,y0,z0, x1,y1,z1):
        faces = [
            [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],
            [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
            [(x0,y0,z0),(x0,y0,z1),(x0,y1,z1),(x0,y1,z0)],
            [(x1,y0,z0),(x1,y0,z1),(x1,y1,z1),(x1,y1,z0)],
            [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],
            [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],
        ]
        for f in faces:
            tris.append((f[0],f[1],f[2]))
            tris.append((f[0],f[2],f[3]))

    def sphere(cx,cy,cz, r, stacks=6, slices=8):
        for i in range(stacks):
            lat0 = math.pi*(-0.5 + i/stacks)
            lat1 = math.pi*(-0.5 + (i+1)/stacks)
            for j in range(slices):
                lng0 = 2*math.pi*j/slices
                lng1 = 2*math.pi*(j+1)/slices
                def pt(lat,lng):
                    return (cx+r*math.cos(lat)*math.cos(lng),
                            cy+r*math.cos(lat)*math.sin(lng),
                            cz+r*math.sin(lat))
                a,b,c_,d = pt(lat0,lng0),pt(lat0,lng1),pt(lat1,lng1),pt(lat1,lng0)
                tris.append((a,b,c_)); tris.append((a,c_,d))

    box(-1.0,-0.5,-0.05, 1.0, 0.5, 0.55)   # body
    box(-0.6,-0.48, 0.55, 0.6, 0.48, 1.0)  # roof
    for wx,wy in [(-0.7,-0.55),(0.7,-0.55),(-0.7,0.55),(0.7,0.55)]:
        sphere(wx, wy, -0.05, 0.22)
    return tris


# ═══════════════════════════════════════════════════════════════════════
#  OPENGL CAR RENDERER  (pygame hidden window → FBO → numpy)
# ═══════════════════════════════════════════════════════════════════════
class CarRenderer:
    """
    Uses pygame to create a hidden OpenGL context, renders into an FBO,
    and reads pixels back as a NumPy RGBA array.
    Works on Windows, macOS, Linux — no EGL needed.
    """

    def __init__(self, width=RENDER_W, height=RENDER_H, model_path=MODEL_PATH):
        self.w = width
        self.h = height
        self.yaw   = 0.0
        self.pitch = 20.0        # slight downward look
        self.scale = 1.0
        self.interior = False

        # ── init pygame / OpenGL hidden window ──────────────────────────
        pygame.init()
        # NOFRAME + tiny size keeps it off-screen / non-intrusive
        os.environ.setdefault("SDL_VIDEODRIVER", "windib")   # Windows
        self._surface = pygame.display.set_mode(
            (width, height),
            DOUBLEBUF | OPENGL | NOFRAME,
        )
        pygame.display.set_caption("AR-Car GL (hidden)")

        # ── FBO for offscreen rendering ──────────────────────────────────
        self._fbo      = glGenFramebuffers(1)
        self._color_rb = glGenRenderbuffers(1)
        self._depth_rb = glGenRenderbuffers(1)

        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)

        glBindRenderbuffer(GL_RENDERBUFFER, self._color_rb)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                  GL_RENDERBUFFER, self._color_rb)

        glBindRenderbuffer(GL_RENDERBUFFER, self._depth_rb)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                  GL_RENDERBUFFER, self._depth_rb)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"[GL] FBO incomplete: {status}")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # ── OpenGL state ─────────────────────────────────────────────────
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)

        # Lights
        glLightfv(GL_LIGHT0, GL_POSITION,  [2.0,  3.0, 4.0, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,   [1.0,  0.95,0.9, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR,  [0.6,  0.6, 0.6, 1.0])
        glLightfv(GL_LIGHT1, GL_POSITION,  [-2.0,-1.0, 2.0, 1.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE,   [0.3,  0.3, 0.35,1.0])
        glLightfv(GL_LIGHT1, GL_SPECULAR,  [0.0,  0.0, 0.0, 1.0])
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.25, 0.25, 0.28, 1.0])

        # ── load mesh ────────────────────────────────────────────────────
        if os.path.exists(model_path):
            raw_tris = load_obj(model_path)
        else:
            print(f"[3D] {model_path} not found — using placeholder")
            raw_tris = make_placeholder_tris()

        self._tris    = normalise_mesh(raw_tris)
        self._normals = compute_normals(self._tris)
        self._dl      = self._build_display_list()

        # ── projection ───────────────────────────────────────────────────
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(55, width / height, 0.01, 100.0)
        glMatrixMode(GL_MODELVIEW)

    # ── build OpenGL display list (draw once, call many) ─────────────────
    def _build_display_list(self):
        dl = glGenLists(1)
        glNewList(dl, GL_COMPILE)
        glColor3f(0.85, 0.12, 0.12)   # red car body
        glBegin(GL_TRIANGLES)
        for (a, b, c), n in zip(self._tris, self._normals):
            glNormal3f(*n)
            glVertex3f(*a)
            glVertex3f(*b)
            glVertex3f(*c)
        glEnd()
        glEndList()
        return dl

    # ── update transform ──────────────────────────────────────────────────
    def update(self, gesture, scale_delta, rotate_delta):
        self.scale = max(0.3, min(3.5, self.scale + scale_delta))
        self.yaw  += rotate_delta

        if gesture == "interior_view" and not self.interior:
            self.interior = True
        elif gesture == "reset":
            self.interior = False
            self.scale    = 1.0
            self.yaw      = 0.0

    # ── render to RGBA numpy ──────────────────────────────────────────────
    def render_rgba(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glViewport(0, 0, self.w, self.h)

        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()

        if self.interior:
            # camera inside cabin looking forward
            gluLookAt(0.1, -0.2, 0.3,
                      0.5,  0.8, 0.3,
                      0.0,  0.0, 1.0)
        else:
            # exterior orbit camera
            eye_dist = 3.5 / max(self.scale, 0.3)
            rad = math.radians(self.yaw)
            ex  = eye_dist * math.sin(rad)
            ey  = -eye_dist * math.cos(rad)
            ez  = eye_dist * 0.4
            gluLookAt(ex, ey, ez,  0, 0, 0,  0, 0, 1)

        glScalef(self.scale, self.scale, self.scale)
        glCallList(self._dl)

        # read pixels
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        raw = glReadPixels(0, 0, self.w, self.h, GL_RGBA, GL_UNSIGNED_BYTE)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(self.h, self.w, 4)
        arr = arr[::-1]   # flip vertically (OpenGL is bottom-up)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # pump pygame event queue so the hidden window doesn't freeze
        pygame.event.pump()

        return arr   # RGBA uint8

    def cleanup(self):
        glDeleteLists(self._dl, 1)
        glDeleteFramebuffers(1, [self._fbo])
        glDeleteRenderbuffers(1, [self._color_rb])
        glDeleteRenderbuffers(1, [self._depth_rb])
        pygame.quit()


# ═══════════════════════════════════════════════════════════════════════
#  GESTURE ENGINE
# ═══════════════════════════════════════════════════════════════════════
class GestureEngine:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
        )
        self.draw = mp.solutions.drawing_utils
        self.prev_pinch = None

    @staticmethod
    def _dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    @staticmethod
    def _fingers_up(lm):
        return sum(lm[t].y < lm[j].y
                   for t, j in zip([8,12,16,20], [6,10,14,18]))

    def process(self, frame):
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        gesture, sd, rd = "none", 0.0, 0.0

        if not result.multi_hand_landmarks:
            self.prev_pinch = None
            return gesture, sd, rd

        for hlm in result.multi_hand_landmarks:
            lm = hlm.landmark
            self.draw.draw_landmarks(
                frame, hlm, self.mp_hands.HAND_CONNECTIONS,
                self.draw.DrawingSpec(color=(0,200,150), thickness=2),
                self.draw.DrawingSpec(color=(0,255,100), thickness=1),
            )
            pinch   = self._dist(lm[4], lm[8])
            fingers = self._fingers_up(lm)

            if self.prev_pinch is not None:
                delta = pinch - self.prev_pinch
                if abs(delta) > 0.005:
                    sd = delta * 1.8
            self.prev_pinch = pinch

            if fingers >= 4:    gesture = "reset"
            elif fingers == 2:  gesture = "interior_view"
            elif pinch < 0.06:  gesture = "pinch_close"
            elif pinch > 0.20:  gesture = "pinch_open"

            wx = lm[0].x
            if wx < 0.30:   rd = -4.0
            elif wx > 0.70: rd = +4.0

        return gesture, sd, rd


# ═══════════════════════════════════════════════════════════════════════
#  ARUCO TRACKER
# ═══════════════════════════════════════════════════════════════════════
class ArucoTracker:
    def __init__(self):
        self.adict    = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.detector = cv2.aruco.ArucoDetector(
            self.adict, cv2.aruco.DetectorParameters())
        if not os.path.exists("../aruco_marker.png"):
            img = cv2.aruco.generateImageMarker(self.adict, ARUCO_ID, 300)
            cv2.imwrite("../aruco_marker.png", img)
            print("[ArUco] Marker saved → aruco_marker.png  (print or display it)")

    def detect(self, frame):
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(grey)
        if ids is None:
            return None, 1.0
        for i, mid in enumerate(ids.flatten()):
            if mid != ARUCO_ID:
                continue
            c    = corners[i][0]
            cx   = int(c[:, 0].mean())
            cy   = int(c[:, 1].mean())
            side = np.linalg.norm(c[0] - c[1])
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            return (cx, cy), max(0.3, min(3.0, side / 100))
        return None, 1.0


# ═══════════════════════════════════════════════════════════════════════
#  COMPOSITOR
# ═══════════════════════════════════════════════════════════════════════
def composite(bg_bgr, car_rgba, anchor=None, scale=1.0):
    hb, wb = bg_bgr.shape[:2]
    hc, wc = car_rgba.shape[:2]
    nw = max(10, min(int(wc * scale), wb))
    nh = max(10, min(int(hc * scale), hb))
    scaled = cv2.resize(car_rgba, (nw, nh), interpolation=cv2.INTER_LINEAR)

    x0 = (anchor[0] - nw//2) if anchor else (wb - nw)//2
    y0 = (anchor[1] - nh//2) if anchor else (hb - nh)//2
    x0 = max(0, min(x0, wb - nw))
    y0 = max(0, min(y0, hb - nh))

    roi = bg_bgr[y0:y0+nh, x0:x0+nw].astype(np.float32)
    # car_rgba is RGBA from OpenGL (RGB already in BGR-equivalent after flip)
    fg  = scaled[:, :, :3].astype(np.float32)        # RGB
    fg  = fg[:, :, ::-1]                              # → BGR for OpenCV
    a   = scaled[:, :, 3:4].astype(np.float32) / 255.0

    out = bg_bgr.copy()
    out[y0:y0+nh, x0:x0+nw] = (fg * a + roi * (1 - a)).astype(np.uint8)
    return out


# ═══════════════════════════════════════════════════════════════════════
#  OLLAMA AI NARRATOR
# ═══════════════════════════════════════════════════════════════════════
class AINarrator:
    SYSTEM = (
        "You are a knowledgeable AR car guide. "
        "Answer in exactly 1-2 short sentences. "
        "Be specific, vivid, and enthusiastic. "
        "No bullet points. No markdown."
    )
    GESTURE_PROMPTS = {
        "interior_view" : "Describe what I see looking inside a modern sports car cabin.",
        "pinch_open"    : "What aerodynamic changes matter most at large car scale?",
        "reset"         : "Give me one surprising fact about automotive design history.",
    }

    def __init__(self, model=OLLAMA_MODEL):
        self.model = model
        self.reply = "Show hand gestures to control the car.  Press V for voice."
        self.busy  = False
        self._lock = threading.Lock()

    def ask_async(self, prompt):
        if self.busy:
            return
        self.busy = True
        threading.Thread(target=self._query, args=(prompt,), daemon=True).start()

    def _query(self, prompt):
        try:
            res = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
            )
            with self._lock:
                self.reply = res["message"]["content"].strip()
        except Exception as e:
            with self._lock:
                self.reply = f"[AI offline — run: ollama serve]  ({e})"
        finally:
            self.busy = False

    def get_reply(self):
        with self._lock:
            return self.reply


# ═══════════════════════════════════════════════════════════════════════
#  VOICE INPUT
# ═══════════════════════════════════════════════════════════════════════
class VoiceInput:
    def __init__(self):
        if not VOICE_OK:
            return
        self.rec = sr.Recognizer()
        self.mic = sr.Microphone()
        self.tts = pyttsx3.init()
        self.tts.setProperty("rate", 165)
        with self.mic as src:
            self.rec.adjust_for_ambient_noise(src, duration=0.5)

    def listen(self):
        if not VOICE_OK:
            return None
        try:
            with self.mic as src:
                print("[Voice] Listening…")
                audio = self.rec.listen(src, timeout=5, phrase_time_limit=8)
            text = self.rec.recognize_google(audio)
            print(f"[Voice] Heard: {text}")
            return text
        except Exception as e:
            print(f"[Voice] {e}")
            return None

    def speak(self, text):
        if not VOICE_OK:
            return
        threading.Thread(
            target=lambda: (self.tts.say(text), self.tts.runAndWait()),
            daemon=True,
        ).start()


# ═══════════════════════════════════════════════════════════════════════
#  HUD
# ═══════════════════════════════════════════════════════════════════════
GESTURE_LABELS = {
    "none":          "",
    "pinch_close":   "Shrinking",
    "pinch_open":    "Scaling up",
    "interior_view": "Interior view",
    "reset":         "Reset",
}

def draw_hud(frame, gesture, scale, ai_text, aruco_found, fps, voice_active):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (330, 115), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    def put(text, y, color=(210, 255, 170), sc=0.62, th=1):
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, color, th, cv2.LINE_AA)

    put(f"Gesture : {GESTURE_LABELS.get(gesture, gesture)}", 26)
    put(f"Scale   : {scale:.2f}x", 50)
    put(f"ArUco   : {'ANCHORED' if aruco_found else 'free float'}", 74,
        (100, 255, 100) if aruco_found else (80, 80, 220))
    put(f"FPS {fps:.0f}   V=voice   Q=quit", 98, (160, 160, 160))

    if voice_active:
        cv2.circle(frame, (w - 22, 22), 11, (0, 60, 255), -1)

    if ai_text:
        words = ai_text.split()
        lines, cur = [], ""
        for word in words:
            test = (cur + " " + word).strip()
            if len(test) * 9 < w - 22:
                cur = test
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        lines = lines[:3]
        bar_h = len(lines) * 28 + 16
        bar = frame.copy()
        cv2.rectangle(bar, (0, h - bar_h), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(bar, 0.6, frame, 0.4, 0, frame)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (12, h - bar_h + 24 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 240, 140), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 58)
    print("  AR Car  |  PyOpenGL + MediaPipe + Ollama  (Windows OK)")
    print("=" * 58)
    print("  Gestures: pinch=scale  2-fingers=interior  palm=reset")
    print("  Keys    : V=voice  Q=quit")
    print("  ArUco   : print aruco_marker.png and hold to camera")
    print("=" * 58)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        sys.exit("[ERROR] Cannot open webcam — check camera index")

    print("[INIT] Loading renderer…")
    car   = CarRenderer()
    print("[INIT] Renderer ready")

    ge    = GestureEngine()
    ai    = AINarrator()
    aruco = ArucoTracker()
    voice = VoiceInput()

    scale        = 1.0
    last_gesture = "none"
    voice_active = False
    fps_timer    = time.time()
    fps          = 0.0
    fc           = 0

    ai.ask_async("Say hello in one sentence as an AR car guide.")

    print("[READY] AR window open\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame capture failed"); break
        frame = cv2.flip(frame, 1)

        # gestures
        gesture, sd, rd = ge.process(frame)
        scale = max(0.3, min(3.5, scale + sd))

        # aruco
        anchor, aruco_scale = aruco.detect(frame)
        eff_scale = scale * aruco_scale if anchor else scale

        # 3D render + composite
        car.update(gesture, sd, rd)
        car_rgba = car.render_rgba()
        frame    = composite(frame, car_rgba, anchor=anchor,
                             scale=eff_scale * 0.7)

        # AI on gesture change
        if gesture != last_gesture:
            p = AINarrator.GESTURE_PROMPTS.get(gesture)
            if p:
                ai.ask_async(p)
        last_gesture = gesture

        # FPS
        fc  += 1
        now  = time.time()
        if now - fps_timer >= 1.0:
            fps = fc / (now - fps_timer)
            fc  = 0
            fps_timer = now

        draw_hud(frame, gesture, scale, ai.get_reply(),
                 anchor is not None, fps, voice_active)

        cv2.imshow("AR Car — Gesture + AI  (Q quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('v') and VOICE_OK and not voice_active:
            def _vt():
                nonlocal voice_active
                voice_active = True
                text = voice.listen()
                if text:
                    ai.ask_async(text)
                voice_active = False
            threading.Thread(target=_vt, daemon=True).start()

    car.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    print("[EXIT] Done.")


if __name__ == "__main__":
    main()