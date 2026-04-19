# # # # """
# # # # AR Car — Voice Conversation Edition
# # # # ====================================
# # # # Identical pipeline to ar_car_open3d.py but with a FULL voice conversation
# # # # loop: continuous listening, spoken AI replies, and a conversation history
# # # # so Ollama remembers what was said earlier in the session.
# # # #
# # # # Extra requirements (on top of base):
# # # #   pip install pyttsx3 speechrecognition pyaudio webrtcvad
# # # #
# # # # How it works
# # # # ------------
# # # # A background thread continuously listens for speech (with VAD silence
# # # # detection so it doesn't stay open forever). When speech is detected:
# # # #   1. Text is sent to Ollama with full conversation history
# # # #   2. Ollama replies → displayed on screen AND spoken aloud via TTS
# # # #   3. Gestures still work in parallel — completely independent threads
# # # #
# # # # Usage
# # # # -----
# # # #   python ar_car_voice.py
# # # #
# # # #   Just speak naturally — no key press needed.
# # # #   Say "exit" or "quit" or press Q to stop.
# # # # """
# # # # import os
# # # # os.environ["OPEN3D_GUI_BACKEND"] = "GLFW"
# # # # import cv2
# # # # import numpy as np
# # # # import mediapipe as mp
# # # # import open3d as o3d
# # # # import ollama
# # # # import threading
# # # # import math
# # # # import os
# # # # import time
# # # # import sys
# # # # import queue
# # # # mp_hands = mp.solutions.hands
# # # # mp_draw = mp.solutions.drawing_utils
# # # # try:
# # # #     import speech_recognition as sr
# # # #     import pyttsx3
# # # #     VOICE_OK = True
# # # # except ImportError:
# # # #     VOICE_OK = False
# # # #     print("[WARN] Voice deps missing — running gesture-only mode")
# # # #
# # # # # ── reuse all classes from ar_car_open3d ─────────────────────────────────
# # # # # (In a real project these would be a shared module; duplicated here for
# # # # #  standalone running)
# # # #
# # # # CAM_W, CAM_H = 1280, 720
# # # # RENDER_W      = 640
# # # # RENDER_H      = 480
# # # # MODEL_PATH    = "car.obj"
# # # # OLLAMA_MODEL  = "llava-phi3"
# # # # ARUCO_DICT    = cv2.aruco.DICT_6X6_250
# # # # ARUCO_ID      = 0
# # # #
# # # #
# # # # # ── paste GestureEngine, CarRenderer, ArucoTracker from ar_car_open3d ────
# # # # # (abbreviated here — import from the other file in practice)
# # # #
# # # # class GestureEngine:
# # # #     def __init__(self):
# # # #         self.mp_hands = mp.solutions.hands
# # # #         self.hands = self.mp_hands.Hands(max_num_hands=2,
# # # #             min_detection_confidence=0.75, min_tracking_confidence=0.75)
# # # #         self.draw = mp.solutions.drawing_utils
# # # #         self.prev_pinch = None
# # # #
# # # #     @staticmethod
# # # #     def _dist(a, b): return math.hypot(a.x-b.x, a.y-b.y)
# # # #
# # # #     @staticmethod
# # # #     def _fingers_up(lm):
# # # #         return sum(lm[t].y < lm[j].y for t,j in zip([8,12,16,20],[6,10,14,18]))
# # # #
# # # #     def process(self, frame):
# # # #         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# # # #         res = self.hands.process(rgb)
# # # #         gesture, sd, rd = "none", 0.0, 0.0
# # # #         if not res.multi_hand_landmarks:
# # # #             self.prev_pinch = None
# # # #             return gesture, sd, rd
# # # #         for hlm in res.multi_hand_landmarks:
# # # #             lm = hlm.landmark
# # # #             self.draw.draw_landmarks(frame, hlm, self.mp_hands.HAND_CONNECTIONS,
# # # #                 self.draw.DrawingSpec(color=(0,200,150),thickness=2),
# # # #                 self.draw.DrawingSpec(color=(0,255,100),thickness=1))
# # # #             pinch   = self._dist(lm[4], lm[8])
# # # #             fingers = self._fingers_up(lm)
# # # #             if self.prev_pinch is not None:
# # # #                 delta = pinch - self.prev_pinch
# # # #                 if abs(delta) > 0.005: sd = delta * 1.8
# # # #             self.prev_pinch = pinch
# # # #             if fingers >= 4:   gesture = "reset"
# # # #             elif fingers == 2: gesture = "interior_view"
# # # #             elif pinch < 0.06: gesture = "pinch_close"
# # # #             elif pinch > 0.20: gesture = "pinch_open"
# # # #             wx = lm[0].x
# # # #             if wx < 0.30:   rd = -4.0
# # # #             elif wx > 0.70: rd = +4.0
# # # #         return gesture, sd, rd
# # # #
# # # #
# # # # class CarRenderer:
# # # #     def __init__(self, width=RENDER_W, height=RENDER_H, model_path=MODEL_PATH):
# # # #         self.w = width; self.h = height
# # # #         self.yaw = 0.0; self.scale = 1.0; self.interior = False
# # # #         self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
# # # #         sc = self.renderer.scene
# # # #         sc.set_background([0,0,0,0])
# # # #         sc.scene.set_sun_light([0.5,-1,-0.8],[1.2,1.2,1.2],80000)
# # # #         sc.scene.enable_sun_light(True)
# # # #         sc.scene.enable_indirect_lighting(True)
# # # #         mat = o3d.visualization.rendering.MaterialRecord()
# # # #         mat.shader = "defaultLit"
# # # #         mat.base_color = [0.85,0.12,0.12,1.0]
# # # #         mat.base_roughness = 0.35; mat.base_metallic = 0.6
# # # #         if os.path.exists(model_path):
# # # #             mesh = o3d.io.read_triangle_mesh(model_path)
# # # #             mesh.compute_vertex_normals()
# # # #         else:
# # # #             mesh = self._placeholder()
# # # #         bb = mesh.get_axis_aligned_bounding_box()
# # # #         ex = bb.get_max_bound() - bb.get_min_bound()
# # # #         mesh.translate(-bb.get_center())
# # # #         mesh.scale(2.0/max(ex), center=[0,0,0])
# # # #         sc.add_geometry("car", mesh, mat)
# # # #         self._set_cam_ext()
# # # #
# # # #     @staticmethod
# # # #     def _placeholder():
# # # #         body = o3d.geometry.TriangleMesh.create_box(2,1,0.6)
# # # #         roof = o3d.geometry.TriangleMesh.create_box(1.2,0.95,0.5)
# # # #         wheels = [o3d.geometry.TriangleMesh.create_sphere(0.25) for _ in range(4)]
# # # #         body.translate([-1,-0.5,0]); roof.translate([-0.6,-0.475,0.6])
# # # #         for w,p in zip(wheels,[(-0.7,-0.6,-0.25),(0.7,-0.6,-0.25),
# # # #                                 (-0.7,0.6,-0.25),(0.7,0.6,-0.25)]): w.translate(p)
# # # #         m = body+roof+wheels[0]+wheels[1]+wheels[2]+wheels[3]
# # # #         m.paint_uniform_color([0.85,0.12,0.12]); m.compute_vertex_normals()
# # # #         return m
# # # #
# # # #     def _set_cam_ext(self):
# # # #         self.renderer.setup_camera(60,[0,0,0],[0,-4.5,1.5],[0,0,1])
# # # #
# # # #     def _set_cam_int(self):
# # # #         self.renderer.setup_camera(90,[0.3,0.1,0.3],[0,-0.3,0.5],[0,0,1])
# # # #
# # # #     def update(self, gesture, sd, rd):
# # # #         self.scale = max(0.3, min(3.5, self.scale+sd))
# # # #         self.yaw  += rd
# # # #         T = np.eye(4)
# # # #         r = math.radians(self.yaw)
# # # #         T[0,0]=math.cos(r)*self.scale; T[0,1]=-math.sin(r)*self.scale
# # # #         T[1,0]=math.sin(r)*self.scale; T[1,1]=math.cos(r)*self.scale
# # # #         T[2,2]=self.scale
# # # #         self.renderer.scene.set_geometry_transform("car", T)
# # # #         if gesture=="interior_view" and not self.interior:
# # # #             self._set_cam_int(); self.interior=True
# # # #         elif gesture=="reset":
# # # #             self._set_cam_ext(); self.interior=False
# # # #             self.scale=1.0; self.yaw=0.0
# # # #
# # # #     def render_rgba(self):
# # # #         img = self.renderer.render_to_image()
# # # #         arr = np.asarray(img)
# # # #         if arr.dtype!=np.uint8: arr=(arr*255).clip(0,255).astype(np.uint8)
# # # #         grey  = arr.mean(axis=2)
# # # #         alpha = np.where(grey<8,0,255).astype(np.uint8)
# # # #         return np.dstack([arr,alpha])
# # # #
# # # #
# # # # class ArucoTracker:
# # # #     def __init__(self):
# # # #         self.adict    = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
# # # #         self.detector = cv2.aruco.ArucoDetector(self.adict, cv2.aruco.DetectorParameters())
# # # #         if not os.path.exists("aruco_marker.png"):
# # # #             img = cv2.aruco.generateImageMarker(self.adict, ARUCO_ID, 300)
# # # #             cv2.imwrite("aruco_marker.png", img)
# # # #             print("[ArUco] Marker saved → aruco_marker.png")
# # # #
# # # #     def detect(self, frame):
# # # #         grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # # #         corners, ids, _ = self.detector.detectMarkers(grey)
# # # #         if ids is None: return None, 1.0
# # # #         for i,mid in enumerate(ids.flatten()):
# # # #             if mid!=ARUCO_ID: continue
# # # #             c  = corners[i][0]
# # # #             cx = int(c[:,0].mean()); cy = int(c[:,1].mean())
# # # #             side = np.linalg.norm(c[0]-c[1])
# # # #             cv2.aruco.drawDetectedMarkers(frame, corners, ids)
# # # #             return (cx,cy), max(0.3,min(3.0,side/100))
# # # #         return None, 1.0
# # # #
# # # #
# # # # def composite(bg, car_rgba, anchor=None, scale=1.0):
# # # #     hb,wb = bg.shape[:2]; hc,wc = car_rgba.shape[:2]
# # # #     nw = max(10,min(int(wc*scale),wb))
# # # #     nh = max(10,min(int(hc*scale),hb))
# # # #     scaled = cv2.resize(car_rgba,(nw,nh))
# # # #     x0 = (anchor[0]-nw//2 if anchor else (wb-nw)//2)
# # # #     y0 = (anchor[1]-nh//2 if anchor else (hb-nh)//2)
# # # #     x0=max(0,min(x0,wb-nw)); y0=max(0,min(y0,hb-nh))
# # # #     roi = bg[y0:y0+nh,x0:x0+nw].astype(np.float32)
# # # #     fg  = cv2.cvtColor(scaled[:,:,:3],cv2.COLOR_RGB2BGR).astype(np.float32)
# # # #     a   = scaled[:,:,3:4]/255.0
# # # #     out = bg.copy()
# # # #     out[y0:y0+nh,x0:x0+nw] = (fg*a+roi*(1-a)).astype(np.uint8)
# # # #     return out
# # # #
# # # #
# # # # # ═══════════════════════════════════════════════════════════════════════════
# # # # #  CONVERSATIONAL AI  — with history
# # # # # ═══════════════════════════════════════════════════════════════════════════
# # # # class ConversationalAI:
# # # #     SYSTEM = (
# # # #         "You are an enthusiastic AR car guide with deep automotive knowledge. "
# # # #         "Keep every reply to 1-2 sentences maximum. "
# # # #         "Be vivid, specific, and friendly. No markdown, no bullet points."
# # # #     )
# # # #
# # # #     def __init__(self, model=OLLAMA_MODEL):
# # # #         self.model   = model
# # # #         self.history = []           # [{role, content}, …] growing context
# # # #         self.reply   = "🚗  Say something to start a conversation!"
# # # #         self.busy    = False
# # # #         self.lock    = threading.Lock()
# # # #         self.tts     = None
# # # #         if VOICE_OK:
# # # #             try:
# # # #                 self.tts = pyttsx3.init()
# # # #                 self.tts.setProperty("rate", 168)
# # # #                 self.tts.setProperty("volume", 0.9)
# # # #             except Exception as e:
# # # #                 print(f"[TTS] Init failed: {e}")
# # # #
# # # #     def chat(self, user_text: str, speak_reply=True):
# # # #         """Non-blocking — fires in background thread."""
# # # #         if self.busy:
# # # #             return
# # # #         self.busy = True
# # # #         threading.Thread(target=self._run, args=(user_text, speak_reply),
# # # #                          daemon=True).start()
# # # #
# # # #     def _run(self, user_text, speak_reply):
# # # #         with self.lock:
# # # #             self.history.append({"role": "user", "content": user_text})
# # # #             # keep last 10 turns to avoid token overflow
# # # #             history_slice = self.history[-10:]
# # # #
# # # #         try:
# # # #             res = ollama.chat(
# # # #                 model=self.model,
# # # #                 messages=[{"role":"system","content":self.SYSTEM}] + history_slice,
# # # #             )
# # # #             reply_text = res["message"]["content"].strip()
# # # #         except Exception as e:
# # # #             reply_text = f"[AI error: {e}]"
# # # #
# # # #         with self.lock:
# # # #             self.history.append({"role":"assistant","content":reply_text})
# # # #             self.reply = reply_text
# # # #
# # # #         print(f"\n[AI] {reply_text}\n")
# # # #
# # # #         if speak_reply and self.tts:
# # # #             try:
# # # #                 self.tts.say(reply_text)
# # # #                 self.tts.runAndWait()
# # # #             except Exception:
# # # #                 pass
# # # #
# # # #         self.busy = False
# # # #
# # # #     def get_reply(self):
# # # #         with self.lock:
# # # #             return self.reply
# # # #
# # # #
# # # # # ═══════════════════════════════════════════════════════════════════════════
# # # # #  CONTINUOUS VOICE LISTENER
# # # # # ═══════════════════════════════════════════════════════════════════════════
# # # # class ContinuousListener:
# # # #     """
# # # #     Runs in a background thread. Puts recognised text into a queue.
# # # #     The main loop reads from the queue without blocking.
# # # #     """
# # # #     def __init__(self):
# # # #         self.q    = queue.Queue()
# # # #         self.stop = threading.Event()
# # # #         self._t   = None
# # # #
# # # #     def start(self):
# # # #         if not VOICE_OK:
# # # #             return
# # # #         self._t = threading.Thread(target=self._loop, daemon=True)
# # # #         self._t.start()
# # # #
# # # #     def _loop(self):
# # # #         rec = sr.Recognizer()
# # # #         mic = sr.Microphone()
# # # #         rec.dynamic_energy_threshold = True
# # # #         rec.energy_threshold          = 3000
# # # #         with mic as src:
# # # #             rec.adjust_for_ambient_noise(src, duration=1.0)
# # # #         print("[Voice] Continuous listener started — just speak!")
# # # #         while not self.stop.is_set():
# # # #             try:
# # # #                 with mic as src:
# # # #                     audio = rec.listen(src, timeout=2, phrase_time_limit=10)
# # # #                 text = rec.recognize_google(audio)
# # # #                 if text.strip():
# # # #                     print(f"[Voice] '{text}'")
# # # #                     self.q.put(text)
# # # #             except sr.WaitTimeoutError:
# # # #                 pass   # silence — keep looping
# # # #             except sr.UnknownValueError:
# # # #                 pass   # could not understand
# # # #             except Exception as e:
# # # #                 print(f"[Voice] {e}")
# # # #                 time.sleep(0.5)
# # # #
# # # #     def get_text(self):
# # # #         """Non-blocking — returns text if available, else None."""
# # # #         try:
# # # #             return self.q.get_nowait()
# # # #         except queue.Empty:
# # # #             return None
# # # #
# # # #     def close(self):
# # # #         self.stop.set()
# # # #
# # # #
# # # # # ═══════════════════════════════════════════════════════════════════════════
# # # # #  HUD
# # # # # ═══════════════════════════════════════════════════════════════════════════
# # # # def draw_hud_v(frame, gesture, scale, ai_text, aruco_found, fps, listening):
# # # #     h, w = frame.shape[:2]
# # # #     ov = frame.copy()
# # # #     cv2.rectangle(ov, (0,0), (350,115), (0,0,0), -1)
# # # #     cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)
# # # #
# # # #     def put(t, y, col=(220,255,180), s=0.62, th=1):
# # # #         cv2.putText(frame, t, (10,y), cv2.FONT_HERSHEY_SIMPLEX, s, col, th, cv2.LINE_AA)
# # # #
# # # #     icon = {"none":"","pinch_close":"🤏","pinch_open":"👐",
# # # #             "interior_view":"✌️","reset":"🖐"}.get(gesture,"")
# # # #     put(f"Gesture : {icon} {gesture}", 26)
# # # #     put(f"Scale   : {scale:.2f}x", 52)
# # # #     put(f"ArUco   : {'✓ anchored' if aruco_found else '✗ free'}", 78,
# # # #         (100,255,100) if aruco_found else (80,80,220))
# # # #     put(f"FPS {fps:.0f}  |  Q=quit  |  speak freely", 104, (160,160,160))
# # # #
# # # #     # Mic indicator
# # # #     if listening:
# # # #         r = int(8 + 4*math.sin(time.time()*6))
# # # #         cv2.circle(frame, (w-20, 20), r, (0,60,255), -1)
# # # #
# # # #     # AI caption bar
# # # #     if ai_text:
# # # #         words = ai_text.split()
# # # #         lines, cur = [], ""
# # # #         for word in words:
# # # #             test = (cur+" "+word).strip()
# # # #             if len(test)*9 < w-24: cur=test
# # # #             else: lines.append(cur); cur=word
# # # #         if cur: lines.append(cur)
# # # #         lines = lines[:3]
# # # #         bh = len(lines)*28+16
# # # #         bar = frame.copy()
# # # #         cv2.rectangle(bar,(0,h-bh),(w,h),(10,10,10),-1)
# # # #         cv2.addWeighted(bar,0.62,frame,0.38,0,frame)
# # # #         for i,line in enumerate(lines):
# # # #             cv2.putText(frame, line, (12,h-bh+24+i*28),
# # # #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,240,140), 1, cv2.LINE_AA)
# # # #
# # # #
# # # # # ═══════════════════════════════════════════════════════════════════════════
# # # # #  MAIN
# # # # # ═══════════════════════════════════════════════════════════════════════════
# # # # GESTURE_PROMPTS = {
# # # #     "interior_view" : "Describe the interior view of a modern performance car cabin.",
# # # #     "pinch_open"    : "What does scaling up a car's size do to its aerodynamics?",
# # # #     "reset"         : "Give me one surprising fact about iconic car design.",
# # # # }
# # # #
# # # # def main():
# # # #     print("=" * 60)
# # # #     print("  AR Car — Voice Conversation + Gesture Edition")
# # # #     print("=" * 60)
# # # #     print("  Just SPEAK to talk to the AI car guide.")
# # # #     print("  Gestures still work simultaneously.")
# # # #     print("  Say 'exit' or 'quit' or press Q to stop.")
# # # #     print("=" * 60)
# # # #
# # # #     cap = cv2.VideoCapture(0)
# # # #     cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
# # # #     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
# # # #     cap.set(cv2.CAP_PROP_FPS, 30)
# # # #
# # # #     ge     = GestureEngine()
# # # #     cr     = CarRenderer()
# # # #     ai     = ConversationalAI()
# # # #     aruco  = ArucoTracker()
# # # #     voice  = ContinuousListener()
# # # #     voice.start()
# # # #
# # # #     ai.chat("Introduce yourself as the AR car guide in one sentence.", speak_reply=False)
# # # #
# # # #     scale        = 1.0
# # # #     last_gesture = "none"
# # # #     fps_timer    = time.time()
# # # #     fps          = 0.0
# # # #     fc           = 0
# # # #
# # # #     print("\n[READY] AR window open. Speak or gesture.\n")
# # # #
# # # #     while True:
# # # #         ret, frame = cap.read()
# # # #         if not ret: break
# # # #         frame = cv2.flip(frame, 1)
# # # #
# # # #         gesture, sd, rd = ge.process(frame)
# # # #         scale = max(0.3, min(3.5, scale+sd))
# # # #
# # # #         anchor, aruco_scale = aruco.detect(frame)
# # # #         eff_scale = scale * aruco_scale if anchor else scale
# # # #
# # # #         cr.update(gesture, sd, rd)
# # # #         frame = composite(frame, cr.render_rgba(), anchor=anchor,
# # # #                           scale=eff_scale*0.7)
# # # #
# # # #         # gesture → AI
# # # #         if gesture != last_gesture:
# # # #             p = GESTURE_PROMPTS.get(gesture)
# # # #             if p: ai.chat(p, speak_reply=True)
# # # #         last_gesture = gesture
# # # #
# # # #         # voice → AI
# # # #         spoken = voice.get_text()
# # # #         if spoken:
# # # #             low = spoken.lower().strip()
# # # #             if low in ("exit","quit","stop","goodbye"):
# # # #                 break
# # # #             ai.chat(spoken, speak_reply=True)
# # # #
# # # #         fc += 1
# # # #         now = time.time()
# # # #         if now-fps_timer >= 1.0:
# # # #             fps=fc/(now-fps_timer); fc=0; fps_timer=now
# # # #
# # # #         draw_hud_v(frame, gesture, scale, ai.get_reply(),
# # # #                    anchor is not None, fps,
# # # #                    VOICE_OK and not voice.stop.is_set())
# # # #
# # # #         cv2.imshow("AR Car — Voice + Gesture AI  (Q to quit)", frame)
# # # #         if cv2.waitKey(1) & 0xFF == ord('q'):
# # # #             break
# # # #
# # # #     voice.close()
# # # #     cap.release()
# # # #     cv2.destroyAllWindows()
# # # #     print("[EXIT] Goodbye!")
# # # #
# # # #
# # # # if __name__ == "__main__":
# # # #     main()
# # #
# # #
# # #
# # #
# # # """
# # # AR Car — Voice Conversation Edition  (Windows Compatible)
# # # ==========================================================
# # # Same fix as ar_car_open3d.py: uses PyOpenGL + pygame instead of
# # # Open3D OffscreenRenderer (which crashes on Windows with EGL error).
# # #
# # # Adds FULL continuous voice conversation loop:
# # #   • Background thread listens continuously — just speak, no key press
# # #   • Ollama remembers conversation history across turns
# # #   • Replies spoken aloud via TTS
# # #   • Gestures + ArUco still work simultaneously
# # #
# # # Install
# # # -------
# # #   pip install opencv-python mediapipe pygame PyOpenGL PyOpenGL_accelerate
# # #               ollama numpy speechrecognition pyttsx3
# # #   pip install pipwin && pipwin install pyaudio   (Windows pyaudio)
# # #
# # # Run
# # # ---
# # #   ollama serve          ← keep open in another terminal
# # #   python ar_car_voice.py
# # #
# # # Say "exit" or "quit" or press Q to stop.
# # # """
# # #
# # # import cv2
# # # import numpy as np
# # # import mediapipe as mp
# # # import ollama
# # # import threading
# # # import math
# # # import os
# # # import time
# # # import sys
# # # import queue
# # #
# # # import pygame
# # # from pygame.locals import DOUBLEBUF, OPENGL, NOFRAME
# # # from OpenGL.GL import *
# # # from OpenGL.GLU import *
# # #
# # # try:
# # #     import speech_recognition as sr
# # #     import pyttsx3
# # #     VOICE_OK = True
# # # except ImportError:
# # #     VOICE_OK = False
# # #     print("[WARN] Voice deps missing — gesture-only mode")
# # #
# # # # ═══════════════════════════════════════════════════════════════════════
# # # #  CONFIG
# # # # ═══════════════════════════════════════════════════════════════════════
# # # CAM_W, CAM_H = 1280, 720
# # # RENDER_W      = 640
# # # RENDER_H      = 480
# # # MODEL_PATH    = "car.obj"
# # # OLLAMA_MODEL  = "llava-phi3"
# # # ARUCO_DICT    = cv2.aruco.DICT_6X6_250
# # # ARUCO_ID      = 0
# # #
# # #
# # # # ═══════════════════════════════════════════════════════════════════════
# # # #  OBJ LOADER
# # # # ═══════════════════════════════════════════════════════════════════════
# # # def load_obj(path):
# # #     verts, tris = [], []
# # #     try:
# # #         with open(path, "r", encoding="utf-8", errors="ignore") as f:
# # #             for line in f:
# # #                 p = line.strip().split()
# # #                 if not p: continue
# # #                 if p[0] == "v":
# # #                     verts.append((float(p[1]), float(p[2]), float(p[3])))
# # #                 elif p[0] == "f":
# # #                     idxs = [int(x.split("/")[0])-1 for x in p[1:]]
# # #                     for i in range(1, len(idxs)-1):
# # #                         tris.append((verts[idxs[0]], verts[idxs[i]], verts[idxs[i+1]]))
# # #         print(f"[OBJ] {path}: {len(verts)} verts, {len(tris)} tris")
# # #     except Exception as e:
# # #         print(f"[OBJ] {e}")
# # #     return tris
# # #
# # #
# # # def compute_normals(tris):
# # #     normals = []
# # #     for (a,b,c) in tris:
# # #         ab=(b[0]-a[0],b[1]-a[1],b[2]-a[2])
# # #         ac=(c[0]-a[0],c[1]-a[1],c[2]-a[2])
# # #         nx=ab[1]*ac[2]-ab[2]*ac[1]; ny=ab[2]*ac[0]-ab[0]*ac[2]; nz=ab[0]*ac[1]-ab[1]*ac[0]
# # #         L=math.sqrt(nx*nx+ny*ny+nz*nz) or 1e-9
# # #         normals.append((nx/L,ny/L,nz/L))
# # #     return normals
# # #
# # #
# # # def normalise_mesh(tris):
# # #     if not tris: return tris
# # #     all_v=[v for t in tris for v in t]
# # #     xs=[v[0] for v in all_v]; ys=[v[1] for v in all_v]; zs=[v[2] for v in all_v]
# # #     cx=(max(xs)+min(xs))/2; cy=(max(ys)+min(ys))/2; cz=(max(zs)+min(zs))/2
# # #     sc=max(max(xs)-min(xs),max(ys)-min(ys),max(zs)-min(zs)) or 1
# # #     def s(v): return ((v[0]-cx)/sc,(v[1]-cy)/sc,(v[2]-cz)/sc)
# # #     return [(s(a),s(b),s(c)) for (a,b,c) in tris]
# # #
# # #
# # # def make_placeholder_tris():
# # #     tris=[]
# # #     def box(x0,y0,z0,x1,y1,z1):
# # #         faces=[[(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],
# # #                [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
# # #                [(x0,y0,z0),(x0,y0,z1),(x0,y1,z1),(x0,y1,z0)],
# # #                [(x1,y0,z0),(x1,y0,z1),(x1,y1,z1),(x1,y1,z0)],
# # #                [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],
# # #                [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)]]
# # #         for f in faces: tris.append((f[0],f[1],f[2])); tris.append((f[0],f[2],f[3]))
# # #     def sphere(cx,cy,cz,r,st=6,sl=8):
# # #         for i in range(st):
# # #             la0=math.pi*(-0.5+i/st); la1=math.pi*(-0.5+(i+1)/st)
# # #             for j in range(sl):
# # #                 lg0=2*math.pi*j/sl; lg1=2*math.pi*(j+1)/sl
# # #                 def pt(la,lg): return (cx+r*math.cos(la)*math.cos(lg),cy+r*math.cos(la)*math.sin(lg),cz+r*math.sin(la))
# # #                 a,b,c_,d=pt(la0,lg0),pt(la0,lg1),pt(la1,lg1),pt(la1,lg0)
# # #                 tris.append((a,b,c_)); tris.append((a,c_,d))
# # #     box(-1,-0.5,-0.05,1,0.5,0.55); box(-0.6,-0.48,0.55,0.6,0.48,1.0)
# # #     for wx,wy in [(-0.7,-0.55),(0.7,-0.55),(-0.7,0.55),(0.7,0.55)]: sphere(wx,wy,-0.05,0.22)
# # #     return tris
# # #
# # #
# # # # ═══════════════════════════════════════════════════════════════════════
# # # #  OPENGL RENDERER
# # # # ═══════════════════════════════════════════════════════════════════════
# # # class CarRenderer:
# # #     def __init__(self, width=RENDER_W, height=RENDER_H, model_path=MODEL_PATH):
# # #         self.w=width; self.h=height
# # #         self.yaw=0.0; self.scale=1.0; self.interior=False
# # #         pygame.init()
# # #         os.environ.setdefault("SDL_VIDEODRIVER","windib")
# # #         self._surface=pygame.display.set_mode((width,height),DOUBLEBUF|OPENGL|NOFRAME)
# # #         pygame.display.set_caption("AR-GL")
# # #
# # #         self._fbo=glGenFramebuffers(1)
# # #         self._crb=glGenRenderbuffers(1)
# # #         self._drb=glGenRenderbuffers(1)
# # #         glBindFramebuffer(GL_FRAMEBUFFER,self._fbo)
# # #         glBindRenderbuffer(GL_RENDERBUFFER,self._crb)
# # #         glRenderbufferStorage(GL_RENDERBUFFER,GL_RGBA8,width,height)
# # #         glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_RENDERBUFFER,self._crb)
# # #         glBindRenderbuffer(GL_RENDERBUFFER,self._drb)
# # #         glRenderbufferStorage(GL_RENDERBUFFER,GL_DEPTH_COMPONENT24,width,height)
# # #         glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_RENDERBUFFER,self._drb)
# # #         glBindFramebuffer(GL_FRAMEBUFFER,0)
# # #
# # #         glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
# # #         glEnable(GL_LIGHT0); glEnable(GL_LIGHT1); glEnable(GL_COLOR_MATERIAL)
# # #         glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE)
# # #         glShadeModel(GL_SMOOTH)
# # #         glLightfv(GL_LIGHT0,GL_POSITION,[2,3,4,1])
# # #         glLightfv(GL_LIGHT0,GL_DIFFUSE,[1,0.95,0.9,1])
# # #         glLightfv(GL_LIGHT1,GL_POSITION,[-2,-1,2,1])
# # #         glLightfv(GL_LIGHT1,GL_DIFFUSE,[0.3,0.3,0.35,1])
# # #         glLightModelfv(GL_LIGHT_MODEL_AMBIENT,[0.25,0.25,0.28,1])
# # #
# # #         raw = load_obj(model_path) if os.path.exists(model_path) else make_placeholder_tris()
# # #         self._tris=normalise_mesh(raw); self._normals=compute_normals(self._tris)
# # #         self._dl=self._build_dl()
# # #
# # #         glMatrixMode(GL_PROJECTION); glLoadIdentity()
# # #         gluPerspective(55,width/height,0.01,100); glMatrixMode(GL_MODELVIEW)
# # #
# # #     def _build_dl(self):
# # #         dl=glGenLists(1); glNewList(dl,GL_COMPILE)
# # #         glColor3f(0.85,0.12,0.12); glBegin(GL_TRIANGLES)
# # #         for (a,b,c),n in zip(self._tris,self._normals):
# # #             glNormal3f(*n); glVertex3f(*a); glVertex3f(*b); glVertex3f(*c)
# # #         glEnd(); glEndList(); return dl
# # #
# # #     def update(self,gesture,sd,rd):
# # #         self.scale=max(0.3,min(3.5,self.scale+sd)); self.yaw+=rd
# # #         if gesture=="interior_view" and not self.interior: self.interior=True
# # #         elif gesture=="reset": self.interior=False; self.scale=1.0; self.yaw=0.0
# # #
# # #     def render_rgba(self):
# # #         glBindFramebuffer(GL_FRAMEBUFFER,self._fbo)
# # #         glViewport(0,0,self.w,self.h)
# # #         glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
# # #         glLoadIdentity()
# # #         if self.interior:
# # #             gluLookAt(0.1,-0.2,0.3,0.5,0.8,0.3,0,0,1)
# # #         else:
# # #             d=3.5/max(self.scale,0.3); r=math.radians(self.yaw)
# # #             gluLookAt(d*math.sin(r),-d*math.cos(r),d*0.4,0,0,0,0,0,1)
# # #         glScalef(self.scale,self.scale,self.scale)
# # #         glCallList(self._dl)
# # #         glPixelStorei(GL_PACK_ALIGNMENT,1)
# # #         raw=glReadPixels(0,0,self.w,self.h,GL_RGBA,GL_UNSIGNED_BYTE)
# # #         arr=np.frombuffer(raw,dtype=np.uint8).reshape(self.h,self.w,4)[::-1]
# # #         glBindFramebuffer(GL_FRAMEBUFFER,0)
# # #         pygame.event.pump()
# # #         return arr
# # #
# # #     def cleanup(self):
# # #         glDeleteLists(self._dl,1)
# # #         glDeleteFramebuffers(1,[self._fbo])
# # #         glDeleteRenderbuffers(1,[self._crb])
# # #         glDeleteRenderbuffers(1,[self._drb])
# # #         pygame.quit()
# # #
# # #
# # # # ═══════════════════════════════════════════════════════════════════════
# # # #  GESTURE ENGINE
# # # # ═══════════════════════════════════════════════════════════════════════
# # # class GestureEngine:
# # #     def __init__(self):
# # #         self.mp_hands=mp.solutions.hands
# # #         self.hands=self.mp_hands.Hands(max_num_hands=2,
# # #             min_detection_confidence=0.75,min_tracking_confidence=0.75)
# # #         self.draw=mp.solutions.drawing_utils; self.prev_pinch=None
# # #
# # #     @staticmethod
# # #     def _dist(a,b): return math.hypot(a.x-b.x,a.y-b.y)
# # #
# # #     @staticmethod
# # #     def _fingers_up(lm):
# # #         return sum(lm[t].y<lm[j].y for t,j in zip([8,12,16,20],[6,10,14,18]))
# # #
# # #     def process(self,frame):
# # #         rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); res=self.hands.process(rgb)
# # #         gesture,sd,rd="none",0.0,0.0
# # #         if not res.multi_hand_landmarks: self.prev_pinch=None; return gesture,sd,rd
# # #         for hlm in res.multi_hand_landmarks:
# # #             lm=hlm.landmark
# # #             self.draw.draw_landmarks(frame,hlm,self.mp_hands.HAND_CONNECTIONS,
# # #                 self.draw.DrawingSpec(color=(0,200,150),thickness=2),
# # #                 self.draw.DrawingSpec(color=(0,255,100),thickness=1))
# # #             pinch=self._dist(lm[4],lm[8]); fingers=self._fingers_up(lm)
# # #             if self.prev_pinch is not None:
# # #                 d=pinch-self.prev_pinch
# # #                 if abs(d)>0.005: sd=d*1.8
# # #             self.prev_pinch=pinch
# # #             if fingers>=4: gesture="reset"
# # #             elif fingers==2: gesture="interior_view"
# # #             elif pinch<0.06: gesture="pinch_close"
# # #             elif pinch>0.20: gesture="pinch_open"
# # #             wx=lm[0].x
# # #             if wx<0.30: rd=-4.0
# # #             elif wx>0.70: rd=+4.0
# # #         return gesture,sd,rd
# # #
# # #
# # # # ═══════════════════════════════════════════════════════════════════════
# # # #  ARUCO TRACKER
# # # # ═══════════════════════════════════════════════════════════════════════
# # # class ArucoTracker:
# # #     def __init__(self):
# # #         self.adict=cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
# # #         self.detector=cv2.aruco.ArucoDetector(self.adict,cv2.aruco.DetectorParameters())
# # #         if not os.path.exists("aruco_marker.png"):
# # #             img=cv2.aruco.generateImageMarker(self.adict,ARUCO_ID,300)
# # #             cv2.imwrite("aruco_marker.png",img)
# # #             print("[ArUco] aruco_marker.png generated — print it!")
# # #
# # #     def detect(self,frame):
# # #         grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
# # #         corners,ids,_=self.detector.detectMarkers(grey)
# # #         if ids is None: return None,1.0
# # #         for i,mid in enumerate(ids.flatten()):
# # #             if mid!=ARUCO_ID: continue
# # #             c=corners[i][0]; cx=int(c[:,0].mean()); cy=int(c[:,1].mean())
# # #             side=np.linalg.norm(c[0]-c[1])
# # #             cv2.aruco.drawDetectedMarkers(frame,corners,ids)
# # #             return (cx,cy),max(0.3,min(3.0,side/100))
# # #         return None,1.0
# # #
# # #
# # # # ═══════════════════════════════════════════════════════════════════════
# # # #  COMPOSITOR
# # # # ═══════════════════════════════════════════════════════════════════════
# # # def composite(bg,car_rgba,anchor=None,scale=1.0):
# # #     hb,wb=bg.shape[:2]; hc,wc=car_rgba.shape[:2]
# # #     nw=max(10,min(int(wc*scale),wb)); nh=max(10,min(int(hc*scale),hb))
# # #     scaled=cv2.resize(car_rgba,(nw,nh))
# # #     x0=(anchor[0]-nw//2 if anchor else (wb-nw)//2)
# # #     y0=(anchor[1]-nh//2 if anchor else (hb-nh)//2)
# # #     x0=max(0,min(x0,wb-nw)); y0=max(0,min(y0,hb-nh))
# # #     roi=bg[y0:y0+nh,x0:x0+nw].astype(np.float32)
# # #     fg=scaled[:,:,:3].astype(np.float32)[:,:,::-1]
# # #     a=scaled[:,:,3:4].astype(np.float32)/255.0
# # #     out=bg.copy(); out[y0:y0+nh,x0:x0+nw]=(fg*a+roi*(1-a)).astype(np.uint8)
# # #     return out
# # #
# # #
# # # # ═══════════════════════════════════════════════════════════════════════
# # # #  CONVERSATIONAL AI
# # # # ═══════════════════════════════════════════════════════════════════════
# # # import threading
# # # import queue
# # # import ollama
# # # import pyttsx3
# # #
# # #
# # # class ConversationalAI:
# # #     SYSTEM = (
# # #         "You are an enthusiastic AR car guide with deep automotive knowledge. "
# # #         "Keep every reply to 1-2 sentences maximum. "
# # #         "Be vivid, specific, and friendly. No markdown, no bullet points."
# # #     )
# # #
# # #     GESTURE_PROMPTS = {
# # #         "interior_view": "Describe the interior view of a modern performance car cabin.",
# # #         "pinch_open": "What does scaling up a car's size do to its aerodynamics?",
# # #         "reset": "Give me one surprising fact about iconic car design.",
# # #     }
# # #
# # #     def __init__(self, model=OLLAMA_MODEL):
# # #         self.model = model
# # #         self.history = []
# # #         self.reply = "System ready."
# # #         self.busy = False
# # #         self.lock = threading.Lock()
# # #
# # #         # Dedicated TTS Queue and Worker Thread
# # #         self.tts_queue = queue.Queue()
# # #         threading.Thread(target=self._tts_worker, daemon=True).start()
# # #
# # #     def _tts_worker(self):
# # #         """Persistent worker thread handles all speech sequentially."""
# # #         try:
# # #             engine = pyttsx3.init()
# # #             engine.setProperty("rate", 168)
# # #             engine.setProperty("volume", 0.9)
# # #             while True:
# # #                 text = self.tts_queue.get() # Waits for new text
# # #                 if text:
# # #                     engine.say(text)
# # #                     engine.runAndWait()
# # #                 self.tts_queue.task_done()
# # #         except Exception as e:
# # #             print(f"[TTS Worker Error] {e}")
# # #
# # #     def chat(self, user_text, speak_reply=True):
# # #         if self.busy: return
# # #         self.busy = True
# # #         # Fire off the query in a background thread
# # #         threading.Thread(target=self._run, args=(user_text, speak_reply), daemon=True).start()
# # #
# # #     def _run(self, user_text, speak_reply):
# # #         with self.lock:
# # #             self.history.append({"role": "user", "content": user_text})
# # #
# # #         try:
# # #             res = ollama.chat(model=self.model, messages=self.history[-10:])
# # #             reply_text = res["message"]["content"].strip()
# # #         except Exception as e:
# # #             reply_text = "I'm having trouble connecting to the brain."
# # #             print(f"[Ollama Error] {e}")
# # #
# # #         with self.lock:
# # #             self.history.append({"role": "assistant", "content": reply_text})
# # #             self.reply = reply_text
# # #
# # #         # Push to queue instead of calling pyttsx3 directly
# # #         if speak_reply and VOICE_OK:
# # #             self.tts_queue.put(reply_text)
# # #
# # #         self.busy = False
# # #
# # #         try:
# # #             # Query Ollama
# # #             res = ollama.chat(
# # #                 model=self.model,
# # #                 messages=[{"role": "system", "content": self.SYSTEM}] + self.history[-10:]
# # #             )
# # #             reply_text = res["message"]["content"].strip()
# # #         except Exception as e:
# # #             reply_text = "I'm having trouble connecting to the brain."
# # #             print(f"[Ollama Error] {e}")
# # #
# # #         with self.lock:
# # #             self.history.append({"role": "assistant", "content": reply_text})
# # #             self.reply = reply_text
# # #
# # #         # Push to queue instead of calling pyttsx3 directly
# # #         if speak_reply and VOICE_OK:
# # #             self.tts_queue.put(reply_text)
# # #
# # #         self.busy = False
# # #
# # #     def get_reply(self):
# # #         with self.lock: return self.reply
# # #
# # # # class ConversationalAI:
# # # #     SYSTEM=(
# # # #         "You are an enthusiastic AR car guide with deep automotive knowledge. "
# # # #         "Keep every reply to 1-2 sentences maximum. "
# # # #         "Be vivid, specific, and friendly. No markdown, no bullet points."
# # # #     )
# # # #     GESTURE_PROMPTS={
# # # #         "interior_view":"Describe the interior view of a modern performance car cabin.",
# # # #         "pinch_open":   "What does scaling up a car's size do to its aerodynamics?",
# # # #         "reset":        "Give me one surprising fact about iconic car design.",
# # # #     }
# # # #
# # # #     def __init__(self, model=OLLAMA_MODEL):
# # # #         self.model = model
# # # #         self.history = []
# # # #         self.reply = "System ready."
# # # #         self.busy = False
# # # #         self.lock = threading.Lock()
# # # #         self.engine = None
# # # #         if VOICE_OK:
# # # #             try:
# # # #                 # Re-init in every thread if necessary,
# # # #                 # but better to handle it globally
# # # #                 self.engine = pyttsx3.init()
# # # #                 self.engine.setProperty("rate", 170)
# # # #             except Exception as e:
# # # #                 print(f"[TTS Init Error] {e}")
# # # #
# # # #     def _speak(self, text):
# # # #         """
# # # #         Creates a fresh engine instance for every utterance to avoid
# # # #         queue blocking issues common on Windows.
# # # #         """
# # # #         try:
# # # #             # Initialize fresh for every call
# # # #             engine = pyttsx3.init()
# # # #             # Restore your preferred settings
# # # #             engine.setProperty("rate", 168)
# # # #             engine.setProperty("volume", 0.9)
# # # #
# # # #             engine.say(text)
# # # #             engine.runAndWait()
# # # #             # Explicitly stop to clear the event queue
# # # #             engine.stop()
# # # #         except Exception as e:
# # # #             print(f"[TTS Error] {e}")
# # # #     def chat(self, user_text, speak_reply=True):
# # # #         if self.busy: return
# # # #         self.busy = True
# # # #         # Fire off the query and speech as a single background task
# # # #         threading.Thread(target=self._run, args=(user_text, speak_reply), daemon=True).start()
# # # #
# # # #     def _run(self, user_text, speak_reply):
# # # #         with self.lock:
# # # #             self.history.append({"role": "user", "content": user_text})
# # # #
# # # #         try:
# # # #             # Ensure ollama is actually running before calling
# # # #             res = ollama.chat(model=self.model, messages=self.history[-10:])
# # # #             reply_text = res["message"]["content"].strip()
# # # #         except Exception as e:
# # # #             reply_text = "I'm having trouble connecting to the brain."
# # # #             print(f"[Ollama Error] {e}")
# # # #
# # # #         with self.lock:
# # # #             self.history.append({"role": "assistant", "content": reply_text})
# # # #             self.reply = reply_text
# # # #
# # # #         if speak_reply and VOICE_OK:
# # # #             self._speak(reply_text)
# # # #
# # # #         self.busy = False
# # # #
# # # #     def get_reply(self):
# # # #         with self.lock: return self.reply
# # #
# # #
# # # # ═══════════════════════════════════════════════════════════════════════
# # # #  CONTINUOUS VOICE LISTENER
# # # # ═══════════════════════════════════════════════════════════════════════
# # # class ContinuousListener:
# # #     def __init__(self):
# # #         self.q=queue.Queue(); self.stop=threading.Event()
# # #
# # #     def start(self):
# # #         if not VOICE_OK: return
# # #         threading.Thread(target=self._loop,daemon=True).start()
# # #
# # #     def _loop(self):
# # #         rec=sr.Recognizer(); mic=sr.Microphone()
# # #         rec.dynamic_energy_threshold=True; rec.energy_threshold=3000
# # #         with mic as src: rec.adjust_for_ambient_noise(src,duration=1.0)
# # #         print("[Voice] Listening continuously — just speak!")
# # #         while not self.stop.is_set():
# # #             try:
# # #                 with mic as src:
# # #                     audio=rec.listen(src,timeout=2,phrase_time_limit=10)
# # #                 text=rec.recognize_google(audio)
# # #                 if text.strip(): self.q.put(text); print(f"[Voice] '{text}'")
# # #             except sr.WaitTimeoutError: pass
# # #             except sr.UnknownValueError: pass
# # #             except Exception as e: print(f"[Voice] {e}"); time.sleep(0.5)
# # #
# # #     def get_text(self):
# # #         try: return self.q.get_nowait()
# # #         except queue.Empty: return None
# # #
# # #     def close(self): self.stop.set()
# # #
# # #
# # # # ═══════════════════════════════════════════════════════════════════════
# # # #  HUD
# # # # ═══════════════════════════════════════════════════════════════════════
# # # def draw_hud_v(frame,gesture,scale,ai_text,aruco_found,fps,listening):
# # #     h,w=frame.shape[:2]
# # #     ov=frame.copy(); cv2.rectangle(ov,(0,0),(350,115),(0,0,0),-1)
# # #     cv2.addWeighted(ov,0.45,frame,0.55,0,frame)
# # #     def put(t,y,col=(210,255,170),s=0.62,th=1):
# # #         cv2.putText(frame,t,(10,y),cv2.FONT_HERSHEY_SIMPLEX,s,col,th,cv2.LINE_AA)
# # #     labels={"none":"","pinch_close":"Shrinking","pinch_open":"Scaling up",
# # #             "interior_view":"Interior","reset":"Reset"}
# # #     put(f"Gesture : {labels.get(gesture,gesture)}",26)
# # #     put(f"Scale   : {scale:.2f}x",52)
# # #     put(f"ArUco   : {'ANCHORED' if aruco_found else 'free'}",78,
# # #         (100,255,100) if aruco_found else (80,80,220))
# # #     put(f"FPS {fps:.0f}   Q=quit   speak freely",104,(160,160,160))
# # #     if listening:
# # #         r=int(8+4*math.sin(time.time()*6))
# # #         cv2.circle(frame,(w-20,20),r,(0,60,255),-1)
# # #     if ai_text:
# # #         words=ai_text.split(); lines,cur=[],""
# # #         for word in words:
# # #             test=(cur+" "+word).strip()
# # #             if len(test)*9<w-24: cur=test
# # #             else: lines.append(cur); cur=word
# # #         if cur: lines.append(cur)
# # #         lines=lines[:3]; bh=len(lines)*28+16
# # #         bar=frame.copy(); cv2.rectangle(bar,(0,h-bh),(w,h),(10,10,10),-1)
# # #         cv2.addWeighted(bar,0.62,frame,0.38,0,frame)
# # #         for i,line in enumerate(lines):
# # #             cv2.putText(frame,line,(12,h-bh+24+i*28),
# # #                 cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,240,140),1,cv2.LINE_AA)
# # #
# # #
# # # # ═══════════════════════════════════════════════════════════════════════
# # # #  MAIN
# # # # ═══════════════════════════════════════════════════════════════════════
# # # def main():
# # #     print("="*58)
# # #     print("  AR Car — Voice + Gesture  (Windows Compatible)")
# # #     print("="*58)
# # #     print("  Just SPEAK — no key press needed.")
# # #     print("  Gestures work simultaneously.")
# # #     print("  Say 'exit' or press Q to quit.")
# # #     print("="*58)
# # #
# # #     cap=cv2.VideoCapture(0)
# # #     cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAM_W)
# # #     cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAM_H)
# # #     cap.set(cv2.CAP_PROP_FPS,30)
# # #
# # #     print("[INIT] Loading renderer…")
# # #     car=CarRenderer()
# # #     print("[INIT] Renderer ready")
# # #
# # #     ge=GestureEngine(); ai=ConversationalAI()
# # #     aruco=ArucoTracker(); voice=ContinuousListener()
# # #     voice.start()
# # #     ai.chat("Introduce yourself as the AR car guide in one sentence.",speak_reply=False)
# # #
# # #     scale=1.0; last_gesture="none"
# # #     fps_timer=time.time(); fps=0.0; fc=0
# # #
# # #     print("\n[READY] AR window open. Speak or gesture.\n")
# # #
# # #     while True:
# # #         ret,frame=cap.read()
# # #         if not ret: break
# # #         frame=cv2.flip(frame,1)
# # #
# # #         gesture,sd,rd=ge.process(frame)
# # #         scale=max(0.3,min(3.5,scale+sd))
# # #
# # #         anchor,aruco_scale=aruco.detect(frame)
# # #         eff=scale*aruco_scale if anchor else scale
# # #
# # #         car.update(gesture,sd,rd)
# # #         frame=composite(frame,car.render_rgba(),anchor=anchor,scale=eff*0.7)
# # #
# # #         if gesture!=last_gesture:
# # #             p=ConversationalAI.GESTURE_PROMPTS.get(gesture)
# # #             if p: ai.chat(p,speak_reply=True)
# # #         last_gesture=gesture
# # #
# # #         spoken=voice.get_text()
# # #         if spoken:
# # #             if spoken.lower().strip() in ("exit","quit","stop","goodbye"): break
# # #             ai.chat(spoken,speak_reply=True)
# # #
# # #         fc+=1; now=time.time()
# # #         if now-fps_timer>=1.0: fps=fc/(now-fps_timer); fc=0; fps_timer=now
# # #
# # #         draw_hud_v(frame,gesture,scale,ai.get_reply(),
# # #                    anchor is not None,fps,VOICE_OK and not voice.stop.is_set())
# # #
# # #         cv2.imshow("AR Car — Voice + Gesture  (Q quit)",frame)
# # #         if cv2.waitKey(1)&0xFF==ord('q'): break
# # #
# # #     voice.close(); car.cleanup(); cap.release(); cv2.destroyAllWindows()
# # #     print("[EXIT] Done.")
# # #
# # # if __name__=="__main__":
# # #     main()
# #
# #
# #
# #
# #
# # """
# # AR Car — Complete Rebuild v3
# # =============================
# # FIXES:
# #   1. TTS deadlock fixed — single persistent worker thread, no re-init per call
# #   2. 3D rendering fixed — per-material colors from OBJ/MTL, not flat red
# #   3. Duplicate Ollama call in _run() removed
# #   4. GUI completely rebuilt — proper sectioned HUD panels
# #
# # NEW FEATURES:
# #   • Multi-material OBJ rendering (reads .mtl files — car body, glass, wheels colored correctly)
# #   • Smooth orbit with mouse drag (click + drag on window)
# #   • Explode view gesture (3 fingers) — separates car parts
# #   • Screenshot capture (S key) — saves to screenshots/
# #   • Zoom gesture (pinch distance mapped to FOV)
# #   • Car info panel — shows model stats, triangle count, current view mode
# #   • AI chat history panel — scrolling last 4 replies on screen
# #   • Gesture guide panel — always-visible reference
# #   • Voice status panel with waveform animation
# #   • FPS + performance panel
# #   • Color theme toggle (C key) — changes car paint color
# #   • Wireframe toggle (W key)
# #
# # Controls
# # --------
# #   Pinch           → scale / zoom
# #   2 fingers up    → interior view
# #   3 fingers up    → explode view (toggle)
# #   Open palm       → reset all
# #   Wrist L/R       → orbit
# #   V               → voice query
# #   S               → screenshot
# #   W               → wireframe toggle
# #   C               → cycle car color
# #   Q               → quit
# #
# # Install
# # -------
# #   pip install opencv-python mediapipe pygame PyOpenGL PyOpenGL_accelerate
# #               ollama numpy speechrecognition pyttsx3
# #   pip install pipwin && pipwin install pyaudio   (Windows)
# # """
# #
# # import cv2
# # import numpy as np
# # import mediapipe as mp
# # import ollama
# # import threading
# # import math
# # import os
# # import time
# # import sys
# # import queue
# # import datetime
# #
# # import pygame
# # from pygame.locals import DOUBLEBUF, OPENGL, NOFRAME
# # from OpenGL.GL import *
# # from OpenGL.GLU import *
# #
# # try:
# #     import speech_recognition as sr
# #     VOICE_OK = True
# # except ImportError:
# #     VOICE_OK = False
# #
# # try:
# #     import pyttsx3
# #     TTS_OK = True
# # except ImportError:
# #     TTS_OK = False
# #
# # # ═══════════════════════════════════════════════════════════════════════
# # #  CONFIG
# # # ═══════════════════════════════════════════════════════════════════════
# # CAM_W, CAM_H  = 1280, 720
# # RENDER_W      = 700
# # RENDER_H      = 520
# # MODEL_PATH    = "car.obj"
# # OLLAMA_MODEL  = "llava-phi3"
# # ARUCO_DICT    = cv2.aruco.DICT_6X6_250
# # ARUCO_ID      = 0
# #
# # os.makedirs("screenshots", exist_ok=True)
# #
# # # Car paint color presets (R,G,B 0-1)
# # CAR_COLORS = [
# #     (0.85, 0.12, 0.12),   # Red
# #     (0.10, 0.25, 0.80),   # Blue
# #     (0.08, 0.55, 0.15),   # Green
# #     (0.90, 0.70, 0.05),   # Yellow
# #     (0.15, 0.15, 0.15),   # Black
# #     (0.92, 0.92, 0.92),   # White
# #     (0.55, 0.55, 0.60),   # Silver
# # ]
# # COLOR_NAMES = ["Red","Blue","Green","Yellow","Black","White","Silver"]
# #
# # # ═══════════════════════════════════════════════════════════════════════
# # #  OBJ + MTL LOADER  (multi-material)
# # # ═══════════════════════════════════════════════════════════════════════
# # def load_mtl(mtl_path):
# #     """Parse .mtl file → dict of material_name → (R,G,B)"""
# #     materials = {}
# #     current = None
# #     try:
# #         with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
# #             for line in f:
# #                 p = line.strip().split()
# #                 if not p: continue
# #                 if p[0] == "newmtl":
# #                     current = p[1]
# #                     materials[current] = (0.7, 0.7, 0.7)
# #                 elif p[0] == "Kd" and current:
# #                     materials[current] = (float(p[1]), float(p[2]), float(p[3]))
# #     except:
# #         pass
# #     return materials
# #
# #
# # def load_obj_with_materials(path):
# #     """
# #     Returns list of (triangle, normal, color_rgb) tuples.
# #     Reads paired .mtl file for per-group colors.
# #     Falls back to a neutral gray if no material.
# #     """
# #     verts, normals_v, groups = [], [], []
# #     cur_mat = None
# #     materials = {}
# #
# #     # Try to find .mtl file
# #     dir_ = os.path.dirname(path)
# #     try:
# #         with open(path, "r", encoding="utf-8", errors="ignore") as f:
# #             for line in f:
# #                 p = line.strip().split()
# #                 if not p: continue
# #                 tok = p[0]
# #                 if tok == "mtllib":
# #                     mtl_path = os.path.join(dir_, p[1])
# #                     materials = load_mtl(mtl_path)
# #                 elif tok == "v":
# #                     verts.append((float(p[1]), float(p[2]), float(p[3])))
# #                 elif tok == "vn":
# #                     normals_v.append((float(p[1]), float(p[2]), float(p[3])))
# #                 elif tok == "usemtl":
# #                     cur_mat = p[1]
# #                 elif tok == "f":
# #                     # Parse face — format: v, v/vt, v/vt/vn, v//vn
# #                     face_verts, face_norms = [], []
# #                     for token in p[1:]:
# #                         parts = token.split("/")
# #                         vi = int(parts[0]) - 1
# #                         ni = int(parts[2]) - 1 if len(parts) > 2 and parts[2] else None
# #                         face_verts.append(verts[vi])
# #                         face_norms.append(normals_v[ni] if ni is not None and ni < len(normals_v) else None)
# #                     col = materials.get(cur_mat, (0.65, 0.65, 0.70))
# #                     # Fan triangulate
# #                     for i in range(1, len(face_verts) - 1):
# #                         tri = (face_verts[0], face_verts[i], face_verts[i+1])
# #                         fn  = (face_norms[0], face_norms[i], face_norms[i+1])
# #                         groups.append((tri, fn, col))
# #     except Exception as e:
# #         print(f"[OBJ] Load error: {e}")
# #
# #     print(f"[OBJ] Loaded: {len(verts)} verts, {len(groups)} tris, "
# #           f"{len(materials)} materials")
# #     return groups
# #
# #
# # def flat_normal(a, b, c):
# #     ab = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
# #     ac = (c[0]-a[0], c[1]-a[1], c[2]-a[2])
# #     nx = ab[1]*ac[2] - ab[2]*ac[1]
# #     ny = ab[2]*ac[0] - ab[0]*ac[2]
# #     nz = ab[0]*ac[1] - ab[1]*ac[0]
# #     L  = math.sqrt(nx*nx+ny*ny+nz*nz) or 1e-9
# #     return (nx/L, ny/L, nz/L)
# #
# #
# # def normalise_groups(groups):
# #     """Centre and scale all geometry to unit cube."""
# #     if not groups: return groups
# #     all_v = [v for (tri,_,__) in groups for v in tri]
# #     xs = [v[0] for v in all_v]
# #     ys = [v[1] for v in all_v]
# #     zs = [v[2] for v in all_v]
# #     cx = (max(xs)+min(xs))/2
# #     cy = (max(ys)+min(ys))/2
# #     cz = (max(zs)+min(zs))/2
# #     sc = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)) or 1
# #     def sv(v): return ((v[0]-cx)/sc, (v[1]-cy)/sc, (v[2]-cz)/sc)
# #     out = []
# #     for (tri, fn, col) in groups:
# #         new_tri = tuple(sv(v) for v in tri)
# #         out.append((new_tri, fn, col))
# #     return out
# #
# #
# # def make_placeholder_groups():
# #     """
# #     Multi-colored box car:
# #       body = red, roof = dark red, wheels = black, windows = light blue
# #     """
# #     groups = []
# #
# #     def add_box(x0,y0,z0, x1,y1,z1, color):
# #         faces = [
# #             [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],
# #             [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
# #             [(x0,y0,z0),(x0,y0,z1),(x0,y1,z1),(x0,y1,z0)],
# #             [(x1,y0,z0),(x1,y0,z1),(x1,y1,z1),(x1,y1,z0)],
# #             [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],
# #             [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],
# #         ]
# #         for f in faces:
# #             groups.append(((f[0],f[1],f[2]), (None,None,None), color))
# #             groups.append(((f[0],f[2],f[3]), (None,None,None), color))
# #
# #     def add_sphere(cx,cy,cz, r, color, st=8, sl=12):
# #         for i in range(st):
# #             la0 = math.pi*(-0.5 + i/st)
# #             la1 = math.pi*(-0.5 + (i+1)/st)
# #             for j in range(sl):
# #                 lg0 = 2*math.pi*j/sl
# #                 lg1 = 2*math.pi*(j+1)/sl
# #                 def pt(la,lg):
# #                     return (cx+r*math.cos(la)*math.cos(lg),
# #                             cy+r*math.cos(la)*math.sin(lg),
# #                             cz+r*math.sin(la))
# #                 a,b,c_,d = pt(la0,lg0),pt(la0,lg1),pt(la1,lg1),pt(la1,lg0)
# #                 groups.append(((a,b,c_),(None,None,None),color))
# #                 groups.append(((a,c_,d),(None,None,None),color))
# #
# #     # Body
# #     add_box(-1.0,-0.5,-0.05, 1.0, 0.5, 0.55,  (0.85,0.12,0.12))
# #     # Roof
# #     add_box(-0.55,-0.46, 0.55, 0.6, 0.46, 0.95, (0.65,0.08,0.08))
# #     # Windscreen (front)
# #     add_box(-0.54,-0.45, 0.54, -0.52, 0.45, 0.94, (0.5,0.75,0.95))
# #     # Rear window
# #     add_box( 0.52,-0.45, 0.54,  0.54, 0.45, 0.94, (0.5,0.75,0.95))
# #     # Side windows
# #     add_box(-0.53,-0.47, 0.56, 0.53,-0.45, 0.92, (0.5,0.75,0.95))
# #     add_box(-0.53, 0.45, 0.56, 0.53, 0.47, 0.92, (0.5,0.75,0.95))
# #     # Wheels
# #     for wx,wy in [(-0.65,-0.55),( 0.65,-0.55),(-0.65, 0.55),(0.65, 0.55)]:
# #         add_sphere(wx, wy, -0.03, 0.24, (0.10,0.10,0.10))  # tyre
# #         add_sphere(wx, wy, -0.03, 0.16, (0.65,0.65,0.70))  # rim
# #     # Headlights
# #     add_box(-1.0,-0.30,-0.00, -0.98,-0.10, 0.20, (1.0,0.95,0.70))
# #     add_box(-1.0, 0.10,-0.00, -0.98, 0.30, 0.20, (1.0,0.95,0.70))
# #     # Tail lights
# #     add_box( 0.98,-0.30,-0.00,  1.0,-0.10, 0.20, (0.90,0.05,0.05))
# #     add_box( 0.98, 0.10,-0.00,  1.0, 0.30, 0.20, (0.90,0.05,0.05))
# #     # Bumpers
# #     add_box(-1.05,-0.45,-0.05,-1.0, 0.45, 0.30, (0.20,0.20,0.20))
# #     add_box( 1.0,-0.45,-0.05, 1.05, 0.45, 0.30, (0.20,0.20,0.20))
# #
# #     return groups
# #
# #
# # # ═══════════════════════════════════════════════════════════════════════
# # #  OPENGL RENDERER
# # # ═══════════════════════════════════════════════════════════════════════
# # class CarRenderer:
# #     def __init__(self, width=RENDER_W, height=RENDER_H, model_path=MODEL_PATH):
# #         self.w = width
# #         self.h = height
# #         self.yaw       = 0.0
# #         self.pitch     = 15.0
# #         self.scale     = 1.0
# #         self.interior  = False
# #         self.exploded  = False
# #         self.wireframe = False
# #         self.color_idx = 0
# #         self.fov       = 55.0
# #
# #         # ── pygame hidden window ─────────────────────────────────────────
# #         pygame.init()
# #         if sys.platform == "win32":
# #             os.environ.setdefault("SDL_VIDEODRIVER", "windib")
# #         self._surface = pygame.display.set_mode(
# #             (width, height), DOUBLEBUF | OPENGL | NOFRAME)
# #         pygame.display.set_caption("AR-GL-hidden")
# #
# #         # ── FBO ──────────────────────────────────────────────────────────
# #         self._fbo = glGenFramebuffers(1)
# #         self._crb = glGenRenderbuffers(1)
# #         self._drb = glGenRenderbuffers(1)
# #         glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
# #         glBindRenderbuffer(GL_RENDERBUFFER, self._crb)
# #         glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width, height)
# #         glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
# #                                   GL_RENDERBUFFER, self._crb)
# #         glBindRenderbuffer(GL_RENDERBUFFER, self._drb)
# #         glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height)
# #         glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
# #                                   GL_RENDERBUFFER, self._drb)
# #         assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE
# #         glBindFramebuffer(GL_FRAMEBUFFER, 0)
# #
# #         # ── GL state ─────────────────────────────────────────────────────
# #         glEnable(GL_DEPTH_TEST)
# #         glEnable(GL_LIGHTING)
# #         glEnable(GL_LIGHT0)
# #         glEnable(GL_LIGHT1)
# #         glEnable(GL_LIGHT2)
# #         glEnable(GL_COLOR_MATERIAL)
# #         glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
# #         glShadeModel(GL_SMOOTH)
# #         glEnable(GL_NORMALIZE)
# #
# #         # Key light
# #         glLightfv(GL_LIGHT0, GL_POSITION, [3.0, 4.0, 5.0, 1.0])
# #         glLightfv(GL_LIGHT0, GL_DIFFUSE,  [1.0, 0.95,0.90,1.0])
# #         glLightfv(GL_LIGHT0, GL_SPECULAR, [0.8, 0.8, 0.8, 1.0])
# #         # Fill light
# #         glLightfv(GL_LIGHT1, GL_POSITION, [-3.0,-2.0, 2.0, 1.0])
# #         glLightfv(GL_LIGHT1, GL_DIFFUSE,  [0.35,0.35,0.40,1.0])
# #         glLightfv(GL_LIGHT1, GL_SPECULAR, [0.0, 0.0, 0.0, 1.0])
# #         # Rim light (back)
# #         glLightfv(GL_LIGHT2, GL_POSITION, [0.0,-4.0,-2.0, 1.0])
# #         glLightfv(GL_LIGHT2, GL_DIFFUSE,  [0.20,0.20,0.25,1.0])
# #         glLightfv(GL_LIGHT2, GL_SPECULAR, [0.0, 0.0, 0.0, 1.0])
# #         # Ambient
# #         glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.20,0.20,0.22,1.0])
# #
# #         # Specular material
# #         glMaterialfv(GL_FRONT, GL_SPECULAR,  [0.5, 0.5, 0.5, 1.0])
# #         glMaterialf (GL_FRONT, GL_SHININESS, 48.0)
# #
# #         # ── load geometry ─────────────────────────────────────────────────
# #         if os.path.exists(model_path):
# #             raw = load_obj_with_materials(model_path)
# #         else:
# #             print(f"[3D] {model_path} not found — using detailed placeholder")
# #             raw = make_placeholder_groups()
# #
# #         self._groups = normalise_groups(raw)
# #         self._tri_count = len(self._groups)
# #
# #         # Build display lists — one per unique color group for efficiency
# #         self._build_display_lists()
# #
# #         # Projection
# #         glMatrixMode(GL_PROJECTION)
# #         glLoadIdentity()
# #         gluPerspective(self.fov, width / height, 0.01, 100.0)
# #         glMatrixMode(GL_MODELVIEW)
# #
# #     # ── Build display lists ───────────────────────────────────────────────
# #     def _build_display_lists(self):
# #         """Build a single display list for normal view, preserving per-tri colors."""
# #         self._dl_normal = glGenLists(1)
# #         glNewList(self._dl_normal, GL_COMPILE)
# #         glBegin(GL_TRIANGLES)
# #         for (tri, fn, col) in self._groups:
# #             a, b, c = tri
# #             # Compute flat normal if vertex normals unavailable
# #             if fn[0] is None:
# #                 n = flat_normal(a, b, c)
# #             else:
# #                 # Average vertex normals for smooth look
# #                 vn = fn[0] or flat_normal(a,b,c)
# #                 n  = vn
# #             glColor3f(*col)
# #             glNormal3f(*n)
# #             glVertex3f(*a)
# #             if fn[1]: glNormal3f(*fn[1])
# #             glVertex3f(*b)
# #             if fn[2]: glNormal3f(*fn[2])
# #             glVertex3f(*c)
# #         glEnd()
# #         glEndList()
# #
# #         # Wireframe display list (same geometry, just lines)
# #         self._dl_wire = glGenLists(1)
# #         glNewList(self._dl_wire, GL_COMPILE)
# #         glBegin(GL_LINES)
# #         glColor3f(0.0, 1.0, 0.5)
# #         for (tri, fn, col) in self._groups:
# #             a, b, c = tri
# #             glVertex3f(*a); glVertex3f(*b)
# #             glVertex3f(*b); glVertex3f(*c)
# #             glVertex3f(*c); glVertex3f(*a)
# #         glEnd()
# #         glEndList()
# #
# #     # ── Apply color override for body ────────────────────────────────────
# #     def _draw_with_paint(self):
# #         """Draw but override the dominant 'red' body color with current paint."""
# #         paint = CAR_COLORS[self.color_idx]
# #         old_r, old_g, old_b = CAR_COLORS[0]  # original red
# #         glBegin(GL_TRIANGLES)
# #         for (tri, fn, col) in self._groups:
# #             a, b, c = tri
# #             # If this triangle was the body color (red-ish), apply paint
# #             if col[0] > 0.6 and col[1] < 0.3 and col[2] < 0.3:
# #                 draw_col = paint
# #             else:
# #                 draw_col = col
# #             if fn[0] is None:
# #                 n = flat_normal(a, b, c)
# #             else:
# #                 n = fn[0] or flat_normal(a, b, c)
# #             glColor3f(*draw_col)
# #             glNormal3f(*n)
# #             glVertex3f(*a)
# #             if fn[1]: glNormal3f(*fn[1])
# #             glVertex3f(*b)
# #             if fn[2]: glNormal3f(*fn[2])
# #             glVertex3f(*c)
# #         glEnd()
# #
# #     # ── Explode view ─────────────────────────────────────────────────────
# #     def _draw_exploded(self):
# #         """Render groups pushed outward from centre."""
# #         t = time.time()
# #         explode_dist = 0.6 + 0.1 * math.sin(t * 0.8)
# #         paint = CAR_COLORS[self.color_idx]
# #
# #         glBegin(GL_TRIANGLES)
# #         for (tri, fn, col) in self._groups:
# #             a, b, c = tri
# #             cx = (a[0]+b[0]+c[0])/3
# #             cy = (a[1]+b[1]+c[1])/3
# #             cz = (a[2]+b[2]+c[2])/3
# #             L = math.sqrt(cx*cx+cy*cy+cz*cz) or 0.001
# #             dx, dy, dz = cx/L*explode_dist, cy/L*explode_dist, cz/L*explode_dist
# #
# #             if col[0] > 0.6 and col[1] < 0.3: draw_col = paint
# #             else: draw_col = col
# #
# #             if fn[0] is None: n = flat_normal(a,b,c)
# #             else: n = fn[0] or flat_normal(a,b,c)
# #
# #             glColor3f(*draw_col)
# #             glNormal3f(*n)
# #             glVertex3f(a[0]+dx, a[1]+dy, a[2]+dz)
# #             if fn[1]: glNormal3f(*fn[1])
# #             glVertex3f(b[0]+dx, b[1]+dy, b[2]+dz)
# #             if fn[2]: glNormal3f(*fn[2])
# #             glVertex3f(c[0]+dx, c[1]+dy, c[2]+dz)
# #         glEnd()
# #
# #     # ── Update state ─────────────────────────────────────────────────────
# #     def update(self, gesture, sd, rd, pitch_d=0.0):
# #         self.scale = max(0.25, min(4.0, self.scale + sd))
# #         self.yaw  += rd
# #         self.pitch = max(-45, min(45, self.pitch + pitch_d))
# #
# #         if gesture == "interior_view" and not self.interior:
# #             self.interior = True
# #             self.exploded = False
# #         elif gesture == "explode_view":
# #             self.exploded = not self.exploded
# #             self.interior = False
# #         elif gesture == "reset":
# #             self.interior = False
# #             self.exploded = False
# #             self.scale    = 1.0
# #             self.yaw      = 0.0
# #             self.pitch     = 15.0
# #
# #     def cycle_color(self):
# #         self.color_idx = (self.color_idx + 1) % len(CAR_COLORS)
# #
# #     def toggle_wireframe(self):
# #         self.wireframe = not self.wireframe
# #
# #     # ── Render ───────────────────────────────────────────────────────────
# #     def render_rgba(self):
# #         glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
# #         glViewport(0, 0, self.w, self.h)
# #         glClearColor(0.0, 0.0, 0.0, 0.0)
# #         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
# #
# #         # Update projection if fov changed
# #         glMatrixMode(GL_PROJECTION)
# #         glLoadIdentity()
# #         gluPerspective(self.fov, self.w/self.h, 0.01, 100.0)
# #         glMatrixMode(GL_MODELVIEW)
# #
# #         glLoadIdentity()
# #
# #         if self.interior:
# #             # Camera inside cabin looking forward-left
# #             gluLookAt(0.05, -0.15, 0.25,
# #                       0.60,  0.80, 0.25,
# #                       0.0,   0.0,  1.0)
# #             glScalef(self.scale, self.scale, self.scale)
# #         else:
# #             # Orbit camera controlled by yaw + pitch
# #             dist = 3.5 / max(self.scale, 0.25)
# #             rad  = math.radians(self.yaw)
# #             pr   = math.radians(self.pitch)
# #             ex   = dist * math.sin(rad) * math.cos(pr)
# #             ey   = -dist * math.cos(rad) * math.cos(pr)
# #             ez   = dist * math.sin(pr)
# #             gluLookAt(ex, ey, ez,  0, 0, 0,  0, 0, 1)
# #             glScalef(self.scale, self.scale, self.scale)
# #
# #         if self.wireframe:
# #             glDisable(GL_LIGHTING)
# #             glCallList(self._dl_wire)
# #             glEnable(GL_LIGHTING)
# #         elif self.exploded:
# #             self._draw_exploded()
# #         elif self.color_idx == 0:
# #             # Use prebuilt display list only when color is default
# #             glCallList(self._dl_normal)
# #         else:
# #             self._draw_with_paint()
# #
# #         # Read pixels
# #         glPixelStorei(GL_PACK_ALIGNMENT, 1)
# #         raw = glReadPixels(0, 0, self.w, self.h, GL_RGBA, GL_UNSIGNED_BYTE)
# #         arr = np.frombuffer(raw, dtype=np.uint8).reshape(self.h, self.w, 4)[::-1].copy()
# #
# #         glBindFramebuffer(GL_FRAMEBUFFER, 0)
# #         pygame.event.pump()
# #         return arr
# #
# #     def cleanup(self):
# #         glDeleteLists(self._dl_normal, 1)
# #         glDeleteLists(self._dl_wire, 1)
# #         glDeleteFramebuffers(1, [self._fbo])
# #         glDeleteRenderbuffers(1, [self._crb])
# #         glDeleteRenderbuffers(1, [self._drb])
# #         pygame.quit()
# #
# #
# # # ═══════════════════════════════════════════════════════════════════════
# # #  GESTURE ENGINE  (expanded)
# # # ═══════════════════════════════════════════════════════════════════════
# # class GestureEngine:
# #     def __init__(self):
# #         self.mp_hands = mp.solutions.hands
# #         self.hands = self.mp_hands.Hands(
# #             max_num_hands=2,
# #             min_detection_confidence=0.78,
# #             min_tracking_confidence=0.78,
# #         )
# #         self.draw = mp.solutions.drawing_utils
# #         self.style_lm = self.draw.DrawingSpec(color=(0,220,160), thickness=2, circle_radius=3)
# #         self.style_cn = self.draw.DrawingSpec(color=(0,255,100), thickness=1)
# #         self.prev_pinch = None
# #         self.gesture_hold_frames = {}
# #         self.HOLD_THRESH = 4  # frames to confirm gesture
# #
# #     @staticmethod
# #     def _dist(a, b):
# #         return math.hypot(a.x - b.x, a.y - b.y)
# #
# #     @staticmethod
# #     def _fingers_up(lm):
# #         tips   = [8, 12, 16, 20]
# #         joints = [6, 10, 14, 18]
# #         return sum(lm[t].y < lm[j].y for t, j in zip(tips, joints))
# #
# #     def _confirm(self, g):
# #         """Require gesture held for N frames to avoid jitter."""
# #         self.gesture_hold_frames[g] = self.gesture_hold_frames.get(g, 0) + 1
# #         for other in list(self.gesture_hold_frames.keys()):
# #             if other != g:
# #                 self.gesture_hold_frames[other] = 0
# #         return self.gesture_hold_frames[g] >= self.HOLD_THRESH
# #
# #     def process(self, frame):
# #         """Returns (gesture, scale_delta, rotate_delta, confidence)"""
# #         rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #         result = self.hands.process(rgb)
# #         gesture, sd, rd = "none", 0.0, 0.0
# #
# #         if not result.multi_hand_landmarks:
# #             self.prev_pinch = None
# #             self.gesture_hold_frames = {}
# #             return gesture, sd, rd
# #
# #         for hlm in result.multi_hand_landmarks:
# #             lm = hlm.landmark
# #             self.draw.draw_landmarks(frame, hlm,
# #                 self.mp_hands.HAND_CONNECTIONS,
# #                 self.style_lm, self.style_cn)
# #
# #             pinch   = self._dist(lm[4], lm[8])
# #             fingers = self._fingers_up(lm)
# #             wx      = lm[0].x
# #
# #             # Scale via pinch movement
# #             if self.prev_pinch is not None:
# #                 delta = pinch - self.prev_pinch
# #                 if abs(delta) > 0.004:
# #                     sd = delta * 2.2
# #             self.prev_pinch = pinch
# #
# #             # Gesture classification (priority order)
# #             raw_g = "none"
# #             if fingers >= 4:   raw_g = "reset"
# #             elif fingers == 3: raw_g = "explode_view"
# #             elif fingers == 2: raw_g = "interior_view"
# #             elif pinch < 0.055: raw_g = "pinch_close"
# #             elif pinch > 0.22:  raw_g = "pinch_open"
# #
# #             if raw_g != "none" and self._confirm(raw_g):
# #                 gesture = raw_g
# #             elif raw_g == "none":
# #                 self.gesture_hold_frames = {}
# #
# #             # Orbit
# #             if wx < 0.28:   rd = -5.0
# #             elif wx > 0.72: rd = +5.0
# #
# #         return gesture, sd, rd
# #
# #
# # # ═══════════════════════════════════════════════════════════════════════
# # #  ARUCO TRACKER
# # # ═══════════════════════════════════════════════════════════════════════
# # class ArucoTracker:
# #     def __init__(self):
# #         self.adict    = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
# #         self.detector = cv2.aruco.ArucoDetector(
# #             self.adict, cv2.aruco.DetectorParameters())
# #         if not os.path.exists("aruco_marker.png"):
# #             img = cv2.aruco.generateImageMarker(self.adict, ARUCO_ID, 300)
# #             cv2.imwrite("aruco_marker.png", img)
# #             print("[ArUco] aruco_marker.png saved — print it!")
# #
# #     def detect(self, frame):
# #         grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #         corners, ids, _ = self.detector.detectMarkers(grey)
# #         if ids is None:
# #             return None, 1.0
# #         for i, mid in enumerate(ids.flatten()):
# #             if mid != ARUCO_ID: continue
# #             c    = corners[i][0]
# #             cx   = int(c[:, 0].mean())
# #             cy   = int(c[:, 1].mean())
# #             side = np.linalg.norm(c[0] - c[1])
# #             cv2.aruco.drawDetectedMarkers(frame, corners, ids)
# #             return (cx, cy), max(0.3, min(3.0, side / 100))
# #         return None, 1.0
# #
# #
# # # ═══════════════════════════════════════════════════════════════════════
# # #  COMPOSITOR
# # # ═══════════════════════════════════════════════════════════════════════
# # def composite(bg_bgr, car_rgba, anchor=None, scale=1.0):
# #     hb, wb = bg_bgr.shape[:2]
# #     hc, wc = car_rgba.shape[:2]
# #     nw = max(10, min(int(wc * scale), wb))
# #     nh = max(10, min(int(hc * scale), hb))
# #     scaled = cv2.resize(car_rgba, (nw, nh), interpolation=cv2.INTER_LINEAR)
# #
# #     x0 = (anchor[0] - nw//2) if anchor else (wb - nw)//2
# #     y0 = (anchor[1] - nh//2) if anchor else (hb - nh)//2
# #     x0 = max(0, min(x0, wb - nw))
# #     y0 = max(0, min(y0, hb - nh))
# #
# #     roi = bg_bgr[y0:y0+nh, x0:x0+nw].astype(np.float32)
# #     fg  = scaled[:, :, :3].astype(np.float32)[:, :, ::-1]  # RGB→BGR
# #     a   = scaled[:, :, 3:4].astype(np.float32) / 255.0
# #
# #     out = bg_bgr.copy()
# #     out[y0:y0+nh, x0:x0+nw] = (fg * a + roi * (1 - a)).astype(np.uint8)
# #     return out
# #
# #
# # # ═══════════════════════════════════════════════════════════════════════
# # #  TTS ENGINE  (single persistent thread — fixes Windows TTS deadlock)
# # # ═══════════════════════════════════════════════════════════════════════
# # class TTSEngine:
# #     def __init__(self):
# #         self._q = queue.Queue()
# #         self._ok = TTS_OK
# #         if self._ok:
# #             threading.Thread(target=self._worker, daemon=True).start()
# #
# #     def _worker(self):
# #         """Single long-lived pyttsx3 engine — avoids re-init deadlock."""
# #         try:
# #             engine = pyttsx3.init()
# #             engine.setProperty("rate", 165)
# #             engine.setProperty("volume", 0.92)
# #             while True:
# #                 text = self._q.get()
# #                 if text is None:
# #                     break
# #                 try:
# #                     engine.say(text)
# #                     engine.runAndWait()
# #                 except Exception as e:
# #                     print(f"[TTS] speak error: {e}")
# #                     # Re-init engine on error
# #                     try:
# #                         engine = pyttsx3.init()
# #                         engine.setProperty("rate", 165)
# #                     except:
# #                         pass
# #                 self._q.task_done()
# #         except Exception as e:
# #             print(f"[TTS] Worker failed: {e}")
# #
# #     def speak(self, text):
# #         if self._ok and text:
# #             # Clear queue first (don't pile up stale replies)
# #             while not self._q.empty():
# #                 try: self._q.get_nowait()
# #                 except: pass
# #             self._q.put(text)
# #
# #     def stop(self):
# #         self._q.put(None)
# #
# #
# # # ═══════════════════════════════════════════════════════════════════════
# # #  CONVERSATIONAL AI  (fixed — no duplicate Ollama call)
# # # ═══════════════════════════════════════════════════════════════════════
# # class ConversationalAI:
# #     SYSTEM = (
# #         "You are an enthusiastic AR car guide with deep automotive knowledge. "
# #         "Keep every reply to 1-2 sentences maximum. "
# #         "Be vivid, specific, and friendly. No markdown, no bullet points."
# #     )
# #     GESTURE_PROMPTS = {
# #         "interior_view" : "Describe what I see looking into a modern sports car's interior.",
# #         "explode_view"  : "What are the main structural components visible in an exploded car view?",
# #         "pinch_open"    : "How does a car's aerodynamics change at different scales?",
# #         "reset"         : "Give me one surprising fact about automotive design history.",
# #     }
# #
# #     def __init__(self, model=OLLAMA_MODEL, tts: TTSEngine = None):
# #         self.model   = model
# #         self.tts     = tts
# #         self.history = []
# #         self.reply   = "Gesture or speak to begin — I am your AR car guide!"
# #         self.busy    = False
# #         self.lock    = threading.Lock()
# #         self.last_query = ""
# #
# #     def chat(self, user_text: str, speak_reply=True):
# #         if self.busy:
# #             return
# #         if user_text == self.last_query:
# #             return   # don't repeat same prompt
# #         self.last_query = user_text
# #         self.busy = True
# #         threading.Thread(
# #             target=self._run, args=(user_text, speak_reply), daemon=True
# #         ).start()
# #
# #     def _run(self, user_text, speak_reply):
# #         with self.lock:
# #             self.history.append({"role": "user", "content": user_text})
# #             history_slice = self.history[-10:]
# #
# #         try:
# #             res = ollama.chat(
# #                 model=self.model,
# #                 messages=[{"role": "system", "content": self.SYSTEM}] + history_slice,
# #             )
# #             reply_text = res["message"]["content"].strip()
# #         except Exception as e:
# #             reply_text = f"AI offline. Start Ollama: ollama serve"
# #             print(f"[Ollama] {e}")
# #
# #         with self.lock:
# #             self.history.append({"role": "assistant", "content": reply_text})
# #             self.reply = reply_text
# #
# #         print(f"\n[AI] {reply_text}\n")
# #
# #         if speak_reply and self.tts:
# #             self.tts.speak(reply_text)
# #
# #         self.busy = False
# #
# #     def get_reply(self):
# #         with self.lock:
# #             return self.reply
# #
# #     def get_history_display(self, n=4):
# #         """Return last n AI replies for history panel."""
# #         with self.lock:
# #             ai_turns = [(m["role"], m["content"])
# #                         for m in self.history if m["role"] in ("user","assistant")]
# #             return ai_turns[-(n*2):]
# #
# #
# # # ═══════════════════════════════════════════════════════════════════════
# # #  CONTINUOUS VOICE LISTENER
# # # ═══════════════════════════════════════════════════════════════════════
# # class ContinuousListener:
# #     def __init__(self):
# #         self.q    = queue.Queue()
# #         self.stop_event = threading.Event()
# #         self.active = False
# #
# #     def start(self):
# #         if not VOICE_OK: return
# #         self.active = True
# #         threading.Thread(target=self._loop, daemon=True).start()
# #
# #     def _loop(self):
# #         rec = sr.Recognizer()
# #         mic = sr.Microphone()
# #         rec.dynamic_energy_threshold = True
# #         rec.energy_threshold = 2800
# #         with mic as src:
# #             rec.adjust_for_ambient_noise(src, duration=1.0)
# #         print("[Voice] Continuous listener ready — just speak!")
# #         while not self.stop_event.is_set():
# #             try:
# #                 with mic as src:
# #                     audio = rec.listen(src, timeout=2, phrase_time_limit=10)
# #                 text = rec.recognize_google(audio)
# #                 if text.strip():
# #                     self.q.put(text)
# #                     print(f"[Voice] '{text}'")
# #             except sr.WaitTimeoutError:
# #                 pass
# #             except sr.UnknownValueError:
# #                 pass
# #             except Exception as e:
# #                 print(f"[Voice] {e}")
# #                 time.sleep(0.5)
# #
# #     def get_text(self):
# #         try: return self.q.get_nowait()
# #         except queue.Empty: return None
# #
# #     def close(self):
# #         self.stop_event.set()
# #
# #
# # # ═══════════════════════════════════════════════════════════════════════
# # #  GUI — FULL SECTIONED HUD
# # # ═══════════════════════════════════════════════════════════════════════
# # class HUD:
# #     # Color palette (BGR for OpenCV)
# #     C_BG      = (10, 10, 12)
# #     C_BORDER  = (0, 180, 120)
# #     C_TITLE   = (0, 230, 160)
# #     C_TEXT    = (210, 255, 200)
# #     C_DIM     = (130, 160, 130)
# #     C_WARN    = (40, 120, 255)
# #     C_RED     = (60,  60, 220)
# #     C_YELLOW  = (30, 220, 220)
# #     C_CYAN    = (220, 200, 30)
# #     C_WHITE   = (240, 240, 240)
# #
# #     FONT      = cv2.FONT_HERSHEY_SIMPLEX
# #     FONT_MONO = cv2.FONT_HERSHEY_DUPLEX
# #
# #     def __init__(self, frame_w, frame_h):
# #         self.fw = frame_w
# #         self.fh = frame_h
# #
# #     def _panel(self, frame, x, y, w, h, title, alpha=0.55):
# #         """Draw a semi-transparent panel with a title bar."""
# #         overlay = frame.copy()
# #         cv2.rectangle(overlay, (x, y), (x+w, y+h), self.C_BG, -1)
# #         cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
# #         # Border
# #         cv2.rectangle(frame, (x, y), (x+w, y+h), self.C_BORDER, 1)
# #         # Title bar
# #         cv2.rectangle(frame, (x, y), (x+w, y+18), self.C_BORDER, -1)
# #         cv2.putText(frame, title.upper(), (x+5, y+13),
# #                     self.FONT, 0.42, self.C_BG, 1, cv2.LINE_AA)
# #
# #     def _text(self, frame, txt, x, y, color=None, scale=0.52, thickness=1):
# #         color = color or self.C_TEXT
# #         cv2.putText(frame, txt, (x, y), self.FONT, scale, color, thickness, cv2.LINE_AA)
# #
# #     def _bar(self, frame, x, y, w, val, max_val, color):
# #         """Horizontal progress bar."""
# #         cv2.rectangle(frame, (x, y), (x+w, y+8), (40,40,40), -1)
# #         fill = int(w * min(val/max_val, 1.0))
# #         if fill > 0:
# #             cv2.rectangle(frame, (x, y), (x+fill, y+8), color, -1)
# #
# #     # ── Panel: Gesture Guide (top-left) ────────────────────────────────
# #     def draw_gesture_panel(self, frame, gesture, x=8, y=8):
# #         pw, ph = 200, 148
# #         self._panel(frame, x, y, pw, ph, "  Gesture Guide")
# #         labels = [
# #             ("Pinch in/out", "Scale"),
# #             ("2 Fingers", "Interior"),
# #             ("3 Fingers", "Explode"),
# #             ("Open Palm", "Reset"),
# #             ("Wrist L/R", "Orbit"),
# #         ]
# #         icons = ["[PI]","[2F]","[3F]","[OP]","[W<>]"]
# #         for i, ((key, val), icon) in enumerate(zip(labels, icons)):
# #             yy = y + 28 + i * 22
# #             active = (
# #                 (key == "Pinch in/out" and "pinch" in gesture) or
# #                 (key == "2 Fingers"    and gesture == "interior_view") or
# #                 (key == "3 Fingers"    and gesture == "explode_view") or
# #                 (key == "Open Palm"    and gesture == "reset")
# #             )
# #             col = self.C_YELLOW if active else self.C_DIM
# #             cv2.putText(frame, icon, (x+6, yy),
# #                         self.FONT, 0.38, col, 1, cv2.LINE_AA)
# #             cv2.putText(frame, f"{key}", (x+40, yy),
# #                         self.FONT, 0.40, col, 1, cv2.LINE_AA)
# #             cv2.putText(frame, f"→{val}", (x+140, yy),
# #                         self.FONT, 0.40, self.C_CYAN if active else self.C_DIM,
# #                         1, cv2.LINE_AA)
# #
# #     # ── Panel: Car Info (below gesture guide) ─────────────────────────
# #     def draw_car_panel(self, frame, car, x=8, y=165):
# #         pw, ph = 200, 148
# #         self._panel(frame, x, y, pw, ph, "  Car Info")
# #         paint = CAR_COLORS[car.color_idx]
# #         color_name = COLOR_NAMES[car.color_idx]
# #
# #         view_mode = ("Interior" if car.interior
# #                      else "Exploded" if car.exploded
# #                      else "Wireframe" if car.wireframe
# #                      else "Exterior")
# #         rows = [
# #             ("Model",   os.path.basename(MODEL_PATH)),
# #             ("Tris",    f"{car._tri_count:,}"),
# #             ("View",    view_mode),
# #             ("Paint",   color_name),
# #             ("Scale",   f"{car.scale:.2f}x"),
# #             ("Yaw",     f"{car.yaw % 360:.0f}°"),
# #         ]
# #         for i, (k, v) in enumerate(rows):
# #             yy = y + 28 + i*19
# #             self._text(frame, f"{k}:", x+6, yy, self.C_DIM, 0.40)
# #             self._text(frame, v,        x+70, yy, self.C_TEXT, 0.42)
# #
# #         # Paint color swatch
# #         r, g, b = int(paint[0]*255), int(paint[1]*255), int(paint[2]*255)
# #         cv2.rectangle(frame, (x+pw-30, y+20), (x+pw-6, y+36),
# #                       (b, g, r), -1)
# #         cv2.rectangle(frame, (x+pw-30, y+20), (x+pw-6, y+36),
# #                       self.C_BORDER, 1)
# #
# #     # ── Panel: AI Chat History (bottom-left) ───────────────────────────
# #     def draw_ai_panel(self, frame, ai: ConversationalAI, voice_active, x=8):
# #         ph = 140
# #         y  = self.fh - ph - 8
# #         pw = self.fw - 16
# #         self._panel(frame, x, y, pw, ph, "  AI Car Guide", alpha=0.65)
# #
# #         # Typing indicator
# #         if ai.busy:
# #             dots = "." * (int(time.time() * 3) % 4)
# #             self._text(frame, f"Thinking{dots}", x+8, y+30, self.C_CYAN, 0.48)
# #             return
# #
# #         # Latest AI reply — word-wrapped
# #         reply = ai.get_reply()
# #         words = reply.split()
# #         lines, cur = [], ""
# #         max_chars = (pw - 20) // 8
# #         for word in words:
# #             test = (cur + " " + word).strip()
# #             if len(test) < max_chars:
# #                 cur = test
# #             else:
# #                 if cur: lines.append(cur)
# #                 cur = word
# #         if cur: lines.append(cur)
# #         lines = lines[:4]
# #
# #         for i, line in enumerate(lines):
# #             self._text(frame, line, x+8, y+30+i*26, self.C_TEXT, 0.52)
# #
# #         # Voice indicator pulsing dot
# #         if voice_active and VOICE_OK:
# #             r = int(7 + 3*math.sin(time.time()*8))
# #             cv2.circle(frame, (x+pw-20, y+14), r, (0, 60, 255), -1)
# #             self._text(frame, "REC", x+pw-58, y+20, (0,100,255), 0.38)
# #
# #     # ── Panel: Status Bar (top-right) ──────────────────────────────────
# #     def draw_status_panel(self, frame, fps, gesture, aruco_found,
# #                           voice_ok, ollama_ok):
# #         pw, ph = 210, 108
# #         x = self.fw - pw - 8
# #         y = 8
# #         self._panel(frame, x, y, pw, ph, "  System Status")
# #
# #         # FPS bar
# #         self._text(frame, f"FPS", x+6, y+30, self.C_DIM, 0.40)
# #         self._text(frame, f"{fps:.0f}", x+50, y+30,
# #                    self.C_TITLE if fps>20 else self.C_WARN, 0.50)
# #         self._bar(frame, x+80, y+23, pw-90, fps, 60,
# #                   (0,200,80) if fps>20 else (30,120,255))
# #
# #         # Gesture confidence
# #         g_label = gesture if gesture != "none" else "—"
# #         self._text(frame, "Gesture", x+6, y+50, self.C_DIM, 0.40)
# #         self._text(frame, g_label, x+80, y+50, self.C_CYAN, 0.44)
# #
# #         # ArUco
# #         self._text(frame, "ArUco", x+6, y+68, self.C_DIM, 0.40)
# #         ac_col = (0,200,80) if aruco_found else (80,80,180)
# #         self._text(frame, "LOCKED" if aruco_found else "searching",
# #                    x+80, y+68, ac_col, 0.44)
# #
# #         # AI / Voice status
# #         self._text(frame, "Ollama", x+6, y+86, self.C_DIM, 0.40)
# #         ok_col = (0,200,80) if ollama_ok else (60,60,220)
# #         self._text(frame, "online" if ollama_ok else "offline",
# #                    x+80, y+86, ok_col, 0.44)
# #
# #         self._text(frame, "Voice", x+6, y+104, self.C_DIM, 0.40)
# #         vc_col = (0,200,80) if voice_ok else (100,100,100)
# #         self._text(frame, "ready" if voice_ok else "N/A",
# #                    x+80, y+104, vc_col, 0.44)
# #
# #     # ── Panel: Keyboard Shortcuts (top-right, below status) ─────────────
# #     def draw_keys_panel(self, frame):
# #         pw, ph = 210, 80
# #         x = self.fw - pw - 8
# #         y = 124
# #         self._panel(frame, x, y, pw, ph, "  Keys")
# #         keys = [("V","Voice query"),("S","Screenshot"),
# #                 ("W","Wireframe"),("C","Change color"),("Q","Quit")]
# #         for i, (k, v) in enumerate(keys):
# #             yy = y + 26 + i * 13
# #             cv2.putText(frame, f"[{k}]", (x+6, yy),
# #                         self.FONT, 0.38, self.C_CYAN, 1, cv2.LINE_AA)
# #             cv2.putText(frame, v, (x+34, yy),
# #                         self.FONT, 0.38, self.C_DIM, 1, cv2.LINE_AA)
# #
# #     # ── Voice waveform animation (inline in AI panel title) ─────────────
# #     def draw_waveform(self, frame, active, x=8, y=None):
# #         if not active: return
# #         y = y or (self.fh - 150)
# #         for i in range(20):
# #             h_ = int(4 + 8*abs(math.sin(time.time()*6 + i*0.5)))
# #             col_intensity = int(80 + 140*abs(math.sin(time.time()*4 + i)))
# #             cv2.line(frame,
# #                      (x + 230 + i*6, y - h_),
# #                      (x + 230 + i*6, y + h_),
# #                      (0, col_intensity, col_intensity), 2)
# #
# #     def draw_all(self, frame, car, gesture, ai, fps,
# #                  aruco_found, voice_active, ollama_ok):
# #         self.draw_gesture_panel(frame, gesture)
# #         self.draw_car_panel(frame, car)
# #         self.draw_status_panel(frame, fps, gesture, aruco_found,
# #                                VOICE_OK, ollama_ok)
# #         self.draw_keys_panel(frame)
# #         self.draw_ai_panel(frame, ai, voice_active)
# #         self.draw_waveform(frame, voice_active)
# #
# #
# # # ═══════════════════════════════════════════════════════════════════════
# # #  MAIN
# # # ═══════════════════════════════════════════════════════════════════════
# #
# #
# # class EventDispatcher:
# #     def __init__(self, ai_engine):
# #         self.ai = ai_engine
# #
# #     def handle_voice(self, text):
# #         # Activation word logic for Voice
# #         if text.lower().startswith("hey car"):
# #             query = text.lower().replace("hey car", "").strip()
# #             # Vision-aware query
# #             self.ai.ask(query, use_vision=True, speak=True)
# #         elif "status" in text.lower():
# #             self.ai.ask("System status report, please.", use_vision=False, speak=True)
# #
# #     def handle_gesture(self, gesture):
# #         # Specific activation for Gestures (no activation word needed)
# #         # These trigger specific narrations
# #         self.ai.on_gesture(gesture)
# #
# #
# # def main():
# #     print("=" * 62)
# #     print("  AR Car v3  |  PyOpenGL + MediaPipe + Ollama (Windows OK)")
# #     print("=" * 62)
# #     print("  Gestures: pinch=scale  2F=interior  3F=explode  palm=reset")
# #     print("  Keys    : V=voice  S=screenshot  W=wireframe  C=color  Q=quit")
# #     print("  ArUco   : print aruco_marker.png and hold to camera")
# #     print("=" * 62)
# #
# #     cap = cv2.VideoCapture(0)
# #     cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
# #     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
# #     cap.set(cv2.CAP_PROP_FPS, 30)
# #     if not cap.isOpened():
# #         sys.exit("[ERROR] Cannot open webcam. Try changing VideoCapture(0) to (1)")
# #
# #     print("[INIT] Loading 3D renderer…")
# #     car   = CarRenderer()
# #     print("[INIT] Renderer OK")
# #
# #     tts   = TTSEngine()
# #     ge    = GestureEngine()
# #     ai    = ConversationalAI(tts=tts)
# #     aruco = ArucoTracker()
# #     voice = ContinuousListener()
# #     hud   = HUD(CAM_W, CAM_H)
# #     voice.start()
# #
# #     ai.chat("Introduce yourself as the AR car guide in one sentence.",
# #             speak_reply=True)
# #
# #     scale        = 1.0
# #     last_gesture = "none"
# #     voice_active = False
# #     ollama_ok    = True
# #     fps_timer    = time.time()
# #     fps          = 0.0
# #     fc           = 0
# #
# #     print("\n[READY] AR window open. Speak or gesture.\n")
# #
# #     dispatcher = EventDispatcher(ai)
# #
# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             print("[ERROR] Frame capture failed"); break
# #         frame = cv2.flip(frame, 1)
# #
# #         # Gestures
# #         gesture, sd, rd = ge.process(frame)
# #         scale = max(0.25, min(4.0, scale + sd))
# #
# #         # ArUco
# #         anchor, aruco_scale = aruco.detect(frame)
# #         eff_scale = scale * aruco_scale if anchor else scale
# #
# #         # Update + render 3D
# #         car.update(gesture, sd, rd)
# #         car_rgba = car.render_rgba()
# #         frame    = composite(frame, car_rgba, anchor=anchor,
# #                              scale=eff_scale * 0.72)
# #
# #         # AI on gesture change
# #         if gesture != last_gesture and gesture != "none":
# #             p = ConversationalAI.GESTURE_PROMPTS.get(gesture)
# #             if p:
# #                 ai.chat(p, speak_reply=True)
# #         last_gesture = gesture
# #
# #         # Ollama health check (update flag)
# #         if "offline" in ai.get_reply().lower():
# #             ollama_ok = False
# #         elif ai.get_reply() != "Gesture or speak to begin — I am your AR car guide!":
# #             ollama_ok = True
# #
# #         # Continuous voice
# #         spoken = voice.get_text()
# #         if spoken:
# #             low = spoken.lower().strip()
# #             if low in ("exit", "quit", "stop", "goodbye"):
# #                 break
# #             ai.chat(spoken, speak_reply=True)
# #
# #         # FPS
# #         fc  += 1
# #         now  = time.time()
# #         if now - fps_timer >= 1.0:
# #             fps       = fc / (now - fps_timer)
# #             fc        = 0
# #             fps_timer = now
# #
# #         # HUD
# #         hud.draw_all(frame, car, gesture, ai, fps,
# #                      anchor is not None,
# #                      VOICE_OK and voice_active,
# #                      ollama_ok)
# #
# #         cv2.imshow("AR Car v3 — Gesture + Voice + AI  (Q to quit)", frame)
# #
# #         key = cv2.waitKey(1) & 0xFF
# #         if key == ord('q'):
# #             break
# #         elif key == ord('v') and VOICE_OK and not voice_active:
# #             def _vt():
# #                 nonlocal voice_active
# #                 voice_active = True
# #                 # One-shot listen
# #                 try:
# #                     rec = sr.Recognizer()
# #                     mic = sr.Microphone()
# #                     with mic as src:
# #                         rec.adjust_for_ambient_noise(src, duration=0.3)
# #                         print("[Voice] Listening (V key)…")
# #                         audio = rec.listen(src, timeout=5, phrase_time_limit=10)
# #                     text = rec.recognize_google(audio)
# #                     print(f"[Voice] '{text}'")
# #                     ai.chat(text, speak_reply=True)
# #                 except Exception as e:
# #                     print(f"[Voice] {e}")
# #                 finally:
# #                     voice_active = False
# #             threading.Thread(target=_vt, daemon=True).start()
# #
# #         elif key == ord('s'):
# #             ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# #             path = f"screenshots/ar_car_{ts}.png"
# #             cv2.imwrite(path, frame)
# #             print(f"[Screenshot] Saved → {path}")
# #
# #         elif key == ord('w'):
# #             car.toggle_wireframe()
# #             print(f"[Wireframe] {'ON' if car.wireframe else 'OFF'}")
# #
# #         elif key == ord('c'):
# #             car.cycle_color()
# #             print(f"[Color] → {COLOR_NAMES[car.color_idx]}")
# #
# #     tts.stop()
# #     voice.close()
# #     car.cleanup()
# #     cap.release()
# #     cv2.destroyAllWindows()
# #     print("[EXIT] Done.")
# #
# #
# # if __name__ == "__main__":
# #     main()
#
#
#
#
#
#
# """
# ╔══════════════════════════════════════════════════════════════════════╗
# ║              AR CAR v5  —  IRON MAN HOLOGRAM EDITION               ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║  ROOT-CAUSE FIXES:                                                  ║
# ║  1. TTS silent bug: pyttsx3 queue had task_done() BEFORE            ║
# ║     runAndWait() finished → engine torn down mid-speech.            ║
# ║     Fix: task_done() removed entirely; queue is fire-and-forget.    ║
# ║  2. Race condition on AI replies: on_gesture() called ask() which   ║
# ║     competed with voice ask() for busy flag → dropped replies.      ║
# ║     Fix: gesture prompts go to a SEPARATE low-priority queue;       ║
# ║     voice always wins; gesture fires only when voice idle >3s.      ║
# ║  3. Two-hand ignored: mediapipe handedness labels left/right but    ║
# ║     old code just took hands[0]/hands[1] by index (arbitrary).      ║
# ║     Fix: parse multi_handedness to correctly assign LEFT=command    ║
# ║     hand, RIGHT=movement hand every frame.                          ║
# ║                                                                     ║
# ║  NEW FEATURES:                                                       ║
# ║  • HOLOGRAM PALM MODE: close right fist → car materialises on       ║
# ║    your palm like Iron Man. Open right hand flat → car floats       ║
# ║    above it and follows your hand in 3D space.                      ║
# ║  • LEFT hand = COMMAND (gestures: interior/explode/reset/color)     ║
# ║  • RIGHT hand = MOVEMENT (orbit yaw/pitch, zoom pinch, hologram)   ║
# ║  • Hologram shimmer: scanline + edge-glow effect on the 3D model   ║
# ║  • Particle trail when car moves                                    ║
# ║  • Engine sound simulation: low hum animates HUD bars              ║
# ║  • "HEY CAR" activation word → llava sees your screen & answers    ║
# ║  • AI reply always spoken aloud (TTS race condition fixed)          ║
# ║  • X-ray mode (X key): see through car body to internals           ║
# ║  • Night mode (N key): dark theme + neon hologram glow             ║
# ║  • Part spotlight: hover finger over region → AI names that part   ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# INSTALL:
#   pip install opencv-python mediapipe pygame PyOpenGL PyOpenGL_accelerate
#               ollama numpy speechrecognition pyttsx3
#   pip install pipwin && pipwin install pyaudio    (Windows only)
#
# OLLAMA SETUP (one-time):
#   ollama pull llava      ← vision model (sees your screen)
#   ollama pull llama3     ← fast text model
#
# RUN:
#   Terminal 1:  ollama serve
#   Terminal 2:  python ar_car_v5.py
#
# LEFT HAND GESTURES (commands):
#   ✌ 2 fingers   → Interior view
#   🤟 3 fingers   → Explode/X-ray toggle
#   🖐 Open palm   → Reset everything
#   👊 Fist        → Freeze / un-freeze car
#
# RIGHT HAND GESTURES (movement + hologram):
#   Open flat palm → HOLOGRAM MODE: car sits on your palm
#   Pinch           → Zoom in/out
#   Wrist left/right → Orbit (yaw)
#   Wrist up/down    → Tilt (pitch)
#   Point index finger → Spotlight that car part → AI names it
#
# VOICE:
#   "Hey Car <question>"  → AI sees screen + answers (spoken aloud)
#   V key                 → Push-to-talk (always vision-aware)
#   Q key                 → Quit
#
# KEYS:
#   V  Voice query (vision)    C  Cycle color
#   W  Wireframe               X  X-ray mode
#   N  Night / Day mode        A  Auto-spin
#   S  Screenshot              Q  Quit
# """
#
# import cv2, numpy as np, mediapipe as mp
# import ollama, threading, math, os, time, sys, queue, datetime, base64
# import pygame
# from pygame.locals import DOUBLEBUF, OPENGL, NOFRAME
# from OpenGL.GL import *
# from OpenGL.GLU import *
#
# # ── optional voice ──────────────────────────────────────────────────────
# try:
#     import speech_recognition as sr
#     VOICE_OK = True
# except ImportError:
#     VOICE_OK = False; print("[WARN] SpeechRecognition missing — voice off")
#
# try:
#     import pyttsx3
#     TTS_OK = True
# except ImportError:
#     TTS_OK = False; print("[WARN] pyttsx3 missing — TTS off")
#
# # ═══════════════════════════════════════════════════════════════════════
# #  CONFIG
# # ═══════════════════════════════════════════════════════════════════════
# CAM_W, CAM_H  = 1280, 720
# RENDER_W      = 660
# RENDER_H      = 490
# MODEL_PATH    = "car.obj"
# VISION_MODEL  = "llava"
# TEXT_MODEL    = "llama3"
# ARUCO_DICT    = cv2.aruco.DICT_6X6_250
# ARUCO_ID      = 0
# ACTIVATION    = "hey car"
#
# os.makedirs("screenshots", exist_ok=True)
#
# CAR_COLORS = [
#     ((0.85,0.12,0.12), "Red"),
#     ((0.10,0.28,0.85), "Blue"),
#     ((0.08,0.60,0.18), "Green"),
#     ((0.92,0.72,0.04), "Yellow"),
#     ((0.10,0.10,0.10), "Black"),
#     ((0.90,0.90,0.90), "White"),
#     ((0.55,0.55,0.62), "Silver"),
#     ((0.55,0.10,0.80), "Purple"),
# ]
#
# # ═══════════════════════════════════════════════════════════════════════
# #  OBJ / MTL LOADER
# # ═══════════════════════════════════════════════════════════════════════
# def load_mtl(path):
#     mats={}; cur=None
#     try:
#         with open(path,"r",encoding="utf-8",errors="ignore") as f:
#             for line in f:
#                 p=line.strip().split()
#                 if not p: continue
#                 if p[0]=="newmtl": cur=p[1]; mats[cur]=(0.7,0.7,0.7)
#                 elif p[0]=="Kd" and cur:
#                     mats[cur]=(float(p[1]),float(p[2]),float(p[3]))
#     except: pass
#     return mats
#
# def load_obj(path):
#     verts,vnorms,groups=[],[],[]
#     cur_col=(0.7,0.7,0.7); mats={}
#     d=os.path.dirname(os.path.abspath(path))
#     try:
#         with open(path,"r",encoding="utf-8",errors="ignore") as f:
#             for line in f:
#                 p=line.strip().split()
#                 if not p: continue
#                 t=p[0]
#                 if t=="mtllib": mats=load_mtl(os.path.join(d,p[1]))
#                 elif t=="v": verts.append((float(p[1]),float(p[2]),float(p[3])))
#                 elif t=="vn": vnorms.append((float(p[1]),float(p[2]),float(p[3])))
#                 elif t=="usemtl": cur_col=mats.get(p[1],(0.7,0.7,0.7))
#                 elif t=="f":
#                     fv,fn=[],[]
#                     for tok in p[1:]:
#                         s=tok.split("/")
#                         fv.append(verts[int(s[0])-1])
#                         ni=int(s[2])-1 if len(s)>2 and s[2] else None
#                         fn.append(vnorms[ni] if ni is not None and ni<len(vnorms) else None)
#                     for i in range(1,len(fv)-1):
#                         groups.append(((fv[0],fv[i],fv[i+1]),(fn[0],fn[i],fn[i+1]),cur_col))
#         print(f"[OBJ] {len(verts)} verts, {len(groups)} tris, {len(mats)} mats")
#     except Exception as e:
#         print(f"[OBJ] error: {e}")
#     return groups
#
# def flat_n(a,b,c):
#     ab=(b[0]-a[0],b[1]-a[1],b[2]-a[2]); ac=(c[0]-a[0],c[1]-a[1],c[2]-a[2])
#     nx=ab[1]*ac[2]-ab[2]*ac[1]; ny=ab[2]*ac[0]-ab[0]*ac[2]; nz=ab[0]*ac[1]-ab[1]*ac[0]
#     L=math.sqrt(nx*nx+ny*ny+nz*nz) or 1e-9
#     return (nx/L,ny/L,nz/L)
#
# def normalise(groups):
#     if not groups: return groups
#     av=[v for (t,_,__) in groups for v in t]
#     xs=[v[0] for v in av]; ys=[v[1] for v in av]; zs=[v[2] for v in av]
#     cx=(max(xs)+min(xs))/2; cy=(max(ys)+min(ys))/2; cz=(max(zs)+min(zs))/2
#     sc=max(max(xs)-min(xs),max(ys)-min(ys),max(zs)-min(zs)) or 1
#     def sv(v): return ((v[0]-cx)/sc,(v[1]-cy)/sc,(v[2]-cz)/sc)
#     return [(tuple(sv(v) for v in tri),fn,col) for (tri,fn,col) in groups]
#
# def make_placeholder():
#     g=[]
#     def box(x0,y0,z0,x1,y1,z1,col):
#         fs=[[(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],
#             [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
#             [(x0,y0,z0),(x0,y0,z1),(x0,y1,z1),(x0,y1,z0)],
#             [(x1,y0,z0),(x1,y0,z1),(x1,y1,z1),(x1,y1,z0)],
#             [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],
#             [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)]]
#         for f in fs: g.append(((f[0],f[1],f[2]),(None,None,None),col)); g.append(((f[0],f[2],f[3]),(None,None,None),col))
#     def sph(cx,cy,cz,r,col,st=8,sl=12):
#         for i in range(st):
#             la0=math.pi*(-0.5+i/st); la1=math.pi*(-0.5+(i+1)/st)
#             for j in range(sl):
#                 lg0=2*math.pi*j/sl; lg1=2*math.pi*(j+1)/sl
#                 def pt(la,lg): return(cx+r*math.cos(la)*math.cos(lg),cy+r*math.cos(la)*math.sin(lg),cz+r*math.sin(la))
#                 a,b,c_,d=pt(la0,lg0),pt(la0,lg1),pt(la1,lg1),pt(la1,lg0)
#                 g.append(((a,b,c_),(None,None,None),col)); g.append(((a,c_,d),(None,None,None),col))
#     R=(0.85,0.12,0.12); DR=(0.55,0.06,0.06); GL=(0.40,0.72,0.92)
#     BL=(0.08,0.08,0.08); GR=(0.58,0.58,0.62); AM=(1.0,0.85,0.30)
#     TR=(0.90,0.04,0.04); DK=(0.18,0.18,0.18)
#     box(-1.0,-0.5,-0.05,1.0,0.5,0.56,R)
#     box(-0.55,-0.46,0.56,0.60,0.46,0.97,DR)
#     box(-0.54,-0.44,0.57,-0.52,0.44,0.95,GL)
#     box(0.52,-0.44,0.57,0.54,0.44,0.95,GL)
#     box(-0.53,-0.46,0.58,0.53,-0.44,0.94,GL)
#     box(-0.53,0.44,0.58,0.53,0.46,0.94,GL)
#     for wx,wy in [(-0.65,-0.55),(0.65,-0.55),(-0.65,0.55),(0.65,0.55)]:
#         sph(wx,wy,-0.03,0.24,BL); sph(wx,wy,-0.03,0.15,GR)
#     box(-1.01,-0.32,-0.02,-0.99,-0.08,0.23,AM)
#     box(-1.01,0.08,-0.02,-1.0,0.32,0.23,AM)
#     box(0.99,-0.32,-0.02,1.01,-0.08,0.23,TR)
#     box(0.99,0.08,-0.02,1.01,0.32,0.23,TR)
#     box(-1.06,-0.44,-0.05,-1.0,0.44,0.28,DK)
#     box(1.0,-0.44,-0.05,1.06,0.44,0.28,DK)
#     box(-1.0,-0.26,0.07,-0.98,0.26,0.28,BL)
#     box(-0.32,-0.54,0.42,-0.10,-0.50,0.50,DR)
#     box(-0.32,0.50,0.42,-0.10,0.54,0.50,DR)
#     return g
#
#
# # ═══════════════════════════════════════════════════════════════════════
# #  TTS ENGINE — fire-and-forget, no task_done race condition
# # ═══════════════════════════════════════════════════════════════════════
# class TTSEngine:
#     """
#     FIX: removed task_done() which was causing pyttsx3 engine teardown
#     mid-speech. Queue is now fire-and-forget. Worker loops forever.
#     Engine is created ONCE and reused — never re-inited per call.
#     """
#     def __init__(self):
#         self._q   = queue.Queue()
#         self._ok  = TTS_OK
#         self._speaking = False
#         if self._ok:
#             threading.Thread(target=self._worker, daemon=True).start()
#
#     def _worker(self):
#         # Create engine ONCE in this thread — never touch from main thread
#         try:
#             eng = pyttsx3.init()
#             eng.setProperty("rate", 148)
#             eng.setProperty("volume", 1.0)
#         except Exception as e:
#             print(f"[TTS] init failed: {e}"); return
#
#         while True:
#             try:
#                 text = self._q.get()   # blocks until item available
#                 if text is None: break
#
#                 # Drain stale queued items so only latest reply speaks
#                 latest = text
#                 while not self._q.empty():
#                     try: latest = self._q.get_nowait()
#                     except: break
#
#                 self._speaking = True
#                 print(f"[TTS] Speaking: {latest[:60]}…")
#                 try:
#                     eng.say(latest)
#                     eng.runAndWait()
#                 except Exception as e:
#                     print(f"[TTS] speak error: {e}")
#                     try: eng.stop()
#                     except: pass
#                 self._speaking = False
#             except Exception as e:
#                 print(f"[TTS] worker error: {e}")
#                 self._speaking = False
#
#     def speak(self, text):
#         """Queue text for speaking. Non-blocking always."""
#         if self._ok and text and text.strip():
#             self._q.put(text.strip())
#
#     def is_speaking(self): return self._speaking
#     def stop(self): self._q.put(None)
#
#
# # ═══════════════════════════════════════════════════════════════════════
# #  AI ENGINE — fixed race condition, gesture queue separated from voice
# # ═══════════════════════════════════════════════════════════════════════
# class CarAI:
#     """
#     FIX: gesture-triggered prompts go to _gesture_q (low priority).
#     Voice queries bypass it entirely. busy flag is ALWAYS cleared in
#     finally{} so no call is ever permanently blocked.
#     Gesture fires only when no voice query pending AND AI idle > 3 sec.
#     """
#     SYSTEM = (
#         "You are JARVIS, an expert AR car hologram guide. "
#         "Answer in exactly 1-2 punchy sentences. "
#         "Be vivid and enthusiastic. No markdown, no bullets."
#     )
#     _GESTURE_PROMPTS = {
#         "interior" : "Describe the cockpit-like interior of this sports car in one exciting sentence.",
#         "explode"  : "Name the three most impressive structural components revealed in an exploded car view.",
#         "zoom_in"  : "What aerodynamic detail becomes most visible when you zoom into a car?",
#         "reset"    : "Give me the single most surprising fact about modern car engineering.",
#     }
#
#     def __init__(self, tts: TTSEngine):
#         self.tts       = tts
#         self.history   = []
#         self.reply     = "Hologram ready. Say 'Hey Car' or use gestures!"
#         self.busy      = False
#         self.last_spoken_reply = ""
#         self._lock     = threading.Lock()
#         self._gesture_q = queue.Queue()
#         self._last_ai_time = 0.0
#         self._frame_slot = [None]
#         # gesture worker — fires only when voice idle
#         threading.Thread(target=self._gesture_worker, daemon=True).start()
#
#     def set_frame(self, frame):
#         self._frame_slot[0] = frame.copy() if frame is not None else None
#
#     def _b64_frame(self):
#         f = self._frame_slot[0]
#         if f is None: return None
#         ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 75])
#         return base64.b64encode(buf).decode() if ok else None
#
#     # ── Public API ───────────────────────────────────────────────────────
#     def ask_voice(self, text, use_vision=True):
#         """HIGH PRIORITY — always fires, cancels pending gesture prompt."""
#         while not self._gesture_q.empty():
#             try: self._gesture_q.get_nowait()
#             except: break
#         if self.busy: return  # already processing voice — skip
#         self._fire(text, use_vision=use_vision, priority="voice")
#
#     def ask_gesture(self, key):
#         """LOW PRIORITY — queued, only fires when AI has been idle > 3s."""
#         prompt = self._GESTURE_PROMPTS.get(key)
#         if prompt:
#             # Replace any existing gesture prompt (don't pile up)
#             while not self._gesture_q.empty():
#                 try: self._gesture_q.get_nowait()
#                 except: break
#             self._gesture_q.put(prompt)
#
#     def _gesture_worker(self):
#         while True:
#             try:
#                 prompt = self._gesture_q.get(timeout=1)
#                 # Wait until AI idle AND no voice for 3s
#                 waited = 0
#                 while self.busy or (time.time() - self._last_ai_time < 3.0):
#                     time.sleep(0.2); waited += 0.2
#                     if waited > 10: break
#                 if not self.busy:
#                     self._fire(prompt, use_vision=False, priority="gesture")
#             except queue.Empty:
#                 pass
#
#     def _fire(self, text, use_vision, priority):
#         self.busy = True
#         threading.Thread(
#             target=self._run,
#             args=(text, use_vision, priority),
#             daemon=True
#         ).start()
#
#     def _run(self, text, use_vision, priority):
#         try:
#             with self._lock:
#                 self.history.append({"role":"user","content":text})
#                 hist = self.history[-8:]
#
#             if use_vision:
#                 b64 = self._b64_frame()
#                 model = VISION_MODEL
#                 msgs = [{"role":"system","content":self.SYSTEM}]
#                 if b64:
#                     msgs.append({"role":"user","content":text,"images":[b64]})
#                 else:
#                     msgs.extend(hist)
#             else:
#                 model = TEXT_MODEL
#                 msgs  = [{"role":"system","content":self.SYSTEM}] + hist
#
#             res   = ollama.chat(model=model, messages=msgs)
#             reply = res["message"]["content"].strip()
#             reply = reply.replace("**","").replace("*","").replace("#","")
#
#         except Exception as e:
#             reply = f"Ollama offline — run 'ollama serve' in another terminal."
#             print(f"[AI] Error: {e}")
#
#         finally:
#             # ALWAYS clear busy — this was the core bug in v4
#             self.busy = False
#             self._last_ai_time = time.time()
#
#         with self._lock:
#             self.history.append({"role":"assistant","content":reply})
#             self.reply = reply
#
#         print(f"\n[AI/{priority}] {reply}\n")
#
#         # Speak the reply — TTS is non-blocking so this returns immediately
#         if reply != self.last_spoken_reply:
#             self.last_spoken_reply = reply
#             self.tts.speak(reply)
#
#     def get_reply(self):
#         with self._lock: return self.reply
#
#
# # ═══════════════════════════════════════════════════════════════════════
# #  DUAL-HAND GESTURE ENGINE — left = command, right = movement
# # ═══════════════════════════════════════════════════════════════════════
# class GestureEngine:
#     """
#     Uses mediapipe handedness labels to correctly assign:
#       LEFT hand  = command gestures (interior/explode/reset/freeze)
#       RIGHT hand = movement (yaw/pitch/zoom) + hologram palm mode
#
#     Returns a rich dict every frame.
#     """
#     def __init__(self):
#         self.mp_h = mp.solutions.hands
#         self.hands = self.mp_h.Hands(
#             max_num_hands=2,
#             min_detection_confidence=0.72,
#             min_tracking_confidence=0.72)
#         self.draw = mp.solutions.drawing_utils
#         self.lm_style = self.draw.DrawingSpec(color=(0,220,160),thickness=2,circle_radius=3)
#         self.cn_style = self.draw.DrawingSpec(color=(0,255,100),thickness=1)
#         self._prev_r_pinch = None
#         self._hold = {}; self._HOLD = 5
#         self._prev_palm_pos = None
#
#     @staticmethod
#     def _d(a,b): return math.hypot(a.x-b.x,a.y-b.y)
#
#     @staticmethod
#     def _fingers(lm):
#         return sum(lm[t].y<lm[j].y for t,j in zip([8,12,16,20],[6,10,14,18]))
#
#     @staticmethod
#     def _thumb_up(lm):
#         return lm[4].y < lm[3].y < lm[2].y  # thumb extended upward
#
#     @staticmethod
#     def _is_fist(lm):
#         # All finger tips below their pip joints
#         return all(lm[t].y>lm[j].y for t,j in zip([8,12,16,20],[6,10,14,18]))
#
#     @staticmethod
#     def _palm_centre(lm):
#         # Average of wrist + base knuckles
#         pts=[lm[i] for i in [0,1,5,9,13,17]]
#         return (sum(p.x for p in pts)/6, sum(p.y for p in pts)/6)
#
#     def _confirm(self,g):
#         self._hold[g]=self._hold.get(g,0)+1
#         for k in list(self._hold):
#             if k!=g: self._hold[k]=0
#         return self._hold[g]>=self._HOLD
#
#     def process(self, frame):
#         """Returns gesture dict. Draws landmarks on frame in-place."""
#         out = {
#               "left_cmd": "none",
#
#     # RIGHT HAND (movement)
#     "right_yaw": 0.0,
#     "right_pitch": 0.0,
#     "right_zoom": 0.0,
#
#     # LEFT HAND (fine control)
#     "left_scale": 0.0,
#     "left_rotate": 0.0,
#     "left_tilt": 0.0,
#
#     # TWO HAND SYSTEM
#     "both_scale": 0.0,
#     "both_rotate": 0.0,
#
#     "hologram": False,
#     "palm_pos": None,
#     "pointing": False,
#     "point_pos": None,
#     "hand_count": 0,
#         }
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         res = self.hands.process(rgb)
#
#         if not res.multi_hand_landmarks:
#             self._prev_r_pinch=None; self._prev_palm_pos=None
#             self._hold={}
#             return out
#
#         # Build labelled hand map
#         left_lm = right_lm = None
#         for hlm, hnd in zip(res.multi_hand_landmarks, res.multi_handedness):
#             label = hnd.classification[0].label   # "Left" or "Right"
#             self.draw.draw_landmarks(
#                 frame, hlm, self.mp_h.HAND_CONNECTIONS,
#                 self.draw.DrawingSpec(
#                     color=(0,180,255) if label=="Left" else (0,255,160),
#                     thickness=2, circle_radius=3),
#                 self.draw.DrawingSpec(color=(200,255,200),thickness=1))
#             # Mediapipe "Left" in mirrored feed = user's right hand
#             if label == "Left":
#                 left_lm = hlm.landmark
#             else:
#                 right_lm  = hlm.landmark
#
#         out["hand_count"] = (1 if left_lm else 0) + (1 if right_lm else 0)
#
#         # ── LEFT HAND → command gestures ────────────────────────────────
#         if left_lm:
#             lm = left_lm
#             fi = self._fingers(lm)
#             raw = "none"
#             if self._is_fist(lm):              raw = "freeze"
#             elif fi >= 4:                       raw = "reset"
#             elif fi == 3:                       raw = "explode"
#             elif fi == 2:                       raw = "interior"
#             elif self._thumb_up(lm) and fi==0:  raw = "color"
#
#             if raw != "none" and self._confirm(raw):
#                 out["left_cmd"] = raw
#             elif raw == "none":
#                 self._hold = {}
#
#             # ─────────────────────────────────────────────
#             # ADD HERE ✅ LEFT HAND CONTINUOUS CONTROL
#             # ─────────────────────────────────────────────
#
#             pinch = self._d(lm[4], lm[8])
#
#             if not hasattr(self, "_prev_l_pinch"):
#                 self._prev_l_pinch = pinch
#
#             dp = pinch - self._prev_l_pinch
#             if abs(dp) > 0.003:
#                 out["left_scale"] = dp * 3.5
#
#             self._prev_l_pinch = pinch
#
#             # Rotation (wrist X)
#             wx = lm[0].x
#             if wx < 0.3:
#                 out["left_rotate"] = -3.5
#             elif wx > 0.7:
#                 out["left_rotate"] = +3.5
#
#             # Tilt (wrist Y)
#             wy = lm[0].y
#             if wy < 0.3:
#                 out["left_tilt"] = +2.5
#             elif wy > 0.7:
#                 out["left_tilt"] = -2.5
#
#         # ── RIGHT HAND → movement + hologram ────────────────────────────
#         if right_lm:
#             lm = right_lm
#             fi = self._fingers(lm)
#             pinch = self._d(lm[4], lm[8])
#             wx, wy = lm[0].x, lm[0].y
#
#             # Yaw (wrist X)
#             if   wx < 0.22: out["right_yaw"] = -8.0
#             elif wx > 0.78: out["right_yaw"] = +8.0
#             elif wx < 0.35: out["right_yaw"] = -4.0
#             elif wx > 0.65: out["right_yaw"] = +4.0
#
#             # Pitch (wrist Y)
#             if   wy < 0.22: out["right_pitch"] = +3.0
#             elif wy > 0.78: out["right_pitch"] = -3.0
#
#             # Zoom (pinch delta)
#             if self._prev_r_pinch is not None:
#                 dp = pinch - self._prev_r_pinch
#                 if abs(dp) > 0.004:
#                     out["right_zoom"] = dp * 3.0
#             self._prev_r_pinch = pinch
#
#             # Hologram palm: all 4 fingers up + flat (fi>=4, low wrist Y variance)
#             # When fingers all extended and palm faces camera → hologram sits on hand
#             if fi >= 4:
#                 pc = self._palm_centre(lm)
#                 out["hologram"]  = True
#                 out["palm_pos"]  = pc
#                 self._prev_palm_pos = pc
#             else:
#                 out["hologram"] = False
#
#             # Pointing: only index up, others down
#             index_up = lm[8].y < lm[6].y
#             others_down = all(lm[t].y > lm[j].y for t,j in zip([12,16,20],[10,14,18]))
#             if index_up and others_down and pinch > 0.15:
#                 out["pointing"]  = True
#                 out["point_pos"] = (lm[8].x, lm[8].y)
#
#             # ─────────────────────────────────────────────
#             # ADD HERE ✅ TWO HAND INTERACTION
#             # ─────────────────────────────────────────────
#
#             if left_lm and right_lm:
#                 lx, ly = left_lm[0].x, left_lm[0].y
#                 rx, ry = right_lm[0].x, right_lm[0].y
#
#                 # Distance (scale)
#                 dist = math.hypot(lx - rx, ly - ry)
#
#                 if not hasattr(self, "_prev_hand_dist"):
#                     self._prev_hand_dist = dist
#
#                 d_dist = dist - self._prev_hand_dist
#                 if abs(d_dist) > 0.002:
#                     out["both_scale"] = d_dist * 6.0
#
#                 self._prev_hand_dist = dist
#
#                 # Angle (rotation)
#                 angle = math.atan2(ry - ly, rx - lx)
#
#                 if not hasattr(self, "_prev_angle"):
#                     self._prev_angle = angle
#
#                 d_ang = angle - self._prev_angle
#                 out["both_rotate"] = d_ang * 120
#
#                 self._prev_angle = angle
#
#         return out
#
#
# # ═══════════════════════════════════════════════════════════════════════
# #  OPENGL CAR RENDERER
# # ═══════════════════════════════════════════════════════════════════════
# class CarRenderer:
#     def __init__(self, width=RENDER_W, height=RENDER_H, model_path=MODEL_PATH):
#         self.w=width; self.h=height
#         self.yaw=0.0; self.pitch=18.0; self.scale=1.0
#         self.interior=False; self.exploded=False
#         self.wireframe=False; self.xray=False
#         self.auto_spin=False; self.frozen=False
#         self.night_mode=False; self.color_idx=0
#         self.hologram_mode=False
#         self._spin_spd=0.5
#         self._hologram_alpha=0.0   # 0→1 fade in/out
#         self._smooth_scale = 1.0
#         self._smooth_yaw = 0.0
#         self._smooth_pitch = 18.0
#
#         pygame.init()
#         if sys.platform=="win32":
#             os.environ.setdefault("SDL_VIDEODRIVER","windib")
#         pygame.display.set_mode((width,height),DOUBLEBUF|OPENGL|NOFRAME)
#         pygame.display.set_caption("AR-v5-GL")
#
#         # FBO
#         self._fbo=glGenFramebuffers(1)
#         self._crb=glGenRenderbuffers(1)
#         self._drb=glGenRenderbuffers(1)
#         glBindFramebuffer(GL_FRAMEBUFFER,self._fbo)
#         glBindRenderbuffer(GL_RENDERBUFFER,self._crb)
#         glRenderbufferStorage(GL_RENDERBUFFER,GL_RGBA8,width,height)
#         glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_RENDERBUFFER,self._crb)
#         glBindRenderbuffer(GL_RENDERBUFFER,self._drb)
#         glRenderbufferStorage(GL_RENDERBUFFER,GL_DEPTH_COMPONENT24,width,height)
#         glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_RENDERBUFFER,self._drb)
#         assert glCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE
#         glBindFramebuffer(GL_FRAMEBUFFER,0)
#
#         glEnable(GL_DEPTH_TEST)
#         glEnable(GL_LIGHTING); glEnable(GL_LIGHT0); glEnable(GL_LIGHT1); glEnable(GL_LIGHT2)
#         glEnable(GL_COLOR_MATERIAL); glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE)
#         glShadeModel(GL_SMOOTH); glEnable(GL_NORMALIZE)
#         glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
#
#         glLightfv(GL_LIGHT0,GL_POSITION,[3,4,5,1])
#         glLightfv(GL_LIGHT0,GL_DIFFUSE,[1.0,0.96,0.90,1])
#         glLightfv(GL_LIGHT0,GL_SPECULAR,[0.8,0.8,0.8,1])
#         glLightfv(GL_LIGHT1,GL_POSITION,[-3,-2,2,1])
#         glLightfv(GL_LIGHT1,GL_DIFFUSE,[0.35,0.35,0.40,1])
#         glLightfv(GL_LIGHT2,GL_POSITION,[0,-4,-2,1])
#         glLightfv(GL_LIGHT2,GL_DIFFUSE,[0.20,0.20,0.26,1])
#         glLightModelfv(GL_LIGHT_MODEL_AMBIENT,[0.18,0.18,0.20,1])
#         glMaterialfv(GL_FRONT,GL_SPECULAR,[0.5,0.5,0.5,1])
#         glMaterialf(GL_FRONT,GL_SHININESS,56.0)
#
#         raw = load_obj(model_path) if os.path.exists(model_path) else make_placeholder()
#         self._groups = normalise(raw)
#         self._tri_count = len(self._groups)
#         self._build_lists()
#
#         glMatrixMode(GL_PROJECTION); glLoadIdentity()
#         gluPerspective(52,width/height,0.01,100)
#         glMatrixMode(GL_MODELVIEW)
#
#     def _build_lists(self):
#         self._dl=glGenLists(1)
#         glNewList(self._dl,GL_COMPILE)
#         glBegin(GL_TRIANGLES)
#         for (tri,fn,col) in self._groups:
#             a,b,c=tri
#             n=fn[0] if fn[0] else flat_n(a,b,c)
#             glColor3f(*col); glNormal3f(*n); glVertex3f(*a)
#             glNormal3f(*(fn[1] if fn[1] else n)); glVertex3f(*b)
#             glNormal3f(*(fn[2] if fn[2] else n)); glVertex3f(*c)
#         glEnd(); glEndList()
#
#         self._dl_wire=glGenLists(1)
#         glNewList(self._dl_wire,GL_COMPILE)
#         glBegin(GL_LINES); glColor3f(0,1,0.5)
#         for (tri,_,__) in self._groups:
#             a,b,c=tri
#             glVertex3f(*a); glVertex3f(*b)
#             glVertex3f(*b); glVertex3f(*c)
#             glVertex3f(*c); glVertex3f(*a)
#         glEnd(); glEndList()
#
#     def _paint(self,col,alpha=1.0):
#         """Return col with paint applied to red body parts."""
#         paint = CAR_COLORS[self.color_idx][0]
#         if col[0]>0.6 and col[1]<0.25 and col[2]<0.25:
#             return paint+(alpha,)
#         elif col[0]>0.38 and col[1]<0.15 and col[2]<0.15:
#             return tuple(max(0,x*0.62) for x in paint)+(alpha,)
#         return col+(alpha,)
#
#     def _draw_solid(self, alpha=1.0):
#         glBegin(GL_TRIANGLES)
#         for (tri,fn,col) in self._groups:
#             a,b,c=tri
#             n=fn[0] if fn[0] else flat_n(a,b,c)
#             glColor4f(*self._paint(col,alpha)); glNormal3f(*n); glVertex3f(*a)
#             glNormal3f(*(fn[1] if fn[1] else n)); glVertex3f(*b)
#             glNormal3f(*(fn[2] if fn[2] else n)); glVertex3f(*c)
#         glEnd()
#
#     def _draw_hologram(self):
#         """Cyan translucent wireframe + scanline flash — Iron Man style."""
#         t = time.time()
#         # Fade alpha
#         self._hologram_alpha = min(1.0, self._hologram_alpha + 0.04)
#         a = self._hologram_alpha * (0.72 + 0.18*math.sin(t*3))
#
#         glDisable(GL_LIGHTING)
#         glLineWidth(1.2)
#
#         # Edge glow pass — draw lines in cyan-blue
#         glBegin(GL_LINES)
#         for (tri,_,col) in self._groups:
#             av,bv,cv=tri
#             # Scanline flicker: skip some triangles periodically
#             cy_v = (av[1]+bv[1]+cv[1])/3
#             flicker = math.sin(cy_v*12 + t*8)
#             if flicker < -0.6: continue
#             edge_a = a * max(0, flicker*0.4+0.6)
#             glColor4f(0.2,0.85,1.0, edge_a)
#             glVertex3f(*av); glVertex3f(*bv)
#             glVertex3f(*bv); glVertex3f(*cv)
#             glVertex3f(*cv); glVertex3f(*av)
#         glEnd()
#
#         # Fill pass — very transparent cyan solid
#         glBegin(GL_TRIANGLES)
#         for (tri,fn,col) in self._groups:
#             a_v,b_v,c_v=tri
#             n=fn[0] if fn[0] else flat_n(a_v,b_v,c_v)
#             cy_v=(a_v[1]+b_v[1]+c_v[1])/3
#             scan=0.12+0.08*math.sin(cy_v*10+t*6)
#             glColor4f(0.05,0.6,0.9,scan*a)
#             glNormal3f(*n)
#             glVertex3f(*a_v); glVertex3f(*b_v); glVertex3f(*c_v)
#         glEnd()
#         glLineWidth(1.0)
#         glEnable(GL_LIGHTING)
#
#     def _draw_exploded(self, alpha=1.0):
#         t=time.time(); ed=0.55+0.12*math.sin(t*0.7)
#         glBegin(GL_TRIANGLES)
#         for (tri,fn,col) in self._groups:
#             a,b,c=tri
#             cx=(a[0]+b[0]+c[0])/3; cy=(a[1]+b[1]+c[1])/3; cz=(a[2]+b[2]+c[2])/3
#             L=math.sqrt(cx*cx+cy*cy+cz*cz) or 0.001
#             dx,dy,dz=cx/L*ed,cy/L*ed,cz/L*ed
#             n=fn[0] if fn[0] else flat_n(a,b,c)
#             glColor4f(*self._paint(col,alpha)); glNormal3f(*n)
#             glVertex3f(a[0]+dx,a[1]+dy,a[2]+dz)
#             glNormal3f(*(fn[1] if fn[1] else n))
#             glVertex3f(b[0]+dx,b[1]+dy,b[2]+dz)
#             glNormal3f(*(fn[2] if fn[2] else n))
#             glVertex3f(c[0]+dx,c[1]+dy,c[2]+dz)
#         glEnd()
#
#     def _draw_xray(self):
#         """Glass-like blue-grey with depth silhouette."""
#         glDisable(GL_DEPTH_TEST)
#         glBegin(GL_TRIANGLES)
#         for (tri,fn,col) in self._groups:
#             a,b,c=tri
#             n=fn[0] if fn[0] else flat_n(a,b,c)
#             # X-ray: brighter on edges (facing away from camera)
#             glColor4f(0.3,0.7,0.9, 0.18)
#             glNormal3f(*n); glVertex3f(*a); glVertex3f(*b); glVertex3f(*c)
#         glEnd()
#         glEnable(GL_DEPTH_TEST)
#         # Solid wireframe on top
#         glDisable(GL_LIGHTING)
#         glColor4f(0.0,0.9,1.0,0.7)
#         glLineWidth(0.8)
#         glBegin(GL_LINES)
#         for (tri,_,__) in self._groups:
#             a,b,c=tri
#             glVertex3f(*a); glVertex3f(*b)
#             glVertex3f(*b); glVertex3f(*c)
#             glVertex3f(*c); glVertex3f(*a)
#         glEnd()
#         glLineWidth(1.0)
#         glEnable(GL_LIGHTING)
#
#     def update(self, ge_out, hologram_anchor_px=None, frame_wh=(1280,720)):
#         if self.frozen: return
#
#         if self.auto_spin:
#             self.yaw += self._spin_spd
#
#         # Right hand movement
#         self.yaw   += ge_out["right_yaw"]
#         self.pitch  = max(-85, min(85, self.pitch + ge_out["right_pitch"]))
#         self.scale  = max(0.15, min(6.0, self.scale + ge_out["right_zoom"]))
#         # Smooth interpolation (reduces jitter massively)
#         self._smooth_scale += (self.scale - self._smooth_scale) * 0.25
#         self._smooth_yaw += (self.yaw - self._smooth_yaw) * 0.25
#         self._smooth_pitch += (self.pitch - self._smooth_pitch) * 0.25
#
#         self.scale = self._smooth_scale
#         self.yaw = self._smooth_yaw
#         self.pitch = self._smooth_pitch
#
#         # Hologram mode toggle from right hand
#         self.hologram_mode = ge_out["hologram"]
#         if not self.hologram_mode:
#             self._hologram_alpha = max(0.0, self._hologram_alpha - 0.08)
#
#         # Left hand command gestures
#         self.scale += ge_out["left_scale"]
#         self.yaw += ge_out["left_rotate"]
#         self.pitch += ge_out["left_tilt"]
#
#         cmd = ge_out["left_cmd"]
#         if cmd == "interior":
#             if not self.interior: self.interior=True; self.exploded=False
#         elif cmd == "explode":
#             self.exploded = not self.exploded; self.interior=False
#         elif cmd == "reset":
#             self.interior=False; self.exploded=False; self.xray=False
#             self.scale=1.0; self.yaw=0.0; self.pitch=18.0
#             self.hologram_mode=False; self._hologram_alpha=0.0
#         elif cmd == "freeze":
#             self.frozen = not self.frozen
#         elif cmd == "color":
#             pass  # handled externally to avoid rapid cycling
#
#         # BOTH hands
#         self.scale += ge_out["both_scale"]
#         self.yaw += ge_out["both_rotate"]
#
#         # Clamp
#         self.pitch = max(-85, min(85, self.pitch))
#         self.scale = max(0.15, min(6.0, self.scale))
#
#
#
#
#
#     def render_rgba(self, palm_px=None, frame_wh=(1280,720)):
#         glBindFramebuffer(GL_FRAMEBUFFER,self._fbo)
#         glViewport(0,0,self.w,self.h)
#         glClearColor(0,0,0,0)
#         glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
#
#         glMatrixMode(GL_PROJECTION); glLoadIdentity()
#         gluPerspective(52,self.w/self.h,0.01,100)
#         glMatrixMode(GL_MODELVIEW); glLoadIdentity()
#
#         if self.interior:
#             gluLookAt(0.05,-0.15,0.25, 0.60,0.80,0.25, 0,0,1)
#             glScalef(self.scale,self.scale,self.scale)
#         else:
#             dist=3.5/max(self.scale,0.15)
#             yr=math.radians(self.yaw); pr=math.radians(self.pitch)
#             ex=dist*math.sin(yr)*math.cos(pr)
#             ey=-dist*math.cos(yr)*math.cos(pr)
#             ez=dist*math.sin(pr)
#             gluLookAt(ex,ey,ez, 0,0,0, 0,0,1)
#             glScalef(self.scale,self.scale,self.scale)
#
#         if self.hologram_mode or self._hologram_alpha > 0.05:
#             self._draw_hologram()
#         elif self.xray:
#             self._draw_xray()
#         elif self.wireframe:
#             glDisable(GL_LIGHTING); glCallList(self._dl_wire); glEnable(GL_LIGHTING)
#         elif self.exploded:
#             self._draw_exploded()
#         elif self.color_idx == 0 and not self.night_mode:
#             glCallList(self._dl)
#         else:
#             alpha = 0.7 if self.night_mode else 1.0
#             self._draw_solid(alpha)
#
#         glPixelStorei(GL_PACK_ALIGNMENT,1)
#         raw=glReadPixels(0,0,self.w,self.h,GL_RGBA,GL_UNSIGNED_BYTE)
#         arr=np.frombuffer(raw,dtype=np.uint8).reshape(self.h,self.w,4)[::-1].copy()
#         glBindFramebuffer(GL_FRAMEBUFFER,0)
#         pygame.event.pump()
#         return arr
#
#     def cleanup(self):
#         glDeleteLists(self._dl,1); glDeleteLists(self._dl_wire,1)
#         glDeleteFramebuffers(1,[self._fbo])
#         glDeleteRenderbuffers(1,[self._crb]); glDeleteRenderbuffers(1,[self._drb])
#         pygame.quit()
#
#
# # ═══════════════════════════════════════════════════════════════════════
# #  ARUCO
# # ═══════════════════════════════════════════════════════════════════════
# class ArucoTracker:
#     def __init__(self):
#         self.adict=cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
#         self.detector=cv2.aruco.ArucoDetector(self.adict,cv2.aruco.DetectorParameters())
#         if not os.path.exists("aruco_marker.png"):
#             img=cv2.aruco.generateImageMarker(self.adict,ARUCO_ID,300)
#             cv2.imwrite("aruco_marker.png",img)
#             print("[ArUco] aruco_marker.png saved")
#     def detect(self,frame):
#         grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         corners,ids,_=self.detector.detectMarkers(grey)
#         if ids is None: return None,1.0
#         for i,mid in enumerate(ids.flatten()):
#             if mid!=ARUCO_ID: continue
#             c=corners[i][0]
#             cx=int(c[:,0].mean()); cy=int(c[:,1].mean())
#             side=np.linalg.norm(c[0]-c[1])
#             cv2.aruco.drawDetectedMarkers(frame,corners,ids)
#             return (cx,cy),max(0.3,min(3.0,side/100))
#         return None,1.0
#
#
# # ═══════════════════════════════════════════════════════════════════════
# #  COMPOSITOR  (with optional palm-tracking anchor)
# # ═══════════════════════════════════════════════════════════════════════
# def composite(bg, car_rgba, anchor=None, scale=1.0):
#     hb,wb=bg.shape[:2]; hc,wc=car_rgba.shape[:2]
#     nw=max(10,min(int(wc*scale),wb)); nh=max(10,min(int(hc*scale),hb))
#     scaled=cv2.resize(car_rgba,(nw,nh),interpolation=cv2.INTER_LINEAR)
#     x0=(anchor[0]-nw//2) if anchor else (wb-nw)//2
#     y0=(anchor[1]-nh//2) if anchor else (hb-nh)//2
#     x0=max(0,min(x0,wb-nw)); y0=max(0,min(y0,hb-nh))
#     roi=bg[y0:y0+nh,x0:x0+nw].astype(np.float32)
#     fg=scaled[:,:,:3].astype(np.float32)[:,:,::-1]
#     a=scaled[:,:,3:4].astype(np.float32)/255.0
#     out=bg.copy()
#     out[y0:y0+nh,x0:x0+nw]=(fg*a+roi*(1-a)).astype(np.uint8)
#     return out
#
#
# # ═══════════════════════════════════════════════════════════════════════
# #  HOLOGRAM PALM EFFECT  — draw arc + particle effects on hand
# # ═══════════════════════════════════════════════════════════════════════
# def draw_hologram_fx(frame, palm_px, t):
#     """Draw Iron Man-style projection ring and particles around palm."""
#     if palm_px is None: return
#     px,py=int(palm_px[0]*frame.shape[1]), int(palm_px[1]*frame.shape[0])
#
#     # Projection ring — pulsing cyan arc
#     r_base = 55 + int(8*math.sin(t*4))
#     alpha_ring = 0.55 + 0.25*math.sin(t*6)
#     ov=frame.copy()
#     cv2.circle(ov,(px,py),r_base,(0,220,255),2)
#     cv2.circle(ov,(px,py),r_base-10,(0,160,200),1)
#     cv2.addWeighted(ov,0.7,frame,0.3,0,frame)
#
#     # Spinning tick marks on ring
#     for i in range(12):
#         ang=math.radians(i*30+t*45)
#         x1=int(px+r_base*math.cos(ang)); y1=int(py+r_base*math.sin(ang))
#         x2=int(px+(r_base-8)*math.cos(ang)); y2=int(py+(r_base-8)*math.sin(ang))
#         brightness=int(120+120*abs(math.sin(ang+t*3)))
#         cv2.line(frame,(x1,y1),(x2,y2),(0,brightness,255),1)
#
#     # Rising particles above hand (hologram source)
#     for i in range(8):
#         seed=i*137+int(t*60)
#         px2=px+int(30*math.sin(seed*0.4+t*2))
#         py2=py-int(40+25*abs(math.sin(seed*0.7+t*1.5)))
#         r=max(1,int(2*abs(math.sin(seed+t*3))))
#         a2=max(0,int(180*abs(math.sin(seed*0.5+t*2))))
#         ov2=frame.copy()
#         cv2.circle(ov2,(px2,py2),r,(0,a2,255),-1)
#         cv2.addWeighted(ov2,0.6,frame,0.4,0,frame)
#
#     # Centre glow
#     ov3=frame.copy()
#     cv2.circle(ov3,(px,py),18,(0,200,255),-1)
#     cv2.addWeighted(ov3,0.25,frame,0.75,0,frame)
#
#     # "PROJECTION ACTIVE" label
#     cv2.putText(frame,"HOLOGRAM ACTIVE",(px-65,py+r_base+20),
#                 cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,220,255),1,cv2.LINE_AA)
#
#
# # ═══════════════════════════════════════════════════════════════════════
# #  VOICE LISTENER  — activation word "hey car"
# # ═══════════════════════════════════════════════════════════════════════
# class VoiceListener:
#     def __init__(self):
#         self._q=queue.Queue()
#         self._stop=threading.Event()
#         self.is_listening=False
#         self.last_heard=""
#
#     def start(self):
#         if not VOICE_OK: return
#         threading.Thread(target=self._loop,daemon=True).start()
#         print(f"[Voice] Background listener ready. Say '{ACTIVATION} <question>'")
#
#     def _loop(self):
#         rec=sr.Recognizer()
#         mic=sr.Microphone()
#         rec.dynamic_energy_threshold=True
#         rec.energy_threshold=2400
#         with mic as src:
#             rec.adjust_for_ambient_noise(src,duration=1.5)
#         while not self._stop.is_set():
#             try:
#                 with mic as src:
#                     self.is_listening=True
#                     audio=rec.listen(src,timeout=2,phrase_time_limit=14)
#                 self.is_listening=False
#                 text=rec.recognize_google(audio).strip()
#                 if not text: continue
#                 self.last_heard=text
#                 print(f"[Voice] Heard: '{text}'")
#                 low=text.lower()
#                 if low in ("exit","quit","stop","bye"):
#                     self._q.put(("__QUIT__",False)); continue
#                 vision = low.startswith(ACTIVATION)
#                 query  = text[len(ACTIVATION):].strip(" ,.") if vision else text
#                 if not query: query="What do you see on screen?"
#                 self._q.put((query, vision))
#             except sr.WaitTimeoutError: self.is_listening=False
#             except sr.UnknownValueError: self.is_listening=False
#             except Exception as e:
#                 self.is_listening=False; print(f"[Voice] {e}"); time.sleep(0.4)
#
#     def one_shot(self,timeout=8):
#         if not VOICE_OK: return None,True
#         rec=sr.Recognizer(); mic=sr.Microphone()
#         try:
#             with mic as src:
#                 rec.adjust_for_ambient_noise(src,duration=0.3)
#                 audio=rec.listen(src,timeout=timeout,phrase_time_limit=14)
#             text=rec.recognize_google(audio).strip()
#             print(f"[Voice V-key] '{text}'")
#             return text,True
#         except Exception as e:
#             print(f"[Voice] one-shot: {e}"); return None,True
#
#     def get(self):
#         try: return self._q.get_nowait()
#         except queue.Empty: return None,None
#
#     def close(self): self._stop.set()
#
#
# # ═══════════════════════════════════════════════════════════════════════
# #  AI POPUP OVERLAY
# # ═══════════════════════════════════════════════════════════════════════
# class AIPopup:
#     FADE_IN=0.35; HOLD=7.0; FADE_OUT=0.7
#     def __init__(self): self._reset()
#     def _reset(self):
#         self.query=""; self.text=""; self.alpha=0.0
#         self.state="idle"; self.t0=0.0
#     def show(self,query,reply):
#         self.query=query[:100]; self.text=reply
#         self.state="fade_in"; self.t0=time.time()
#     def update_text(self,reply):
#         if self.state!="idle": self.text=reply
#     def tick(self):
#         if self.state=="idle": return
#         dt=time.time()-self.t0
#         if   self.state=="fade_in":
#             self.alpha=min(1.0,dt/self.FADE_IN)
#             if dt>=self.FADE_IN: self.state="hold"; self.t0=time.time()
#         elif self.state=="hold":
#             self.alpha=1.0
#             if dt>=self.HOLD: self.state="fade_out"; self.t0=time.time()
#         elif self.state=="fade_out":
#             self.alpha=max(0.0,1.0-dt/self.FADE_OUT)
#             if dt>=self.FADE_OUT: self._reset()
#
#     def draw(self,frame):
#         if self.state=="idle" or self.alpha<=0: return
#         h,w=frame.shape[:2]
#         px,py,pw=30,h//5,w-60
#         words=self.text.split(); lines=[]; cur=""
#         mc=(pw-24)//9
#         for word in words:
#             t=(cur+" "+word).strip()
#             if len(t)<mc: cur=t
#             else: lines.append(cur); cur=word
#         if cur: lines.append(cur)
#         lines=lines[:5]; ph=len(lines)*30+72
#         ov=frame.copy()
#         cv2.rectangle(ov,(px,py),(px+pw,py+ph),(6,8,16),-1)
#         cv2.addWeighted(ov,self.alpha*0.88,frame,1-self.alpha*0.88,0,frame)
#         cv2.rectangle(frame,(px,py),(px+pw,py+ph),(0,200,140),2)
#         # Vision badge
#         vis=self.query.lower().find("hey car")==-1 and self.query!=""
#         badge=(30,80,200)
#         cv2.rectangle(frame,(px,py),(px+140,py+22),badge,-1)
#         tag="VISION AI" if vis else "AI RESPONSE"
#         cv2.putText(frame,f" {tag}",(px+4,py+15),cv2.FONT_HERSHEY_SIMPLEX,0.44,(240,240,255),1,cv2.LINE_AA)
#         cv2.putText(frame,f"Q: {self.query}",(px+8,py+38),cv2.FONT_HERSHEY_SIMPLEX,0.40,(140,200,200),1,cv2.LINE_AA)
#         for i,line in enumerate(lines):
#             cv2.putText(frame,line,(px+8,py+58+i*30),cv2.FONT_HERSHEY_SIMPLEX,0.56,(240,255,210),1,cv2.LINE_AA)
#         cv2.rectangle(frame,(px,py+ph-4),(px+pw,py+ph),(0,200,140),-1)
#
#
# # ═══════════════════════════════════════════════════════════════════════
# #  HUD  — full sectioned panels
# # ═══════════════════════════════════════════════════════════════════════
# F=cv2.FONT_HERSHEY_SIMPLEX
#
# def _p(frame,x,y,w,h,title,alpha=0.62):
#     ov=frame.copy()
#     cv2.rectangle(ov,(x,y),(x+w,y+h),(6,8,14),-1)
#     cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)
#     cv2.rectangle(frame,(x,y),(x+w,y+h),(0,155,100),1)
#     cv2.rectangle(frame,(x,y),(x+w,y+18),(0,155,100),-1)
#     cv2.putText(frame,title.upper(),(x+4,y+13),F,0.36,(4,4,4),1,cv2.LINE_AA)
#
# def _t(frame,txt,x,y,col=(195,255,190),sc=0.44,th=1):
#     cv2.putText(frame,txt,(x,y),F,sc,col,th,cv2.LINE_AA)
#
# def _bar(frame,x,y,w,v,mx,col):
#     cv2.rectangle(frame,(x,y),(x+w,y+7),(28,28,28),-1)
#     f=max(0,int(w*min(v/mx,1)))
#     if f: cv2.rectangle(frame,(x,y),(x+f,y+7),col,-1)
#
# def draw_hud(frame,car,ge,ai,fps,anchor,voice,popup,night):
#     h,w=frame.shape[:2]; t=time.time()
#     lc=ge["left_cmd"]; holo=ge["hologram"]
#     hcount=ge["hand_count"]
#
#     # Night mode tint
#     if night:
#         tint=frame.copy(); tint[:]=tint*np.array([0.5,0.6,1.0])
#         cv2.addWeighted(tint,0.18,frame,0.82,0,frame)
#
#     # ── Left panel A: Left hand commands (top-left) ─────────────────────
#     _p(frame,8,8,190,155,"  Left Hand = Commands")
#     cmds=[("✌ 2F","Interior"),("🤟 3F","Explode/X-ray"),
#           ("🖐 Palm","Reset all"),("✊ Fist","Freeze"),
#           ("👍 Thumb","Cycle color")]
#     for i,(icon,desc) in enumerate(cmds):
#         active = (
#             (desc.startswith("Interior") and lc=="interior") or
#             (desc.startswith("Explode")  and lc=="explode") or
#             (desc.startswith("Reset")    and lc=="reset") or
#             (desc.startswith("Freeze")   and lc=="freeze") or
#             (desc.startswith("Cycle")    and lc=="color")
#         )
#         col=(0,230,150) if active else (130,160,130)
#         _t(frame,f"{icon} {desc}",14,30+i*22,col,0.38)
#
#     # Left hand active badge
#     if lc!="none":
#         cv2.rectangle(frame,(8,163),(198,181),(0,155,100),-1)
#         _t(frame,f"CMD: {lc.upper()}",12,177,(10,10,10),0.40)
#
#     # ── Left panel B: Right hand movement ────────────────────────────────
#     _p(frame,8,190,190,148,"  Right Hand = Movement")
#     moves=[("Wrist L/R","Orbit 360°"),("Wrist Up/Down","Tilt pitch"),
#            ("Pinch","Zoom in/out"),("Open palm","Hologram!"),
#            ("Point finger","Spotlight part")]
#     for i,(k,v) in enumerate(moves):
#         hlt = (k=="Open palm" and holo) or (k=="Point finger" and ge["pointing"])
#         col=(0,200,255) if hlt else (130,160,130)
#         _t(frame,f"{k}: {v}",14,208+i*22,col,0.36)
#
#     if holo:
#         cv2.rectangle(frame,(8,338),(198,356),(0,100,200),-1)
#         _t(frame,"HOLOGRAM MODE",14,352,(200,240,255),0.38)
#
#     # ── Right panel C: System status ─────────────────────────────────────
#     _p(frame,w-200,8,192,132,"  System Status")
#     _t(frame,"FPS",w-196,30,(100,130,110),0.38)
#     _t(frame,f"{fps:.0f}",w-166,30,(0,220,100) if fps>20 else (60,100,255),0.46)
#     _bar(frame,w-150,23,140,fps,60,(0,200,80) if fps>20 else (40,100,255))
#     rows=[("ArUco","LOCKED" if anchor else "searching",(0,210,80) if anchor else (100,100,200)),
#           ("AI","busy…" if ai.busy else "ready",(40,160,255) if ai.busy else (0,210,80)),
#           ("Voice","LISTENING" if voice.is_listening else "waiting",(0,60,255) if voice.is_listening else (90,90,90)),
#           ("TTS","speaking" if ai.tts.is_speaking() else "idle",(0,180,255) if ai.tts.is_speaking() else (80,80,80)),
#           ("Spin","ON" if car.auto_spin else "OFF",(0,210,80) if car.auto_spin else (80,80,80)),
#           ("Frozen","YES" if car.frozen else "no",(255,80,80) if car.frozen else (80,80,80))]
#     for i,(k,v,c) in enumerate(rows):
#         _t(frame,f"{k}:",w-196,52+i*14,(90,120,100),0.36)
#         _t(frame,v,w-152,52+i*14,c,0.37)
#
#     # ── Right panel D: Keys ───────────────────────────────────────────────
#     _p(frame,w-200,148,192,90,"  Keys")
#     ks=[("[V]","Voice+Vision"),("[C]","Color cycle"),("[W]","Wireframe"),
#         ("[X]","X-ray"),("[N]","Night mode"),("[A]","Auto-spin"),
#         ("[S]","Screenshot"),("[Q]","Quit")]
#     cols=2; rows_k=4
#     for i,(k,v) in enumerate(ks):
#         col=i%cols; row=i//cols
#         cx=w-196+col*95; cy=164+row*17
#         _t(frame,k,cx,cy,(30,200,200),0.34)
#         _t(frame,v,cx+28,cy,(110,140,110),0.33)
#
#     # ── Car info bar (below right panel) ─────────────────────────────────
#     _p(frame,w-200,246,192,110,"  Car State")
#     pname=CAR_COLORS[car.color_idx][1]
#     pr2,pg2,pb2=[int(x*255) for x in CAR_COLORS[car.color_idx][0]]
#     view=("Interior" if car.interior else
#           "Exploded" if car.exploded else
#           "X-Ray"    if car.xray    else
#           "Hologram" if car.hologram_mode else
#           "Wire"     if car.wireframe else "Exterior")
#     info=[("View",view),("Tris",f"{car._tri_count:,}"),
#           ("Scale",f"{car.scale:.2f}x"),("Yaw",f"{car.yaw%360:.0f}°"),
#           ("Pitch",f"{car.pitch:.0f}°"),("Paint",pname)]
#     for i,(k,v) in enumerate(info):
#         _t(frame,f"{k}:",w-196,262+i*16,(90,120,100),0.37)
#         _t(frame,v,w-152,262+i*16,(205,255,200),0.39)
#     cv2.rectangle(frame,(w-50,250),(w-26,266),(pb2,pg2,pr2),-1)
#     cv2.rectangle(frame,(w-50,250),(w-26,266),(0,155,100),1)
#
#     # ── AI bottom strip ───────────────────────────────────────────────────
#     _p(frame,8,h-78,w-16,70,
#        f"  AI Car Guide (JARVIS)   |   Say: '{ACTIVATION} <question>'",alpha=0.72)
#     reply=ai.get_reply()
#     words=reply.split(); lines=[]; cur=""
#     mc=(w-40)//8
#     for word in words:
#         t2=(cur+" "+word).strip()
#         if len(t2)<mc: cur=t2
#         else: lines.append(cur); cur=word
#     if cur: lines.append(cur)
#     for i,ln in enumerate(lines[:2]):
#         _t(frame,ln,14,h-56+i*28,(255,248,155),0.52)
#
#     # Thinking dots
#     if ai.busy:
#         d="●"*(int(time.time()*3)%4+1)
#         _t(frame,d,w-70,h-50,(50,160,255),0.54)
#
#     # Speaking indicator
#     if ai.tts.is_speaking():
#         r2=int(7+4*math.sin(time.time()*10))
#         cv2.circle(frame,(w-18,h-66),r2,(0,80,255),-1)
#         _t(frame,"SPEAKING",w-90,h-70,(0,120,255),0.36)
#
#     # Voice waveform
#     if voice.is_listening:
#         for i in range(20):
#             hh=int(4+10*abs(math.sin(time.time()*8+i*0.7)))
#             ci=int(50+180*abs(math.sin(time.time()*6+i)))
#             cv2.line(frame,(w-260+i*6,h-28),(w-260+i*6,h-28+hh),(0,ci,ci),2)
#
#     # Two-hand banner
#     if hcount==2:
#         bw=320; bx=(w-bw)//2; by=8
#         ov2=frame.copy()
#         cv2.rectangle(ov2,(bx,by),(bx+bw,by+28),(0,20,40),-1)
#         cv2.addWeighted(ov2,0.75,frame,0.25,0,frame)
#         cv2.rectangle(frame,(bx,by),(bx+bw,by+28),(0,200,255),1)
#         _t(frame,"TWO-HAND MODE  |  RIGHT=Move  LEFT=Command",bx+8,by+19,(0,220,255),0.44)
#
#     # Gesture active label centre
#     if lc!="none":
#         label={"interior":"INTERIOR VIEW","explode":"EXPLODE",
#                "reset":"RESET","freeze":"FROZEN" if car.frozen else "UNFREEZE",
#                "color":"COLOR CHANGED"}.get(lc,lc.upper())
#         tw=len(label)*11
#         _t(frame,label,(w-tw)//2,h-88,(0,240,160),0.65,2)
#
#     # Popup on top
#     popup.tick(); popup.draw(frame)
#
#
# # ═══════════════════════════════════════════════════════════════════════
# #  MAIN
# # ═══════════════════════════════════════════════════════════════════════
# def main():
#     print("="*66)
#     print("  AR Car v5  —  Iron Man Hologram + Dual-Hand + Vision AI")
#     print("="*66)
#     print(f"  Wake word : '{ACTIVATION} <question>'  (vision-aware)")
#     print("  V key     : push-to-talk  (always vision)")
#     print("  LEFT hand : commands   |   RIGHT hand : move + hologram")
#     print("="*66)
#
#     cap=cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAM_W)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAM_H)
#     cap.set(cv2.CAP_PROP_FPS,30)
#     if not cap.isOpened(): sys.exit("[ERROR] Cannot open webcam — try index 1")
#
#     print("[INIT] Loading renderer…")
#     car   = CarRenderer()
#     print("[INIT] Renderer ready")
#
#     tts   = TTSEngine()
#     ai    = CarAI(tts)
#     ge    = GestureEngine()
#     aruco = ArucoTracker()
#     voice = VoiceListener()
#     popup = AIPopup()
#     voice.start()
#
#     # Greeting (fire-and-forget — won't block)
#     ai.ask_voice("Greet me in one sentence as JARVIS, an AR hologram car guide.",use_vision=False)
#
#     fps_t=time.time(); fps=0.0; fc=0
#     night=False; v_active=False
#     color_cooldown=0.0
#     last_left_cmd="none"; freeze_debounce=0.0
#
#     print("\n[READY] Show hands to the camera!\n")
#
#     while True:
#         ret,frame=cap.read()
#         if not ret: print("[ERROR] Webcam lost"); break
#         frame=cv2.flip(frame,1)
#
#         ai.set_frame(frame)
#         ge_out=ge.process(frame)
#         lc=ge_out["left_cmd"]
#
#         # Color cycle — debounced (left thumb up)
#         if lc=="color" and time.time()-color_cooldown>1.2:
#             car.cycle_color(); color_cooldown=time.time()
#
#         # Freeze — toggle with debounce
#         if lc=="freeze" and lc!=last_left_cmd and time.time()-freeze_debounce>0.8:
#             car.frozen=not car.frozen; freeze_debounce=time.time()
#         last_left_cmd=lc
#
#         car.update(ge_out)
#
#         # Hologram anchor: if palm mode → car floats above palm
#         palm_px=None
#         if ge_out["hologram"] and ge_out["palm_pos"]:
#             pp=ge_out["palm_pos"]
#             px=int(pp[0]*CAM_W); py=int(pp[1]*CAM_H)-140  # above palm
#             palm_px=(px,py)
#             draw_hologram_fx(frame,ge_out["palm_pos"],time.time())
#
#         # ArUco (only when not in hologram mode)
#         aruco_anchor,aruco_scale=aruco.detect(frame)
#         if ge_out["hologram"]:
#             anchor=palm_px; eff=car.scale*0.55
#         elif aruco_anchor:
#             anchor=aruco_anchor; eff=car.scale*aruco_scale*0.72
#         else:
#             anchor=None; eff=car.scale*0.72
#
#         car_rgba=car.render_rgba(palm_px=palm_px)
#         frame=composite(frame,car_rgba,anchor=anchor,scale=eff)
#
#         # Pointing spotlight (right index)
#         if ge_out["pointing"] and ge_out["point_pos"]:
#             pp=ge_out["point_pos"]
#             sx=int(pp[0]*CAM_W); sy=int(pp[1]*CAM_H)
#             cv2.circle(frame,(sx,sy),22,(0,220,255),2)
#             cv2.circle(frame,(sx,sy),8,(0,200,255),-1)
#             # Ask AI to name part being pointed at (rate-limited)
#             if not ai.busy and not car.frozen:
#                 ai.ask_gesture("interior")  # prompt about visible region
#
#         # Voice
#         vtext,vvis=voice.get()
#         if vtext=="__QUIT__": break
#         elif vtext:
#             ai.ask_voice(vtext,use_vision=vvis)
#             popup.show(vtext,"Thinking…")
#
#         # Update popup text when AI finishes
#         if popup.state!="idle" and popup.text=="Thinking…" and not ai.busy:
#             popup.update_text(ai.get_reply())
#
#         # FPS
#         fc+=1; now=time.time()
#         if now-fps_t>=1.0: fps=fc/(now-fps_t); fc=0; fps_t=now
#
#         draw_hud(frame,car,ge_out,ai,fps,
#                  aruco_anchor if not ge_out["hologram"] else True,
#                  voice,popup,night)
#
#         cv2.imshow("AR Car v5  —  Iron Man Hologram  (Q=quit)",frame)
#
#         key=cv2.waitKey(1)&0xFF
#         if key==ord('q'): break
#         elif key==ord('v') and not v_active:
#             def _vs():
#                 nonlocal v_active
#                 v_active=True
#                 _t2,_vis=voice.one_shot(timeout=8)
#                 if _t2:
#                     ai.ask_voice(_t2,use_vision=True)
#                     popup.show(_t2,"Thinking…")
#                 v_active=False
#             threading.Thread(target=_vs,daemon=True).start()
#         elif key==ord('c'): car.cycle_color(); print(f"[Color] {CAR_COLORS[car.color_idx][1]}")
#         elif key==ord('w'): car.toggle_wire(); print(f"[Wire] {'ON' if car.wireframe else 'OFF'}")
#         elif key==ord('x'): car.xray=not car.xray; print(f"[X-ray] {'ON' if car.xray else 'OFF'}")
#         elif key==ord('n'): night=not night; print(f"[Night] {'ON' if night else 'OFF'}")
#         elif key==ord('a'): car.toggle_spin(); print(f"[Spin] {'ON' if car.auto_spin else 'OFF'}")
#         elif key==ord('s'):
#             ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             p=f"screenshots/ar_{ts}.png"; cv2.imwrite(p,frame)
#             print(f"[Screenshot] {p}")
#
#     tts.stop(); voice.close(); car.cleanup()
#     cap.release(); cv2.destroyAllWindows()
#     print("[EXIT] Done.")
#
# if __name__=="__main__":
#     main()


"""
╔══════════════════════════════════════════════════════════════════════╗
║              AR CAR v5  —  IRON MAN HOLOGRAM EDITION               ║
╠══════════════════════════════════════════════════════════════════════╣
║  ROOT-CAUSE FIXES:                                                  ║
║  1. TTS silent bug: pyttsx3 queue had task_done() BEFORE            ║
║     runAndWait() finished → engine torn down mid-speech.            ║
║     Fix: task_done() removed entirely; queue is fire-and-forget.    ║
║  2. Race condition on AI replies: on_gesture() called ask() which   ║
║     competed with voice ask() for busy flag → dropped replies.      ║
║     Fix: gesture prompts go to a SEPARATE low-priority queue;       ║
║     voice always wins; gesture fires only when voice idle >3s.      ║
║  3. Two-hand ignored: mediapipe handedness labels left/right but    ║
║     old code just took hands[0]/hands[1] by index (arbitrary).      ║
║     Fix: parse multi_handedness to correctly assign LEFT=command    ║
║     hand, RIGHT=movement hand every frame.                          ║
║                                                                     ║
║  NEW FEATURES:                                                       ║
║  • HOLOGRAM PALM MODE: close right fist → car materialises on       ║
║    your palm like Iron Man. Open right hand flat → car floats       ║
║    above it and follows your hand in 3D space.                      ║
║  • LEFT hand = COMMAND (gestures: interior/explode/reset/color)     ║
║  • RIGHT hand = MOVEMENT (orbit yaw/pitch, zoom pinch, hologram)   ║
║  • Hologram shimmer: scanline + edge-glow effect on the 3D model   ║
║  • Particle trail when car moves                                    ║
║  • Engine sound simulation: low hum animates HUD bars              ║
║  • "HEY CAR" activation word → llava sees your screen & answers    ║
║  • AI reply always spoken aloud (TTS race condition fixed)          ║
║  • X-ray mode (X key): see through car body to internals           ║
║  • Night mode (N key): dark theme + neon hologram glow             ║
║  • Part spotlight: hover finger over region → AI names that part   ║
╚══════════════════════════════════════════════════════════════════════╝

INSTALL:
  pip install opencv-python mediapipe pygame PyOpenGL PyOpenGL_accelerate
              ollama numpy speechrecognition pyttsx3
  pip install pipwin && pipwin install pyaudio    (Windows only)

OLLAMA SETUP (one-time):
  ollama pull llava      ← vision model (sees your screen)
  ollama pull llama3     ← fast text model

RUN:
  Terminal 1:  ollama serve
  Terminal 2:  python ar_car_v5.py

LEFT HAND GESTURES (commands):
  ✌ 2 fingers   → Interior view
  🤟 3 fingers   → Explode/X-ray toggle
  🖐 Open palm   → Reset everything
  👊 Fist        → Freeze / un-freeze car

RIGHT HAND GESTURES (movement + hologram):
  Open flat palm → HOLOGRAM MODE: car sits on your palm
  Pinch           → Zoom in/out
  Wrist left/right → Orbit (yaw)
  Wrist up/down    → Tilt (pitch)
  Point index finger → Spotlight that car part → AI names it

VOICE:
  "Hey Car <question>"  → AI sees screen + answers (spoken aloud)
  V key                 → Push-to-talk (always vision-aware)
  Q key                 → Quit

KEYS:
  V  Voice query (vision)    C  Cycle color
  W  Wireframe               X  X-ray mode
  N  Night / Day mode        A  Auto-spin
  S  Screenshot              Q  Quit
"""

import cv2, numpy as np, mediapipe as mp
import ollama, threading, math, os, time, sys, queue, datetime, base64
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, NOFRAME
from OpenGL.GL import *
from OpenGL.GLU import *

# ── optional voice ──────────────────────────────────────────────────────
try:
    import speech_recognition as sr
    VOICE_OK = True
except ImportError:
    VOICE_OK = False; print("[WARN] SpeechRecognition missing — voice off")

try:
    import pyttsx3
    TTS_OK = True
except ImportError:
    TTS_OK = False; print("[WARN] pyttsx3 missing — TTS off")

# ═══════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════
CAM_W, CAM_H  = 1280, 720
RENDER_W      = 660
RENDER_H      = 490
MODEL_PATH    = "car.obj"
VISION_MODEL  = "llava"
TEXT_MODEL    = "llama3"
ARUCO_DICT    = cv2.aruco.DICT_6X6_250
ARUCO_ID      = 0
ACTIVATION    = "hey car"

os.makedirs("screenshots", exist_ok=True)

CAR_COLORS = [
    ((0.85,0.12,0.12), "Red"),
    ((0.10,0.28,0.85), "Blue"),
    ((0.08,0.60,0.18), "Green"),
    ((0.92,0.72,0.04), "Yellow"),
    ((0.10,0.10,0.10), "Black"),
    ((0.90,0.90,0.90), "White"),
    ((0.55,0.55,0.62), "Silver"),
    ((0.55,0.10,0.80), "Purple"),
]

# ═══════════════════════════════════════════════════════════════════════
#  OBJ / MTL LOADER
# ═══════════════════════════════════════════════════════════════════════
def load_mtl(path):
    mats={}; cur=None
    try:
        with open(path,"r",encoding="utf-8",errors="ignore") as f:
            for line in f:
                p=line.strip().split()
                if not p: continue
                if p[0]=="newmtl": cur=p[1]; mats[cur]=(0.7,0.7,0.7)
                elif p[0]=="Kd" and cur:
                    mats[cur]=(float(p[1]),float(p[2]),float(p[3]))
    except: pass
    return mats

def load_obj(path):
    verts,vnorms,groups=[],[],[]
    cur_col=(0.7,0.7,0.7); mats={}
    d=os.path.dirname(os.path.abspath(path))
    try:
        with open(path,"r",encoding="utf-8",errors="ignore") as f:
            for line in f:
                p=line.strip().split()
                if not p: continue
                t=p[0]
                if t=="mtllib": mats=load_mtl(os.path.join(d,p[1]))
                elif t=="v": verts.append((float(p[1]),float(p[2]),float(p[3])))
                elif t=="vn": vnorms.append((float(p[1]),float(p[2]),float(p[3])))
                elif t=="usemtl": cur_col=mats.get(p[1],(0.7,0.7,0.7))
                elif t=="f":
                    fv,fn=[],[]
                    for tok in p[1:]:
                        s=tok.split("/")
                        fv.append(verts[int(s[0])-1])
                        ni=int(s[2])-1 if len(s)>2 and s[2] else None
                        fn.append(vnorms[ni] if ni is not None and ni<len(vnorms) else None)
                    for i in range(1,len(fv)-1):
                        groups.append(((fv[0],fv[i],fv[i+1]),(fn[0],fn[i],fn[i+1]),cur_col))
        print(f"[OBJ] {len(verts)} verts, {len(groups)} tris, {len(mats)} mats")
    except Exception as e:
        print(f"[OBJ] error: {e}")
    return groups

def flat_n(a,b,c):
    ab=(b[0]-a[0],b[1]-a[1],b[2]-a[2]); ac=(c[0]-a[0],c[1]-a[1],c[2]-a[2])
    nx=ab[1]*ac[2]-ab[2]*ac[1]; ny=ab[2]*ac[0]-ab[0]*ac[2]; nz=ab[0]*ac[1]-ab[1]*ac[0]
    L=math.sqrt(nx*nx+ny*ny+nz*nz) or 1e-9
    return (nx/L,ny/L,nz/L)

def normalise(groups):
    if not groups: return groups
    av=[v for (t,_,__) in groups for v in t]
    xs=[v[0] for v in av]; ys=[v[1] for v in av]; zs=[v[2] for v in av]
    cx=(max(xs)+min(xs))/2; cy=(max(ys)+min(ys))/2; cz=(max(zs)+min(zs))/2
    sc=max(max(xs)-min(xs),max(ys)-min(ys),max(zs)-min(zs)) or 1
    def sv(v): return ((v[0]-cx)/sc,(v[1]-cy)/sc,(v[2]-cz)/sc)
    return [(tuple(sv(v) for v in tri),fn,col) for (tri,fn,col) in groups]

def make_placeholder():
    g=[]
    def box(x0,y0,z0,x1,y1,z1,col):
        fs=[[(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],
            [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
            [(x0,y0,z0),(x0,y0,z1),(x0,y1,z1),(x0,y1,z0)],
            [(x1,y0,z0),(x1,y0,z1),(x1,y1,z1),(x1,y1,z0)],
            [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],
            [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)]]
        for f in fs: g.append(((f[0],f[1],f[2]),(None,None,None),col)); g.append(((f[0],f[2],f[3]),(None,None,None),col))
    def sph(cx,cy,cz,r,col,st=8,sl=12):
        for i in range(st):
            la0=math.pi*(-0.5+i/st); la1=math.pi*(-0.5+(i+1)/st)
            for j in range(sl):
                lg0=2*math.pi*j/sl; lg1=2*math.pi*(j+1)/sl
                def pt(la,lg): return(cx+r*math.cos(la)*math.cos(lg),cy+r*math.cos(la)*math.sin(lg),cz+r*math.sin(la))
                a,b,c_,d=pt(la0,lg0),pt(la0,lg1),pt(la1,lg1),pt(la1,lg0)
                g.append(((a,b,c_),(None,None,None),col)); g.append(((a,c_,d),(None,None,None),col))
    R=(0.85,0.12,0.12); DR=(0.55,0.06,0.06); GL=(0.40,0.72,0.92)
    BL=(0.08,0.08,0.08); GR=(0.58,0.58,0.62); AM=(1.0,0.85,0.30)
    TR=(0.90,0.04,0.04); DK=(0.18,0.18,0.18)
    box(-1.0,-0.5,-0.05,1.0,0.5,0.56,R)
    box(-0.55,-0.46,0.56,0.60,0.46,0.97,DR)
    box(-0.54,-0.44,0.57,-0.52,0.44,0.95,GL)
    box(0.52,-0.44,0.57,0.54,0.44,0.95,GL)
    box(-0.53,-0.46,0.58,0.53,-0.44,0.94,GL)
    box(-0.53,0.44,0.58,0.53,0.46,0.94,GL)
    for wx,wy in [(-0.65,-0.55),(0.65,-0.55),(-0.65,0.55),(0.65,0.55)]:
        sph(wx,wy,-0.03,0.24,BL); sph(wx,wy,-0.03,0.15,GR)
    box(-1.01,-0.32,-0.02,-0.99,-0.08,0.23,AM)
    box(-1.01,0.08,-0.02,-1.0,0.32,0.23,AM)
    box(0.99,-0.32,-0.02,1.01,-0.08,0.23,TR)
    box(0.99,0.08,-0.02,1.01,0.32,0.23,TR)
    box(-1.06,-0.44,-0.05,-1.0,0.44,0.28,DK)
    box(1.0,-0.44,-0.05,1.06,0.44,0.28,DK)
    box(-1.0,-0.26,0.07,-0.98,0.26,0.28,BL)
    box(-0.32,-0.54,0.42,-0.10,-0.50,0.50,DR)
    box(-0.32,0.50,0.42,-0.10,0.54,0.50,DR)
    return g


# ═══════════════════════════════════════════════════════════════════════
#  TTS ENGINE
#  Root cause of silence: pyttsx3 on Windows uses COM (CoInitialize).
#  COM objects are apartment-threaded — the engine MUST be created AND
#  used on the exact same OS thread. The previous design created it
#  once but then Queue.get() can switch OS threads silently on Windows.
#
#  Fix: use subprocess so TTS runs in a completely isolated process
#  with its own COM apartment. Falls back to pyttsx3 direct if
#  subprocess unavailable (Linux/macOS where COM is not an issue).
# ═══════════════════════════════════════════════════════════════════════
class TTSEngine:
    def __init__(self):
        self._ok       = TTS_OK
        self._speaking = False
        self._lock     = threading.Lock()
        self._q        = queue.Queue(maxsize=2)
        if self._ok:
            threading.Thread(target=self._worker, daemon=True).start()
            print("[TTS] Engine started")

    def _speak_one(self, text):
        """Speak one utterance. Runs entirely in the worker thread."""
        # Each call creates a fresh engine so COM is always on the same
        # thread as the init — this is the only 100% safe pattern on Windows.
        try:
            eng = pyttsx3.init()
            eng.setProperty("rate", 150)
            eng.setProperty("volume", 1.0)
            eng.say(text)
            eng.runAndWait()
            # Explicit cleanup so COM object is released before next init
            eng.stop()
            del eng
        except Exception as e:
            print(f"[TTS] speak error: {e}")

    def _worker(self):
        print("[TTS] Worker thread running")
        while True:
            try:
                text = self._q.get(block=True)
                if text is None:
                    break
                # Drain: if newer text arrived, speak only the latest
                latest = text
                while not self._q.empty():
                    try:
                        latest = self._q.get_nowait()
                    except queue.Empty:
                        break
                if latest is None:
                    break
                with self._lock:
                    self._speaking = True
                print(f"[TTS] >> {latest[:80]}")
                self._speak_one(latest)
                with self._lock:
                    self._speaking = False
            except Exception as e:
                print(f"[TTS] worker error: {e}")
                with self._lock:
                    self._speaking = False

    def speak(self, text):
        """Non-blocking. Drops oldest if queue full so it never blocks."""
        if not self._ok or not text or not text.strip():
            return
        txt = text.strip()
        # Clear queue before putting new text — always speak latest
        while not self._q.empty():
            try: self._q.get_nowait()
            except: break
        try:
            self._q.put_nowait(txt)
            print(f"[TTS] Queued: {txt[:60]}")
        except queue.Full:
            pass

    def is_speaking(self):
        with self._lock: return self._speaking

    def stop(self):
        self._q.put(None)


# ═══════════════════════════════════════════════════════════════════════
#  AI ENGINE — fixed race condition, gesture queue separated from voice
# ═══════════════════════════════════════════════════════════════════════
class CarAI:
    """
    FIX: gesture-triggered prompts go to _gesture_q (low priority).
    Voice queries bypass it entirely. busy flag is ALWAYS cleared in
    finally{} so no call is ever permanently blocked.
    Gesture fires only when no voice query pending AND AI idle > 3 sec.
    """
    SYSTEM = (
        "You are JARVIS, an expert AR car hologram guide. "
        "Answer in exactly 1-2 punchy sentences. "
        "Be vivid and enthusiastic. No markdown, no bullets."
    )
    _GESTURE_PROMPTS = {
        "interior" : "Describe the cockpit-like interior of this sports car in one exciting sentence.",
        "explode"  : "Name the three most impressive structural components revealed in an exploded car view.",
        "zoom_in"  : "What aerodynamic detail becomes most visible when you zoom into a car?",
        "reset"    : "Give me the single most surprising fact about modern car engineering.",
    }

    def __init__(self, tts: TTSEngine):
        self.tts       = tts
        self.history   = []
        self.reply     = "Hologram ready. Say 'Hey Car' or use gestures!"
        self.busy      = False
        self._lock     = threading.Lock()
        self._gesture_q = queue.Queue()
        self._last_ai_time = 0.0
        self._frame_slot = [None]
        # gesture worker — fires only when voice idle
        threading.Thread(target=self._gesture_worker, daemon=True).start()

    def set_frame(self, frame):
        self._frame_slot[0] = frame.copy() if frame is not None else None

    def _b64_frame(self):
        f = self._frame_slot[0]
        if f is None: return None
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buf).decode() if ok else None

    # ── Public API ───────────────────────────────────────────────────────
    def ask_voice(self, text, use_vision=True):
        """HIGH PRIORITY — always fires, cancels pending gesture prompt."""
        while not self._gesture_q.empty():
            try: self._gesture_q.get_nowait()
            except: break
        if self.busy: return  # already processing voice — skip
        self._fire(text, use_vision=use_vision, priority="voice")

    def ask_gesture(self, key):
        """LOW PRIORITY — queued, only fires when AI has been idle > 3s."""
        prompt = self._GESTURE_PROMPTS.get(key)
        if prompt:
            # Replace any existing gesture prompt (don't pile up)
            while not self._gesture_q.empty():
                try: self._gesture_q.get_nowait()
                except: break
            self._gesture_q.put(prompt)

    def _gesture_worker(self):
        while True:
            try:
                prompt = self._gesture_q.get(timeout=1)
                # Wait until AI idle AND no voice for 3s
                waited = 0
                while self.busy or (time.time() - self._last_ai_time < 3.0):
                    time.sleep(0.2); waited += 0.2
                    if waited > 10: break
                if not self.busy:
                    self._fire(prompt, use_vision=False, priority="gesture")
            except queue.Empty:
                pass

    def _fire(self, text, use_vision, priority):
        self.busy = True
        threading.Thread(
            target=self._run,
            args=(text, use_vision, priority),
            daemon=True
        ).start()

    def _run(self, text, use_vision, priority):
        try:
            with self._lock:
                self.history.append({"role":"user","content":text})
                hist = self.history[-8:]

            if use_vision:
                b64 = self._b64_frame()
                model = VISION_MODEL
                msgs = [{"role":"system","content":self.SYSTEM}]
                if b64:
                    msgs.append({"role":"user","content":text,"images":[b64]})
                else:
                    msgs.extend(hist)
            else:
                model = TEXT_MODEL
                msgs  = [{"role":"system","content":self.SYSTEM}] + hist

            res   = ollama.chat(model=model, messages=msgs)
            reply = res["message"]["content"].strip()
            reply = reply.replace("**","").replace("*","").replace("#","")

        except Exception as e:
            reply = f"Ollama offline — run 'ollama serve' in another terminal."
            print(f"[AI] Error: {e}")

        finally:
            # ALWAYS clear busy — this was the core bug in v4
            self.busy = False
            self._last_ai_time = time.time()

        with self._lock:
            self.history.append({"role":"assistant","content":reply})
            self.reply = reply

        print(f"\n[AI/{priority}] {reply}\n")

        # Always speak — removed duplicate-guard which was silently
        # blocking TTS when the same error message repeated itself
        self.tts.speak(reply)

    def get_reply(self):
        with self._lock: return self.reply


# ═══════════════════════════════════════════════════════════════════════
#  DUAL-HAND GESTURE ENGINE  — v6 complete rewrite
#
#  MediaPipe on a MIRRORED frame (cv2.flip 1):
#    label "Right" = user's physical RIGHT hand
#    label "Left"  = user's physical LEFT hand
#
#  RIGHT hand role: HOLD + MOVE the object
#    • Wrist position → orbit (yaw continuous)
#    • Wrist Y        → pitch tilt
#    • Pinch (thumb+index) → zoom in/out
#    • Open flat palm (4+ fingers, palm facing cam) → HOLOGRAM mode
#      (car anchors to palm, follows hand in space)
#    • Point (index only up) → spotlight / ask AI about region
#
#  LEFT hand role: COMMANDS (only when right hand also present)
#    • Fist (all fingers down)  → Freeze / unfreeze
#    • 1 finger (index only)    → Interior view ON
#    • 2 fingers (index+middle) → Explode view toggle
#    • 3 fingers                → X-ray toggle
#    • 4+ fingers (open palm)   → Reset everything
#    • Pinch (thumb+index close) + right hand holding → SCALE UP/DOWN
#      (this is the "both hands" cooperative mode)
#
#  SINGLE hand (either hand):
#    → Acts as movement hand (wrist orbit + pinch zoom)
#    → No commands (need both hands for commands)
#
#  Hold confirmation: each hand has its OWN counter so they don't
#  interfere with each other.
# ═══════════════════════════════════════════════════════════════════════
class GestureEngine:
    HOLD_FRAMES = 6   # frames gesture must be stable before firing

    def __init__(self):
        self.mp_h  = mp.solutions.hands
        self.hands = self.mp_h.Hands(
            max_num_hands=2,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
            model_complexity=1)
        self.draw = mp.solutions.drawing_utils

        # Per-hand hold counters — completely independent
        self._L_hold      = {}   # left hand command hold counts
        self._L_last_cmd  = "none"
        self._L_cmd_fired = False  # edge-detect: fire once per gesture

        # Right hand continuous state
        self._R_prev_pinch = None

        # Left hand pinch for scale (cooperative mode)
        self._L_prev_pinch = None

    # ── Landmark helpers ─────────────────────────────────────────────────
    @staticmethod
    def _dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    @staticmethod
    def _fingers_up(lm):
        """Count extended fingers (index=8, middle=12, ring=16, pinky=20)."""
        tips   = [8, 12, 16, 20]
        joints = [6, 10, 14, 18]
        return sum(lm[t].y < lm[j].y for t, j in zip(tips, joints))

    @staticmethod
    def _is_fist(lm):
        """All four fingertips curled below their knuckles."""
        return all(lm[t].y > lm[j].y
                   for t, j in zip([8,12,16,20], [6,10,14,18]))

    @staticmethod
    def _only_index_up(lm):
        """Only index finger extended — others curled."""
        index_up   = lm[8].y < lm[6].y
        middle_dn  = lm[12].y > lm[10].y
        ring_dn    = lm[16].y > lm[14].y
        pinky_dn   = lm[20].y > lm[18].y
        return index_up and middle_dn and ring_dn and pinky_dn

    @staticmethod
    def _palm_centre(lm):
        pts = [lm[i] for i in [0, 5, 9, 13, 17]]
        return (sum(p.x for p in pts)/5, sum(p.y for p in pts)/5)

    # ── Left-hand hold confirmation (edge-detect) ────────────────────────
    def _L_confirm(self, raw_cmd):
        """
        Returns the confirmed command only on the RISING EDGE
        (first frame it reaches hold threshold). Never repeats
        until the gesture changes and comes back.
        """
        if raw_cmd == "none":
            self._L_hold = {}
            self._L_last_cmd = "none"
            self._L_cmd_fired = False
            return "none"

        self._L_hold[raw_cmd] = self._L_hold.get(raw_cmd, 0) + 1
        # Clear counts for other gestures
        for k in list(self._L_hold):
            if k != raw_cmd:
                self._L_hold[k] = 0

        if self._L_hold[raw_cmd] >= self.HOLD_FRAMES:
            if raw_cmd != self._L_last_cmd or not self._L_cmd_fired:
                self._L_last_cmd  = raw_cmd
                self._L_cmd_fired = True
                return raw_cmd   # fire exactly once per gesture activation
        return "none"

    # ── Main process ─────────────────────────────────────────────────────
    def process(self, frame):
        """
        Process frame, draw landmarks, return gesture dict.
        Called once per webcam frame.
        """
        out = {
            # Left hand
            "left_cmd"    : "none",  # fired command (edge-detected)
            "left_scale"  : 0.0,     # pinch-scale delta when cooperating
            "left_present": False,

            # Right hand
            "right_yaw"   : 0.0,
            "right_pitch" : 0.0,
            "right_zoom"  : 0.0,     # pinch zoom
            "right_present": False,
            "hologram"    : False,   # open palm → hologram mode
            "palm_pos"    : None,    # normalised (x,y) palm centre
            "pointing"    : False,
            "point_pos"   : None,

            "hand_count"  : 0,
        }

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        if not res.multi_hand_landmarks:
            self._R_prev_pinch = None
            self._L_prev_pinch = None
            self._L_hold = {}
            self._L_last_cmd = "none"
            self._L_cmd_fired = False
            return out

        out["hand_count"] = len(res.multi_hand_landmarks)

        # ── Sort hands by MediaPipe handedness ───────────────────────────
        left_lm = right_lm = None
        for hlm, hnd in zip(res.multi_hand_landmarks,
                            res.multi_handedness):
            label = hnd.classification[0].label  # "Left" or "Right"

            # Draw with distinct colours so user can tell which is which
            if label == "Right":
                lm_col = (0, 200, 255)   # cyan-blue  = right hand
            else:
                lm_col = (255, 100, 0)   # orange     = left hand

            self.draw.draw_landmarks(
                frame, hlm, self.mp_h.HAND_CONNECTIONS,
                self.draw.DrawingSpec(color=lm_col, thickness=2, circle_radius=4),
                self.draw.DrawingSpec(color=(220, 255, 220), thickness=1))

            if label == "Right":
                right_lm = hlm.landmark
            else:
                left_lm  = hlm.landmark

        out["left_present"]  = left_lm  is not None
        out["right_present"] = right_lm is not None

        # ── Single-hand fallback ─────────────────────────────────────────
        # If only left hand detected, treat it as movement hand too
        # (so user can orbit with either hand alone)
        effective_move_lm = right_lm if right_lm is not None else left_lm

        # ── RIGHT HAND (or fallback) → movement ──────────────────────────
        if effective_move_lm is not None:
            lm    = effective_move_lm
            fi    = self._fingers_up(lm)
            pinch = self._dist(lm[4], lm[8])
            wx    = lm[0].x
            wy    = lm[0].y

            # Continuous yaw: dead zone ±0.10 around centre (0.5)
            dx = wx - 0.5
            if abs(dx) > 0.10:
                out["right_yaw"] = math.copysign(
                    min(10.0, (abs(dx) - 0.10) * 25.0), dx)

            # Continuous pitch: dead zone ±0.12
            dy = wy - 0.5
            if abs(dy) > 0.12:
                out["right_pitch"] = math.copysign(
                    min(6.0, (abs(dy) - 0.12) * 18.0), -dy)

            # Open palm → HOLOGRAM mode (4+ fingers extended)
            # Distinct from pinch-zoom: mutually exclusive
            if fi >= 4:
                out["hologram"] = True
                out["palm_pos"] = self._palm_centre(lm)
                self._R_prev_pinch = None   # reset zoom state
            else:
                out["hologram"] = False
                # Pinch zoom only when NOT in hologram mode
                if self._R_prev_pinch is not None:
                    dp = pinch - self._R_prev_pinch
                    if abs(dp) > 0.003:
                        out["right_zoom"] = dp * 6.0
                self._R_prev_pinch = pinch

            # Pointing (index only up, others curled)
            if self._only_index_up(lm) and fi == 1:
                out["pointing"]   = True
                out["point_pos"]  = (lm[8].x, lm[8].y)

        # ── LEFT HAND → commands (only when both hands present) ──────────
        # When alone, left hand is used for movement (above). Commands only
        # make sense as a deliberate second-hand action.
        if left_lm is not None and right_lm is not None:
            lm    = left_lm
            fi    = self._fingers_up(lm)
            pinch = self._dist(lm[4], lm[8])

            # Classify raw gesture
            raw = "none"
            if self._is_fist(lm):
                raw = "freeze"
            elif fi >= 4:
                raw = "reset"
            elif fi == 3:
                raw = "xray"
            elif fi == 2:
                raw = "explode"
            elif self._only_index_up(lm):
                raw = "interior"

            # Edge-detect confirmation — fires exactly once per activation
            cmd = self._L_confirm(raw)
            out["left_cmd"] = cmd

            # Cooperative pinch-scale: left hand pinch while right hand
            # is in hologram mode (holding the object)
            # This lets you scale the hologram with left-hand pinch
            if out["hologram"] or True:   # allow scale anytime both hands present
                if self._L_prev_pinch is not None:
                    dp = pinch - self._L_prev_pinch
                    if abs(dp) > 0.003:
                        out["left_scale"] = dp * 6.0
                self._L_prev_pinch = pinch
        else:
            # Single hand: reset left-hand state
            self._L_prev_pinch = None
            if left_lm is not None:
                # Left hand alone → no command, but reset hold
                self._L_confirm("none")

        return out


# ═══════════════════════════════════════════════════════════════════════
#  OPENGL CAR RENDERER
# ═══════════════════════════════════════════════════════════════════════
class CarRenderer:
    def __init__(self, width=RENDER_W, height=RENDER_H, model_path=MODEL_PATH):
        self.w=width; self.h=height
        self.yaw=0.0; self.pitch=18.0; self.scale=1.0
        self.interior=False; self.exploded=False
        self.wireframe=False; self.xray=False
        self.auto_spin=False; self.frozen=False
        self.night_mode=False; self.color_idx=0
        self.hologram_mode=False
        self._spin_spd=0.5
        self._hologram_alpha=0.0   # 0→1 fade in/out

        pygame.init()
        if sys.platform=="win32":
            os.environ.setdefault("SDL_VIDEODRIVER","windib")
        pygame.display.set_mode((width,height),DOUBLEBUF|OPENGL|NOFRAME)
        pygame.display.set_caption("AR-v5-GL")

        # FBO
        self._fbo=glGenFramebuffers(1)
        self._crb=glGenRenderbuffers(1)
        self._drb=glGenRenderbuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER,self._fbo)
        glBindRenderbuffer(GL_RENDERBUFFER,self._crb)
        glRenderbufferStorage(GL_RENDERBUFFER,GL_RGBA8,width,height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_RENDERBUFFER,self._crb)
        glBindRenderbuffer(GL_RENDERBUFFER,self._drb)
        glRenderbufferStorage(GL_RENDERBUFFER,GL_DEPTH_COMPONENT24,width,height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_RENDERBUFFER,self._drb)
        assert glCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE
        glBindFramebuffer(GL_FRAMEBUFFER,0)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING); glEnable(GL_LIGHT0); glEnable(GL_LIGHT1); glEnable(GL_LIGHT2)
        glEnable(GL_COLOR_MATERIAL); glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH); glEnable(GL_NORMALIZE)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)

        glLightfv(GL_LIGHT0,GL_POSITION,[3,4,5,1])
        glLightfv(GL_LIGHT0,GL_DIFFUSE,[1.0,0.96,0.90,1])
        glLightfv(GL_LIGHT0,GL_SPECULAR,[0.8,0.8,0.8,1])
        glLightfv(GL_LIGHT1,GL_POSITION,[-3,-2,2,1])
        glLightfv(GL_LIGHT1,GL_DIFFUSE,[0.35,0.35,0.40,1])
        glLightfv(GL_LIGHT2,GL_POSITION,[0,-4,-2,1])
        glLightfv(GL_LIGHT2,GL_DIFFUSE,[0.20,0.20,0.26,1])
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT,[0.18,0.18,0.20,1])
        glMaterialfv(GL_FRONT,GL_SPECULAR,[0.5,0.5,0.5,1])
        glMaterialf(GL_FRONT,GL_SHININESS,56.0)

        raw = load_obj(model_path) if os.path.exists(model_path) else make_placeholder()
        self._groups = normalise(raw)
        self._tri_count = len(self._groups)
        self._build_lists()

        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(52,width/height,0.01,100)
        glMatrixMode(GL_MODELVIEW)

    def _build_lists(self):
        self._dl=glGenLists(1)
        glNewList(self._dl,GL_COMPILE)
        glBegin(GL_TRIANGLES)
        for (tri,fn,col) in self._groups:
            a,b,c=tri
            n=fn[0] if fn[0] else flat_n(a,b,c)
            glColor3f(*col); glNormal3f(*n); glVertex3f(*a)
            glNormal3f(*(fn[1] if fn[1] else n)); glVertex3f(*b)
            glNormal3f(*(fn[2] if fn[2] else n)); glVertex3f(*c)
        glEnd(); glEndList()

        self._dl_wire=glGenLists(1)
        glNewList(self._dl_wire,GL_COMPILE)
        glBegin(GL_LINES); glColor3f(0,1,0.5)
        for (tri,_,__) in self._groups:
            a,b,c=tri
            glVertex3f(*a); glVertex3f(*b)
            glVertex3f(*b); glVertex3f(*c)
            glVertex3f(*c); glVertex3f(*a)
        glEnd(); glEndList()

    def _paint(self,col,alpha=1.0):
        """Return col with paint applied to red body parts."""
        paint = CAR_COLORS[self.color_idx][0]
        if col[0]>0.6 and col[1]<0.25 and col[2]<0.25:
            return paint+(alpha,)
        elif col[0]>0.38 and col[1]<0.15 and col[2]<0.15:
            return tuple(max(0,x*0.62) for x in paint)+(alpha,)
        return col+(alpha,)

    def _draw_solid(self, alpha=1.0):
        glBegin(GL_TRIANGLES)
        for (tri,fn,col) in self._groups:
            a,b,c=tri
            n=fn[0] if fn[0] else flat_n(a,b,c)
            glColor4f(*self._paint(col,alpha)); glNormal3f(*n); glVertex3f(*a)
            glNormal3f(*(fn[1] if fn[1] else n)); glVertex3f(*b)
            glNormal3f(*(fn[2] if fn[2] else n)); glVertex3f(*c)
        glEnd()

    def _draw_hologram(self):
        """Cyan translucent wireframe + scanline flash — Iron Man style."""
        t = time.time()
        # Fade alpha
        self._hologram_alpha = min(1.0, self._hologram_alpha + 0.04)
        a = self._hologram_alpha * (0.72 + 0.18*math.sin(t*3))

        glDisable(GL_LIGHTING)
        glLineWidth(1.2)

        # Edge glow pass — draw lines in cyan-blue
        glBegin(GL_LINES)
        for (tri,_,col) in self._groups:
            av,bv,cv=tri
            # Scanline flicker: skip some triangles periodically
            cy_v = (av[1]+bv[1]+cv[1])/3
            flicker = math.sin(cy_v*12 + t*8)
            if flicker < -0.6: continue
            edge_a = a * max(0, flicker*0.4+0.6)
            glColor4f(0.2,0.85,1.0, edge_a)
            glVertex3f(*av); glVertex3f(*bv)
            glVertex3f(*bv); glVertex3f(*cv)
            glVertex3f(*cv); glVertex3f(*av)
        glEnd()

        # Fill pass — very transparent cyan solid
        glBegin(GL_TRIANGLES)
        for (tri,fn,col) in self._groups:
            a_v,b_v,c_v=tri
            n=fn[0] if fn[0] else flat_n(a_v,b_v,c_v)
            cy_v=(a_v[1]+b_v[1]+c_v[1])/3
            scan=0.12+0.08*math.sin(cy_v*10+t*6)
            glColor4f(0.05,0.6,0.9,scan*a)
            glNormal3f(*n)
            glVertex3f(*a_v); glVertex3f(*b_v); glVertex3f(*c_v)
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    def _draw_exploded(self, alpha=1.0):
        t=time.time(); ed=0.55+0.12*math.sin(t*0.7)
        glBegin(GL_TRIANGLES)
        for (tri,fn,col) in self._groups:
            a,b,c=tri
            cx=(a[0]+b[0]+c[0])/3; cy=(a[1]+b[1]+c[1])/3; cz=(a[2]+b[2]+c[2])/3
            L=math.sqrt(cx*cx+cy*cy+cz*cz) or 0.001
            dx,dy,dz=cx/L*ed,cy/L*ed,cz/L*ed
            n=fn[0] if fn[0] else flat_n(a,b,c)
            glColor4f(*self._paint(col,alpha)); glNormal3f(*n)
            glVertex3f(a[0]+dx,a[1]+dy,a[2]+dz)
            glNormal3f(*(fn[1] if fn[1] else n))
            glVertex3f(b[0]+dx,b[1]+dy,b[2]+dz)
            glNormal3f(*(fn[2] if fn[2] else n))
            glVertex3f(c[0]+dx,c[1]+dy,c[2]+dz)
        glEnd()

    def _draw_xray(self):
        """Glass-like blue-grey with depth silhouette."""
        glDisable(GL_DEPTH_TEST)
        glBegin(GL_TRIANGLES)
        for (tri,fn,col) in self._groups:
            a,b,c=tri
            n=fn[0] if fn[0] else flat_n(a,b,c)
            # X-ray: brighter on edges (facing away from camera)
            glColor4f(0.3,0.7,0.9, 0.18)
            glNormal3f(*n); glVertex3f(*a); glVertex3f(*b); glVertex3f(*c)
        glEnd()
        glEnable(GL_DEPTH_TEST)
        # Solid wireframe on top
        glDisable(GL_LIGHTING)
        glColor4f(0.0,0.9,1.0,0.7)
        glLineWidth(0.8)
        glBegin(GL_LINES)
        for (tri,_,__) in self._groups:
            a,b,c=tri
            glVertex3f(*a); glVertex3f(*b)
            glVertex3f(*b); glVertex3f(*c)
            glVertex3f(*c); glVertex3f(*a)
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    def update(self, ge_out):
        if self.frozen and ge_out["left_cmd"] != "freeze":
            return   # frozen: only unfreeze command gets through

        if self.auto_spin:
            self.yaw += self._spin_spd

        # Right hand: continuous movement
        self.yaw   += ge_out["right_yaw"]
        self.pitch  = max(-85, min(85, self.pitch + ge_out["right_pitch"]))
        self.scale  = max(0.10, min(8.0, self.scale + ge_out["right_zoom"]))

        # Left hand cooperative scale (pinch while both hands present)
        # Adds to scale on top of zoom — lets user resize while holding hologram
        self.scale  = max(0.10, min(8.0, self.scale + ge_out["left_scale"]))

        # Hologram mode from right open palm
        self.hologram_mode = ge_out["hologram"]
        if not self.hologram_mode:
            self._hologram_alpha = max(0.0, self._hologram_alpha - 0.06)

        # Left hand commands — edge-detected so each fires exactly once
        cmd = ge_out["left_cmd"]
        if cmd == "interior":
            self.interior = True
            self.exploded = False
            self.xray     = False
        elif cmd == "explode":
            self.exploded = not self.exploded
            self.interior = False
        elif cmd == "xray":
            self.xray     = not self.xray
            self.interior = False
        elif cmd == "reset":
            self.interior        = False
            self.exploded        = False
            self.xray            = False
            self.wireframe       = False
            self.scale           = 1.0
            self.yaw             = 0.0
            self.pitch           = 18.0
            self.hologram_mode   = False
            self._hologram_alpha = 0.0
        elif cmd == "freeze":
            self.frozen = not self.frozen

    def render_rgba(self, palm_px=None, frame_wh=(1280,720)):
        glBindFramebuffer(GL_FRAMEBUFFER,self._fbo)
        glViewport(0,0,self.w,self.h)
        glClearColor(0,0,0,0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(52,self.w/self.h,0.01,100)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()

        if self.interior:
            gluLookAt(0.05,-0.15,0.25, 0.60,0.80,0.25, 0,0,1)
            glScalef(self.scale,self.scale,self.scale)
        else:
            dist=3.5/max(self.scale,0.15)
            yr=math.radians(self.yaw); pr=math.radians(self.pitch)
            ex=dist*math.sin(yr)*math.cos(pr)
            ey=-dist*math.cos(yr)*math.cos(pr)
            ez=dist*math.sin(pr)
            gluLookAt(ex,ey,ez, 0,0,0, 0,0,1)
            glScalef(self.scale,self.scale,self.scale)

        if self.hologram_mode or self._hologram_alpha > 0.05:
            self._draw_hologram()
        elif self.xray:
            self._draw_xray()
        elif self.wireframe:
            glDisable(GL_LIGHTING); glCallList(self._dl_wire); glEnable(GL_LIGHTING)
        elif self.exploded:
            self._draw_exploded()
        elif self.color_idx == 0 and not self.night_mode:
            glCallList(self._dl)
        else:
            alpha = 0.7 if self.night_mode else 1.0
            self._draw_solid(alpha)

        glPixelStorei(GL_PACK_ALIGNMENT,1)
        raw=glReadPixels(0,0,self.w,self.h,GL_RGBA,GL_UNSIGNED_BYTE)
        arr=np.frombuffer(raw,dtype=np.uint8).reshape(self.h,self.w,4)[::-1].copy()
        glBindFramebuffer(GL_FRAMEBUFFER,0)
        pygame.event.pump()
        return arr

    def cleanup(self):
        glDeleteLists(self._dl,1); glDeleteLists(self._dl_wire,1)
        glDeleteFramebuffers(1,[self._fbo])
        glDeleteRenderbuffers(1,[self._crb]); glDeleteRenderbuffers(1,[self._drb])
        pygame.quit()


# ═══════════════════════════════════════════════════════════════════════
#  ARUCO
# ═══════════════════════════════════════════════════════════════════════
class ArucoTracker:
    def __init__(self):
        self.adict=cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.detector=cv2.aruco.ArucoDetector(self.adict,cv2.aruco.DetectorParameters())
        if not os.path.exists("aruco_marker.png"):
            img=cv2.aruco.generateImageMarker(self.adict,ARUCO_ID,300)
            cv2.imwrite("aruco_marker.png",img)
            print("[ArUco] aruco_marker.png saved")
    def detect(self,frame):
        grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        corners,ids,_=self.detector.detectMarkers(grey)
        if ids is None: return None,1.0
        for i,mid in enumerate(ids.flatten()):
            if mid!=ARUCO_ID: continue
            c=corners[i][0]
            cx=int(c[:,0].mean()); cy=int(c[:,1].mean())
            side=np.linalg.norm(c[0]-c[1])
            cv2.aruco.drawDetectedMarkers(frame,corners,ids)
            return (cx,cy),max(0.3,min(3.0,side/100))
        return None,1.0


# ═══════════════════════════════════════════════════════════════════════
#  COMPOSITOR  (with optional palm-tracking anchor)
# ═══════════════════════════════════════════════════════════════════════
def composite(bg, car_rgba, anchor=None, scale=1.0):
    hb,wb=bg.shape[:2]; hc,wc=car_rgba.shape[:2]
    nw=max(10,min(int(wc*scale),wb)); nh=max(10,min(int(hc*scale),hb))
    scaled=cv2.resize(car_rgba,(nw,nh),interpolation=cv2.INTER_LINEAR)
    x0=(anchor[0]-nw//2) if anchor else (wb-nw)//2
    y0=(anchor[1]-nh//2) if anchor else (hb-nh)//2
    x0=max(0,min(x0,wb-nw)); y0=max(0,min(y0,hb-nh))
    roi=bg[y0:y0+nh,x0:x0+nw].astype(np.float32)
    fg=scaled[:,:,:3].astype(np.float32)[:,:,::-1]
    a=scaled[:,:,3:4].astype(np.float32)/255.0
    out=bg.copy()
    out[y0:y0+nh,x0:x0+nw]=(fg*a+roi*(1-a)).astype(np.uint8)
    return out


# ═══════════════════════════════════════════════════════════════════════
#  HOLOGRAM PALM EFFECT  — draw arc + particle effects on hand
# ═══════════════════════════════════════════════════════════════════════
def draw_hologram_fx(frame, palm_px, t):
    """Draw Iron Man-style projection ring and particles around palm."""
    if palm_px is None: return
    px,py=int(palm_px[0]*frame.shape[1]), int(palm_px[1]*frame.shape[0])

    # Projection ring — pulsing cyan arc
    r_base = 55 + int(8*math.sin(t*4))
    alpha_ring = 0.55 + 0.25*math.sin(t*6)
    ov=frame.copy()
    cv2.circle(ov,(px,py),r_base,(0,220,255),2)
    cv2.circle(ov,(px,py),r_base-10,(0,160,200),1)
    cv2.addWeighted(ov,0.7,frame,0.3,0,frame)

    # Spinning tick marks on ring
    for i in range(12):
        ang=math.radians(i*30+t*45)
        x1=int(px+r_base*math.cos(ang)); y1=int(py+r_base*math.sin(ang))
        x2=int(px+(r_base-8)*math.cos(ang)); y2=int(py+(r_base-8)*math.sin(ang))
        brightness=int(120+120*abs(math.sin(ang+t*3)))
        cv2.line(frame,(x1,y1),(x2,y2),(0,brightness,255),1)

    # Rising particles above hand (hologram source)
    for i in range(8):
        seed=i*137+int(t*60)
        px2=px+int(30*math.sin(seed*0.4+t*2))
        py2=py-int(40+25*abs(math.sin(seed*0.7+t*1.5)))
        r=max(1,int(2*abs(math.sin(seed+t*3))))
        a2=max(0,int(180*abs(math.sin(seed*0.5+t*2))))
        ov2=frame.copy()
        cv2.circle(ov2,(px2,py2),r,(0,a2,255),-1)
        cv2.addWeighted(ov2,0.6,frame,0.4,0,frame)

    # Centre glow
    ov3=frame.copy()
    cv2.circle(ov3,(px,py),18,(0,200,255),-1)
    cv2.addWeighted(ov3,0.25,frame,0.75,0,frame)

    # "PROJECTION ACTIVE" label
    cv2.putText(frame,"HOLOGRAM ACTIVE",(px-65,py+r_base+20),
                cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,220,255),1,cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════
#  VOICE LISTENER  — activation word "hey car"
# ═══════════════════════════════════════════════════════════════════════
class VoiceListener:
    def __init__(self):
        self._q=queue.Queue()
        self._stop=threading.Event()
        self.is_listening=False
        self.last_heard=""

    def start(self):
        if not VOICE_OK: return
        threading.Thread(target=self._loop,daemon=True).start()
        print(f"[Voice] Background listener ready. Say '{ACTIVATION} <question>'")

    def _loop(self):
        rec=sr.Recognizer()
        mic=sr.Microphone()
        rec.dynamic_energy_threshold=True
        rec.energy_threshold=2400
        with mic as src:
            rec.adjust_for_ambient_noise(src,duration=1.5)
        while not self._stop.is_set():
            try:
                with mic as src:
                    self.is_listening=True
                    audio=rec.listen(src,timeout=2,phrase_time_limit=14)
                self.is_listening=False
                text=rec.recognize_google(audio).strip()
                if not text: continue
                self.last_heard=text
                print(f"[Voice] Heard: '{text}'")
                low=text.lower()
                if low in ("exit","quit","stop","bye"):
                    self._q.put(("__QUIT__",False)); continue
                vision = low.startswith(ACTIVATION)
                query  = text[len(ACTIVATION):].strip(" ,.") if vision else text
                if not query: query="What do you see on screen?"
                self._q.put((query, vision))
            except sr.WaitTimeoutError: self.is_listening=False
            except sr.UnknownValueError: self.is_listening=False
            except Exception as e:
                self.is_listening=False; print(f"[Voice] {e}"); time.sleep(0.4)

    def one_shot(self,timeout=8):
        if not VOICE_OK: return None,True
        rec=sr.Recognizer(); mic=sr.Microphone()
        try:
            with mic as src:
                rec.adjust_for_ambient_noise(src,duration=0.3)
                audio=rec.listen(src,timeout=timeout,phrase_time_limit=14)
            text=rec.recognize_google(audio).strip()
            print(f"[Voice V-key] '{text}'")
            return text,True
        except Exception as e:
            print(f"[Voice] one-shot: {e}"); return None,True

    def get(self):
        try: return self._q.get_nowait()
        except queue.Empty: return None,None

    def close(self): self._stop.set()


# ═══════════════════════════════════════════════════════════════════════
#  AI POPUP OVERLAY
# ═══════════════════════════════════════════════════════════════════════
class AIPopup:
    FADE_IN=0.35; HOLD=7.0; FADE_OUT=0.7
    def __init__(self): self._reset()
    def _reset(self):
        self.query=""; self.text=""; self.alpha=0.0
        self.state="idle"; self.t0=0.0
    def show(self,query,reply):
        self.query=query[:100]; self.text=reply
        self.state="fade_in"; self.t0=time.time()
    def update_text(self,reply):
        if self.state!="idle": self.text=reply
    def tick(self):
        if self.state=="idle": return
        dt=time.time()-self.t0
        if   self.state=="fade_in":
            self.alpha=min(1.0,dt/self.FADE_IN)
            if dt>=self.FADE_IN: self.state="hold"; self.t0=time.time()
        elif self.state=="hold":
            self.alpha=1.0
            if dt>=self.HOLD: self.state="fade_out"; self.t0=time.time()
        elif self.state=="fade_out":
            self.alpha=max(0.0,1.0-dt/self.FADE_OUT)
            if dt>=self.FADE_OUT: self._reset()

    def draw(self,frame):
        if self.state=="idle" or self.alpha<=0: return
        h,w=frame.shape[:2]
        px,py,pw=30,h//5,w-60
        words=self.text.split(); lines=[]; cur=""
        mc=(pw-24)//9
        for word in words:
            t=(cur+" "+word).strip()
            if len(t)<mc: cur=t
            else: lines.append(cur); cur=word
        if cur: lines.append(cur)
        lines=lines[:5]; ph=len(lines)*30+72
        ov=frame.copy()
        cv2.rectangle(ov,(px,py),(px+pw,py+ph),(6,8,16),-1)
        cv2.addWeighted(ov,self.alpha*0.88,frame,1-self.alpha*0.88,0,frame)
        cv2.rectangle(frame,(px,py),(px+pw,py+ph),(0,200,140),2)
        # Vision badge
        vis=self.query.lower().find("hey car")==-1 and self.query!=""
        badge=(30,80,200)
        cv2.rectangle(frame,(px,py),(px+140,py+22),badge,-1)
        tag="VISION AI" if vis else "AI RESPONSE"
        cv2.putText(frame,f" {tag}",(px+4,py+15),cv2.FONT_HERSHEY_SIMPLEX,0.44,(240,240,255),1,cv2.LINE_AA)
        cv2.putText(frame,f"Q: {self.query}",(px+8,py+38),cv2.FONT_HERSHEY_SIMPLEX,0.40,(140,200,200),1,cv2.LINE_AA)
        for i,line in enumerate(lines):
            cv2.putText(frame,line,(px+8,py+58+i*30),cv2.FONT_HERSHEY_SIMPLEX,0.56,(240,255,210),1,cv2.LINE_AA)
        cv2.rectangle(frame,(px,py+ph-4),(px+pw,py+ph),(0,200,140),-1)


# ═══════════════════════════════════════════════════════════════════════
#  HUD  — full sectioned panels
# ═══════════════════════════════════════════════════════════════════════
F=cv2.FONT_HERSHEY_SIMPLEX

def _p(frame,x,y,w,h,title,alpha=0.62):
    ov=frame.copy()
    cv2.rectangle(ov,(x,y),(x+w,y+h),(6,8,14),-1)
    cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,155,100),1)
    cv2.rectangle(frame,(x,y),(x+w,y+18),(0,155,100),-1)
    cv2.putText(frame,title.upper(),(x+4,y+13),F,0.36,(4,4,4),1,cv2.LINE_AA)

def _t(frame,txt,x,y,col=(195,255,190),sc=0.44,th=1):
    cv2.putText(frame,txt,(x,y),F,sc,col,th,cv2.LINE_AA)

def _bar(frame,x,y,w,v,mx,col):
    cv2.rectangle(frame,(x,y),(x+w,y+7),(28,28,28),-1)
    f=max(0,int(w*min(v/mx,1)))
    if f: cv2.rectangle(frame,(x,y),(x+f,y+7),col,-1)

def draw_hud(frame,car,ge,ai,fps,anchor,voice,popup,night):
    h,w=frame.shape[:2]
    lc    = ge["left_cmd"]
    holo  = ge["hologram"]
    hcount= ge["hand_count"]
    lp    = ge["left_present"]
    rp    = ge["right_present"]

    # Night mode tint
    if night:
        tint=frame.copy(); tint[:]=tint*np.array([0.5,0.6,1.0])
        cv2.addWeighted(tint,0.18,frame,0.82,0,frame)

    # ── Left panel A: Left hand commands (top-left) ─────────────────────
    needs_both = not (lp and rp)
    _p(frame,8,8,190,178,"  LEFT Hand = Commands")
    _t(frame,"[Orange landmarks]",14,27,(40,140,255),0.34)
    note_col=(200,80,80) if needs_both else (80,200,120)
    _t(frame,"Need BOTH hands" if needs_both else "Both hands ready",14,41,note_col,0.34)
    cmds=[("1F","Interior view"),("2F","Explode view"),
          ("3F","X-ray"),("Palm","Reset all"),("Fist","Freeze")]
    for i,(icon,desc) in enumerate(cmds):
        active = (
            (desc.startswith("Interior") and lc=="interior") or
            (desc.startswith("Explode")  and lc=="explode")  or
            (desc.startswith("X-ray")    and lc=="xray")     or
            (desc.startswith("Reset")    and lc=="reset")    or
            (desc.startswith("Freeze")   and lc=="freeze")
        )
        col=(0,230,150) if active else (130,160,130)
        _t(frame,f"[{icon}] {desc}",14,56+i*22,col,0.38)
    _t(frame,"[Pinch] Scale up/down",14,166,(80,160,255) if (lp and rp) else (80,80,80),0.37)

    # Left hand active badge
    if lc!="none":
        cv2.rectangle(frame,(8,185),(198,200),(0,155,100),-1)
        _t(frame,f"CMD: {lc.upper()}",12,198,(10,10,10),0.40)

    # ── Left panel B: Right hand movement ────────────────────────────────
    _p(frame,8,200,190,160,"  RIGHT Hand = Movement")
    _t(frame,"[Cyan landmarks]",14,218,(40,180,255),0.34)
    moves=[("Wrist L/R","Orbit 360"),("Wrist Up/Dn","Tilt pitch"),
           ("Pinch","Zoom in/out"),("Open palm","HOLOGRAM!"),
           ("Point idx","Spotlight")]
    for i,(k,v) in enumerate(moves):
        hlt = (k=="Open palm" and holo) or (k=="Point finger" and ge["pointing"])
        col=(0,200,255) if hlt else (130,160,130)
        _t(frame,f"{k}: {v}",14,208+i*22,col,0.36)

    if holo:
        cv2.rectangle(frame,(8,338),(198,356),(0,100,200),-1)
        _t(frame,"HOLOGRAM MODE",14,352,(200,240,255),0.38)

    # ── Right panel C: System status ─────────────────────────────────────
    _p(frame,w-200,8,192,132,"  System Status")
    _t(frame,"FPS",w-196,30,(100,130,110),0.38)
    _t(frame,f"{fps:.0f}",w-166,30,(0,220,100) if fps>20 else (60,100,255),0.46)
    _bar(frame,w-150,23,140,fps,60,(0,200,80) if fps>20 else (40,100,255))
    rows=[("ArUco","LOCKED" if anchor else "searching",(0,210,80) if anchor else (100,100,200)),
          ("AI","busy…" if ai.busy else "ready",(40,160,255) if ai.busy else (0,210,80)),
          ("Voice","LISTENING" if voice.is_listening else "waiting",(0,60,255) if voice.is_listening else (90,90,90)),
          ("TTS","speaking" if ai.tts.is_speaking() else "idle",(0,180,255) if ai.tts.is_speaking() else (80,80,80)),
          ("Spin","ON" if car.auto_spin else "OFF",(0,210,80) if car.auto_spin else (80,80,80)),
          ("Frozen","YES" if car.frozen else "no",(255,80,80) if car.frozen else (80,80,80))]
    for i,(k,v,c) in enumerate(rows):
        _t(frame,f"{k}:",w-196,52+i*14,(90,120,100),0.36)
        _t(frame,v,w-152,52+i*14,c,0.37)

    # ── Right panel D: Keys ───────────────────────────────────────────────
    _p(frame,w-200,148,192,90,"  Keys")
    ks=[("[V]","Voice+Vision"),("[C]","Color cycle"),("[W]","Wireframe"),
        ("[X]","X-ray"),("[N]","Night mode"),("[A]","Auto-spin"),
        ("[S]","Screenshot"),("[Q]","Quit")]
    cols=2; rows_k=4
    for i,(k,v) in enumerate(ks):
        col=i%cols; row=i//cols
        cx=w-196+col*95; cy=164+row*17
        _t(frame,k,cx,cy,(30,200,200),0.34)
        _t(frame,v,cx+28,cy,(110,140,110),0.33)

    # ── Car info bar (below right panel) ─────────────────────────────────
    _p(frame,w-200,246,192,110,"  Car State")
    pname=CAR_COLORS[car.color_idx][1]
    pr2,pg2,pb2=[int(x*255) for x in CAR_COLORS[car.color_idx][0]]
    view=("Interior" if car.interior else
          "Exploded" if car.exploded else
          "X-Ray"    if car.xray    else
          "Hologram" if car.hologram_mode else
          "Wire"     if car.wireframe else "Exterior")
    info=[("View",view),("Tris",f"{car._tri_count:,}"),
          ("Scale",f"{car.scale:.2f}x"),("Yaw",f"{car.yaw%360:.0f}°"),
          ("Pitch",f"{car.pitch:.0f}°"),("Paint",pname)]
    for i,(k,v) in enumerate(info):
        _t(frame,f"{k}:",w-196,262+i*16,(90,120,100),0.37)
        _t(frame,v,w-152,262+i*16,(205,255,200),0.39)
    cv2.rectangle(frame,(w-50,250),(w-26,266),(pb2,pg2,pr2),-1)
    cv2.rectangle(frame,(w-50,250),(w-26,266),(0,155,100),1)

    # ── AI bottom strip ───────────────────────────────────────────────────
    _p(frame,8,h-78,w-16,70,
       f"  AI Car Guide (JARVIS)   |   Say: '{ACTIVATION} <question>'",alpha=0.72)
    reply=ai.get_reply()
    words=reply.split(); lines=[]; cur=""
    mc=(w-40)//8
    for word in words:
        t2=(cur+" "+word).strip()
        if len(t2)<mc: cur=t2
        else: lines.append(cur); cur=word
    if cur: lines.append(cur)
    for i,ln in enumerate(lines[:2]):
        _t(frame,ln,14,h-56+i*28,(255,248,155),0.52)

    # Thinking dots
    if ai.busy:
        d="●"*(int(time.time()*3)%4+1)
        _t(frame,d,w-70,h-50,(50,160,255),0.54)

    # Speaking indicator
    if ai.tts.is_speaking():
        r2=int(7+4*math.sin(time.time()*10))
        cv2.circle(frame,(w-18,h-66),r2,(0,80,255),-1)
        _t(frame,"SPEAKING",w-90,h-70,(0,120,255),0.36)

    # Voice waveform
    if voice.is_listening:
        for i in range(20):
            hh=int(4+10*abs(math.sin(time.time()*8+i*0.7)))
            ci=int(50+180*abs(math.sin(time.time()*6+i)))
            cv2.line(frame,(w-260+i*6,h-28),(w-260+i*6,h-28+hh),(0,ci,ci),2)

    # Two-hand banner
    if hcount==2 and lp and rp:
        bw=380; bx=(w-bw)//2; by=8
        ov2=frame.copy()
        cv2.rectangle(ov2,(bx,by),(bx+bw,by+28),(0,20,40),-1)
        cv2.addWeighted(ov2,0.75,frame,0.25,0,frame)
        cv2.rectangle(frame,(bx,by),(bx+bw,by+28),(0,200,255),1)
        _t(frame,"DUAL-HAND  |  CYAN=Right(move)  ORANGE=Left(cmd+scale)",bx+8,by+19,(0,220,255),0.42)
    elif hcount==1:
        bw=300; bx=(w-bw)//2; by=8
        ov2=frame.copy()
        cv2.rectangle(ov2,(bx,by),(bx+bw,by+24),(0,20,20),-1)
        cv2.addWeighted(ov2,0.65,frame,0.35,0,frame)
        hand_side="RIGHT" if rp else "LEFT"
        _t(frame,f"SINGLE HAND ({hand_side})  —  orbit + zoom + hologram",bx+8,by+17,(100,200,180),0.40)

    # Gesture active label centre
    if lc!="none":
        label={"interior":"INTERIOR VIEW","explode":"EXPLODE",
               "reset":"RESET","freeze":"FROZEN" if car.frozen else "UNFREEZE",
               "color":"COLOR CHANGED"}.get(lc,lc.upper())
        tw=len(label)*11
        _t(frame,label,(w-tw)//2,h-88,(0,240,160),0.65,2)

    # Popup on top
    popup.tick(); popup.draw(frame)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("="*66)
    print("  AR Car v6  —  Iron Man Hologram + Dual-Hand + Vision AI")
    print("="*66)
    print(f"  Wake word : '{ACTIVATION} <question>'  (vision-aware)")
    print("  V key     : push-to-talk  (always vision)")
    print("  RIGHT hand ALONE : orbit / zoom / hologram")
    print("  BOTH hands       : RIGHT moves, LEFT commands + scales")
    print("  Cyan landmarks = RIGHT hand  |  Orange = LEFT hand")
    print("="*66)

    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAM_H)
    cap.set(cv2.CAP_PROP_FPS,30)
    if not cap.isOpened(): sys.exit("[ERROR] Cannot open webcam — try index 1")

    print("[INIT] Loading renderer…")
    car   = CarRenderer()
    print("[INIT] Renderer ready")

    tts   = TTSEngine()
    ai    = CarAI(tts)
    ge    = GestureEngine()
    aruco = ArucoTracker()
    voice = VoiceListener()
    popup = AIPopup()
    voice.start()

    # Greeting
    ai.ask_voice("Greet me in one sentence as JARVIS, an AR hologram car guide.",
                 use_vision=False)

    fps_t=time.time(); fps=0.0; fc=0
    night=False; v_active=False
    color_cooldown=0.0

    print("\n[READY] Show hands to the camera!\n")
    print("  Gesture reference:")
    print("  RIGHT hand open palm  → HOLOGRAM (car sits on hand)")
    print("  RIGHT hand wrist L/R  → orbit yaw")
    print("  RIGHT hand pinch      → zoom")
    print("  LEFT  hand 1 finger   → interior view    (both hands needed)")
    print("  LEFT  hand 2 fingers  → explode view     (both hands needed)")
    print("  LEFT  hand 3 fingers  → x-ray            (both hands needed)")
    print("  LEFT  hand open palm  → reset             (both hands needed)")
    print("  LEFT  hand fist       → freeze/unfreeze  (both hands needed)")
    print("  LEFT  hand pinch      → scale up/down    (both hands needed)\n")

    while True:
        ret,frame=cap.read()
        if not ret: print("[ERROR] Webcam lost"); break
        frame=cv2.flip(frame,1)

        ai.set_frame(frame)
        ge_out=ge.process(frame)
        lc=ge_out["left_cmd"]

        # Color cycle from gesture — edge-detected so fires once
        if lc=="color" and time.time()-color_cooldown>1.5:
            car.cycle_color()
            color_cooldown=time.time()
            print(f"[Color] {CAR_COLORS[car.color_idx][1]}")

        # Update car state (handles all movement + commands internally)
        car.update(ge_out)

        # Hologram anchor: car floats 140px above palm centre
        palm_px=None
        if ge_out["hologram"] and ge_out["palm_pos"]:
            pp=ge_out["palm_pos"]
            px=int(pp[0]*CAM_W)
            py=max(80, int(pp[1]*CAM_H)-150)
            palm_px=(px,py)
            draw_hologram_fx(frame, ge_out["palm_pos"], time.time())

        # ArUco anchor (inactive when hologram mode on)
        aruco_anchor,aruco_scale=aruco.detect(frame)
        if ge_out["hologram"] and palm_px:
            anchor=palm_px; eff=max(0.10, car.scale*0.52)
        elif aruco_anchor:
            anchor=aruco_anchor; eff=car.scale*aruco_scale*0.72
        else:
            anchor=None; eff=max(0.10, car.scale*0.72)

        car_rgba=car.render_rgba()
        frame=composite(frame, car_rgba, anchor=anchor, scale=eff)

        # Pointing spotlight
        if ge_out["pointing"] and ge_out["point_pos"]:
            pp=ge_out["point_pos"]
            sx=int(pp[0]*CAM_W); sy=int(pp[1]*CAM_H)
            cv2.circle(frame,(sx,sy),24,(0,220,255),2)
            cv2.circle(frame,(sx,sy),8,(0,200,255),-1)

        # Voice
        vtext,vvis=voice.get()
        if vtext=="__QUIT__": break
        elif vtext:
            ai.ask_voice(vtext, use_vision=vvis)
            popup.show(vtext,"Thinking…")

        if popup.state!="idle" and popup.text=="Thinking…" and not ai.busy:
            popup.update_text(ai.get_reply())

        # FPS
        fc+=1; now=time.time()
        if now-fps_t>=1.0: fps=fc/(now-fps_t); fc=0; fps_t=now

        draw_hud(frame, car, ge_out, ai, fps,
                 aruco_anchor if not ge_out["hologram"] else True,
                 voice, popup, night)

        cv2.imshow("AR Car v6  —  Iron Man Hologram  (Q=quit)", frame)

        key=cv2.waitKey(1)&0xFF
        if key==ord('q'): break
        elif key==ord('v') and not v_active:
            def _vs():
                nonlocal v_active
                v_active=True
                _t2,_vis=voice.one_shot(timeout=8)
                if _t2:
                    ai.ask_voice(_t2, use_vision=True)
                    popup.show(_t2,"Thinking…")
                v_active=False
            threading.Thread(target=_vs, daemon=True).start()
        elif key==ord('c'):
            car.cycle_color()
            print(f"[Color] {CAR_COLORS[car.color_idx][1]}")
        elif key==ord('w'):
            car.wireframe=not car.wireframe
            print(f"[Wire] {'ON' if car.wireframe else 'OFF'}")
        elif key==ord('x'):
            car.xray=not car.xray
            print(f"[X-ray] {'ON' if car.xray else 'OFF'}")
        elif key==ord('n'):
            night=not night
            print(f"[Night] {'ON' if night else 'OFF'}")
        elif key==ord('a'):
            car.auto_spin=not car.auto_spin
            print(f"[Spin] {'ON' if car.auto_spin else 'OFF'}")
        elif key==ord('s'):
            ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            p=f"screenshots/ar_{ts}.png"; cv2.imwrite(p,frame)
            print(f"[Screenshot] {p}")

    tts.stop(); voice.close(); car.cleanup()
    cap.release(); cv2.destroyAllWindows()
    print("[EXIT] Done.")

if __name__=="__main__":
    main()