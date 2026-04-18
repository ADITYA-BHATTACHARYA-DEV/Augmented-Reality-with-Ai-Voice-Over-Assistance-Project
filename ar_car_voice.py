# """
# AR Car — Voice Conversation Edition
# ====================================
# Identical pipeline to ar_car_open3d.py but with a FULL voice conversation
# loop: continuous listening, spoken AI replies, and a conversation history
# so Ollama remembers what was said earlier in the session.
#
# Extra requirements (on top of base):
#   pip install pyttsx3 speechrecognition pyaudio webrtcvad
#
# How it works
# ------------
# A background thread continuously listens for speech (with VAD silence
# detection so it doesn't stay open forever). When speech is detected:
#   1. Text is sent to Ollama with full conversation history
#   2. Ollama replies → displayed on screen AND spoken aloud via TTS
#   3. Gestures still work in parallel — completely independent threads
#
# Usage
# -----
#   python ar_car_voice.py
#
#   Just speak naturally — no key press needed.
#   Say "exit" or "quit" or press Q to stop.
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
# import queue
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
# try:
#     import speech_recognition as sr
#     import pyttsx3
#     VOICE_OK = True
# except ImportError:
#     VOICE_OK = False
#     print("[WARN] Voice deps missing — running gesture-only mode")
#
# # ── reuse all classes from ar_car_open3d ─────────────────────────────────
# # (In a real project these would be a shared module; duplicated here for
# #  standalone running)
#
# CAM_W, CAM_H = 1280, 720
# RENDER_W      = 640
# RENDER_H      = 480
# MODEL_PATH    = "car.obj"
# OLLAMA_MODEL  = "llava-phi3"
# ARUCO_DICT    = cv2.aruco.DICT_6X6_250
# ARUCO_ID      = 0
#
#
# # ── paste GestureEngine, CarRenderer, ArucoTracker from ar_car_open3d ────
# # (abbreviated here — import from the other file in practice)
#
# class GestureEngine:
#     def __init__(self):
#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(max_num_hands=2,
#             min_detection_confidence=0.75, min_tracking_confidence=0.75)
#         self.draw = mp.solutions.drawing_utils
#         self.prev_pinch = None
#
#     @staticmethod
#     def _dist(a, b): return math.hypot(a.x-b.x, a.y-b.y)
#
#     @staticmethod
#     def _fingers_up(lm):
#         return sum(lm[t].y < lm[j].y for t,j in zip([8,12,16,20],[6,10,14,18]))
#
#     def process(self, frame):
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         res = self.hands.process(rgb)
#         gesture, sd, rd = "none", 0.0, 0.0
#         if not res.multi_hand_landmarks:
#             self.prev_pinch = None
#             return gesture, sd, rd
#         for hlm in res.multi_hand_landmarks:
#             lm = hlm.landmark
#             self.draw.draw_landmarks(frame, hlm, self.mp_hands.HAND_CONNECTIONS,
#                 self.draw.DrawingSpec(color=(0,200,150),thickness=2),
#                 self.draw.DrawingSpec(color=(0,255,100),thickness=1))
#             pinch   = self._dist(lm[4], lm[8])
#             fingers = self._fingers_up(lm)
#             if self.prev_pinch is not None:
#                 delta = pinch - self.prev_pinch
#                 if abs(delta) > 0.005: sd = delta * 1.8
#             self.prev_pinch = pinch
#             if fingers >= 4:   gesture = "reset"
#             elif fingers == 2: gesture = "interior_view"
#             elif pinch < 0.06: gesture = "pinch_close"
#             elif pinch > 0.20: gesture = "pinch_open"
#             wx = lm[0].x
#             if wx < 0.30:   rd = -4.0
#             elif wx > 0.70: rd = +4.0
#         return gesture, sd, rd
#
#
# class CarRenderer:
#     def __init__(self, width=RENDER_W, height=RENDER_H, model_path=MODEL_PATH):
#         self.w = width; self.h = height
#         self.yaw = 0.0; self.scale = 1.0; self.interior = False
#         self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
#         sc = self.renderer.scene
#         sc.set_background([0,0,0,0])
#         sc.scene.set_sun_light([0.5,-1,-0.8],[1.2,1.2,1.2],80000)
#         sc.scene.enable_sun_light(True)
#         sc.scene.enable_indirect_lighting(True)
#         mat = o3d.visualization.rendering.MaterialRecord()
#         mat.shader = "defaultLit"
#         mat.base_color = [0.85,0.12,0.12,1.0]
#         mat.base_roughness = 0.35; mat.base_metallic = 0.6
#         if os.path.exists(model_path):
#             mesh = o3d.io.read_triangle_mesh(model_path)
#             mesh.compute_vertex_normals()
#         else:
#             mesh = self._placeholder()
#         bb = mesh.get_axis_aligned_bounding_box()
#         ex = bb.get_max_bound() - bb.get_min_bound()
#         mesh.translate(-bb.get_center())
#         mesh.scale(2.0/max(ex), center=[0,0,0])
#         sc.add_geometry("car", mesh, mat)
#         self._set_cam_ext()
#
#     @staticmethod
#     def _placeholder():
#         body = o3d.geometry.TriangleMesh.create_box(2,1,0.6)
#         roof = o3d.geometry.TriangleMesh.create_box(1.2,0.95,0.5)
#         wheels = [o3d.geometry.TriangleMesh.create_sphere(0.25) for _ in range(4)]
#         body.translate([-1,-0.5,0]); roof.translate([-0.6,-0.475,0.6])
#         for w,p in zip(wheels,[(-0.7,-0.6,-0.25),(0.7,-0.6,-0.25),
#                                 (-0.7,0.6,-0.25),(0.7,0.6,-0.25)]): w.translate(p)
#         m = body+roof+wheels[0]+wheels[1]+wheels[2]+wheels[3]
#         m.paint_uniform_color([0.85,0.12,0.12]); m.compute_vertex_normals()
#         return m
#
#     def _set_cam_ext(self):
#         self.renderer.setup_camera(60,[0,0,0],[0,-4.5,1.5],[0,0,1])
#
#     def _set_cam_int(self):
#         self.renderer.setup_camera(90,[0.3,0.1,0.3],[0,-0.3,0.5],[0,0,1])
#
#     def update(self, gesture, sd, rd):
#         self.scale = max(0.3, min(3.5, self.scale+sd))
#         self.yaw  += rd
#         T = np.eye(4)
#         r = math.radians(self.yaw)
#         T[0,0]=math.cos(r)*self.scale; T[0,1]=-math.sin(r)*self.scale
#         T[1,0]=math.sin(r)*self.scale; T[1,1]=math.cos(r)*self.scale
#         T[2,2]=self.scale
#         self.renderer.scene.set_geometry_transform("car", T)
#         if gesture=="interior_view" and not self.interior:
#             self._set_cam_int(); self.interior=True
#         elif gesture=="reset":
#             self._set_cam_ext(); self.interior=False
#             self.scale=1.0; self.yaw=0.0
#
#     def render_rgba(self):
#         img = self.renderer.render_to_image()
#         arr = np.asarray(img)
#         if arr.dtype!=np.uint8: arr=(arr*255).clip(0,255).astype(np.uint8)
#         grey  = arr.mean(axis=2)
#         alpha = np.where(grey<8,0,255).astype(np.uint8)
#         return np.dstack([arr,alpha])
#
#
# class ArucoTracker:
#     def __init__(self):
#         self.adict    = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
#         self.detector = cv2.aruco.ArucoDetector(self.adict, cv2.aruco.DetectorParameters())
#         if not os.path.exists("aruco_marker.png"):
#             img = cv2.aruco.generateImageMarker(self.adict, ARUCO_ID, 300)
#             cv2.imwrite("aruco_marker.png", img)
#             print("[ArUco] Marker saved → aruco_marker.png")
#
#     def detect(self, frame):
#         grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         corners, ids, _ = self.detector.detectMarkers(grey)
#         if ids is None: return None, 1.0
#         for i,mid in enumerate(ids.flatten()):
#             if mid!=ARUCO_ID: continue
#             c  = corners[i][0]
#             cx = int(c[:,0].mean()); cy = int(c[:,1].mean())
#             side = np.linalg.norm(c[0]-c[1])
#             cv2.aruco.drawDetectedMarkers(frame, corners, ids)
#             return (cx,cy), max(0.3,min(3.0,side/100))
#         return None, 1.0
#
#
# def composite(bg, car_rgba, anchor=None, scale=1.0):
#     hb,wb = bg.shape[:2]; hc,wc = car_rgba.shape[:2]
#     nw = max(10,min(int(wc*scale),wb))
#     nh = max(10,min(int(hc*scale),hb))
#     scaled = cv2.resize(car_rgba,(nw,nh))
#     x0 = (anchor[0]-nw//2 if anchor else (wb-nw)//2)
#     y0 = (anchor[1]-nh//2 if anchor else (hb-nh)//2)
#     x0=max(0,min(x0,wb-nw)); y0=max(0,min(y0,hb-nh))
#     roi = bg[y0:y0+nh,x0:x0+nw].astype(np.float32)
#     fg  = cv2.cvtColor(scaled[:,:,:3],cv2.COLOR_RGB2BGR).astype(np.float32)
#     a   = scaled[:,:,3:4]/255.0
#     out = bg.copy()
#     out[y0:y0+nh,x0:x0+nw] = (fg*a+roi*(1-a)).astype(np.uint8)
#     return out
#
#
# # ═══════════════════════════════════════════════════════════════════════════
# #  CONVERSATIONAL AI  — with history
# # ═══════════════════════════════════════════════════════════════════════════
# class ConversationalAI:
#     SYSTEM = (
#         "You are an enthusiastic AR car guide with deep automotive knowledge. "
#         "Keep every reply to 1-2 sentences maximum. "
#         "Be vivid, specific, and friendly. No markdown, no bullet points."
#     )
#
#     def __init__(self, model=OLLAMA_MODEL):
#         self.model   = model
#         self.history = []           # [{role, content}, …] growing context
#         self.reply   = "🚗  Say something to start a conversation!"
#         self.busy    = False
#         self.lock    = threading.Lock()
#         self.tts     = None
#         if VOICE_OK:
#             try:
#                 self.tts = pyttsx3.init()
#                 self.tts.setProperty("rate", 168)
#                 self.tts.setProperty("volume", 0.9)
#             except Exception as e:
#                 print(f"[TTS] Init failed: {e}")
#
#     def chat(self, user_text: str, speak_reply=True):
#         """Non-blocking — fires in background thread."""
#         if self.busy:
#             return
#         self.busy = True
#         threading.Thread(target=self._run, args=(user_text, speak_reply),
#                          daemon=True).start()
#
#     def _run(self, user_text, speak_reply):
#         with self.lock:
#             self.history.append({"role": "user", "content": user_text})
#             # keep last 10 turns to avoid token overflow
#             history_slice = self.history[-10:]
#
#         try:
#             res = ollama.chat(
#                 model=self.model,
#                 messages=[{"role":"system","content":self.SYSTEM}] + history_slice,
#             )
#             reply_text = res["message"]["content"].strip()
#         except Exception as e:
#             reply_text = f"[AI error: {e}]"
#
#         with self.lock:
#             self.history.append({"role":"assistant","content":reply_text})
#             self.reply = reply_text
#
#         print(f"\n[AI] {reply_text}\n")
#
#         if speak_reply and self.tts:
#             try:
#                 self.tts.say(reply_text)
#                 self.tts.runAndWait()
#             except Exception:
#                 pass
#
#         self.busy = False
#
#     def get_reply(self):
#         with self.lock:
#             return self.reply
#
#
# # ═══════════════════════════════════════════════════════════════════════════
# #  CONTINUOUS VOICE LISTENER
# # ═══════════════════════════════════════════════════════════════════════════
# class ContinuousListener:
#     """
#     Runs in a background thread. Puts recognised text into a queue.
#     The main loop reads from the queue without blocking.
#     """
#     def __init__(self):
#         self.q    = queue.Queue()
#         self.stop = threading.Event()
#         self._t   = None
#
#     def start(self):
#         if not VOICE_OK:
#             return
#         self._t = threading.Thread(target=self._loop, daemon=True)
#         self._t.start()
#
#     def _loop(self):
#         rec = sr.Recognizer()
#         mic = sr.Microphone()
#         rec.dynamic_energy_threshold = True
#         rec.energy_threshold          = 3000
#         with mic as src:
#             rec.adjust_for_ambient_noise(src, duration=1.0)
#         print("[Voice] Continuous listener started — just speak!")
#         while not self.stop.is_set():
#             try:
#                 with mic as src:
#                     audio = rec.listen(src, timeout=2, phrase_time_limit=10)
#                 text = rec.recognize_google(audio)
#                 if text.strip():
#                     print(f"[Voice] '{text}'")
#                     self.q.put(text)
#             except sr.WaitTimeoutError:
#                 pass   # silence — keep looping
#             except sr.UnknownValueError:
#                 pass   # could not understand
#             except Exception as e:
#                 print(f"[Voice] {e}")
#                 time.sleep(0.5)
#
#     def get_text(self):
#         """Non-blocking — returns text if available, else None."""
#         try:
#             return self.q.get_nowait()
#         except queue.Empty:
#             return None
#
#     def close(self):
#         self.stop.set()
#
#
# # ═══════════════════════════════════════════════════════════════════════════
# #  HUD
# # ═══════════════════════════════════════════════════════════════════════════
# def draw_hud_v(frame, gesture, scale, ai_text, aruco_found, fps, listening):
#     h, w = frame.shape[:2]
#     ov = frame.copy()
#     cv2.rectangle(ov, (0,0), (350,115), (0,0,0), -1)
#     cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)
#
#     def put(t, y, col=(220,255,180), s=0.62, th=1):
#         cv2.putText(frame, t, (10,y), cv2.FONT_HERSHEY_SIMPLEX, s, col, th, cv2.LINE_AA)
#
#     icon = {"none":"","pinch_close":"🤏","pinch_open":"👐",
#             "interior_view":"✌️","reset":"🖐"}.get(gesture,"")
#     put(f"Gesture : {icon} {gesture}", 26)
#     put(f"Scale   : {scale:.2f}x", 52)
#     put(f"ArUco   : {'✓ anchored' if aruco_found else '✗ free'}", 78,
#         (100,255,100) if aruco_found else (80,80,220))
#     put(f"FPS {fps:.0f}  |  Q=quit  |  speak freely", 104, (160,160,160))
#
#     # Mic indicator
#     if listening:
#         r = int(8 + 4*math.sin(time.time()*6))
#         cv2.circle(frame, (w-20, 20), r, (0,60,255), -1)
#
#     # AI caption bar
#     if ai_text:
#         words = ai_text.split()
#         lines, cur = [], ""
#         for word in words:
#             test = (cur+" "+word).strip()
#             if len(test)*9 < w-24: cur=test
#             else: lines.append(cur); cur=word
#         if cur: lines.append(cur)
#         lines = lines[:3]
#         bh = len(lines)*28+16
#         bar = frame.copy()
#         cv2.rectangle(bar,(0,h-bh),(w,h),(10,10,10),-1)
#         cv2.addWeighted(bar,0.62,frame,0.38,0,frame)
#         for i,line in enumerate(lines):
#             cv2.putText(frame, line, (12,h-bh+24+i*28),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,240,140), 1, cv2.LINE_AA)
#
#
# # ═══════════════════════════════════════════════════════════════════════════
# #  MAIN
# # ═══════════════════════════════════════════════════════════════════════════
# GESTURE_PROMPTS = {
#     "interior_view" : "Describe the interior view of a modern performance car cabin.",
#     "pinch_open"    : "What does scaling up a car's size do to its aerodynamics?",
#     "reset"         : "Give me one surprising fact about iconic car design.",
# }
#
# def main():
#     print("=" * 60)
#     print("  AR Car — Voice Conversation + Gesture Edition")
#     print("=" * 60)
#     print("  Just SPEAK to talk to the AI car guide.")
#     print("  Gestures still work simultaneously.")
#     print("  Say 'exit' or 'quit' or press Q to stop.")
#     print("=" * 60)
#
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
#     cap.set(cv2.CAP_PROP_FPS, 30)
#
#     ge     = GestureEngine()
#     cr     = CarRenderer()
#     ai     = ConversationalAI()
#     aruco  = ArucoTracker()
#     voice  = ContinuousListener()
#     voice.start()
#
#     ai.chat("Introduce yourself as the AR car guide in one sentence.", speak_reply=False)
#
#     scale        = 1.0
#     last_gesture = "none"
#     fps_timer    = time.time()
#     fps          = 0.0
#     fc           = 0
#
#     print("\n[READY] AR window open. Speak or gesture.\n")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret: break
#         frame = cv2.flip(frame, 1)
#
#         gesture, sd, rd = ge.process(frame)
#         scale = max(0.3, min(3.5, scale+sd))
#
#         anchor, aruco_scale = aruco.detect(frame)
#         eff_scale = scale * aruco_scale if anchor else scale
#
#         cr.update(gesture, sd, rd)
#         frame = composite(frame, cr.render_rgba(), anchor=anchor,
#                           scale=eff_scale*0.7)
#
#         # gesture → AI
#         if gesture != last_gesture:
#             p = GESTURE_PROMPTS.get(gesture)
#             if p: ai.chat(p, speak_reply=True)
#         last_gesture = gesture
#
#         # voice → AI
#         spoken = voice.get_text()
#         if spoken:
#             low = spoken.lower().strip()
#             if low in ("exit","quit","stop","goodbye"):
#                 break
#             ai.chat(spoken, speak_reply=True)
#
#         fc += 1
#         now = time.time()
#         if now-fps_timer >= 1.0:
#             fps=fc/(now-fps_timer); fc=0; fps_timer=now
#
#         draw_hud_v(frame, gesture, scale, ai.get_reply(),
#                    anchor is not None, fps,
#                    VOICE_OK and not voice.stop.is_set())
#
#         cv2.imshow("AR Car — Voice + Gesture AI  (Q to quit)", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     voice.close()
#     cap.release()
#     cv2.destroyAllWindows()
#     print("[EXIT] Goodbye!")
#
#
# if __name__ == "__main__":
#     main()




"""
AR Car — Voice Conversation Edition  (Windows Compatible)
==========================================================
Same fix as ar_car_open3d.py: uses PyOpenGL + pygame instead of
Open3D OffscreenRenderer (which crashes on Windows with EGL error).

Adds FULL continuous voice conversation loop:
  • Background thread listens continuously — just speak, no key press
  • Ollama remembers conversation history across turns
  • Replies spoken aloud via TTS
  • Gestures + ArUco still work simultaneously

Install
-------
  pip install opencv-python mediapipe pygame PyOpenGL PyOpenGL_accelerate
              ollama numpy speechrecognition pyttsx3
  pip install pipwin && pipwin install pyaudio   (Windows pyaudio)

Run
---
  ollama serve          ← keep open in another terminal
  python ar_car_voice.py

Say "exit" or "quit" or press Q to stop.
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
import queue

import pygame
from pygame.locals import DOUBLEBUF, OPENGL, NOFRAME
from OpenGL.GL import *
from OpenGL.GLU import *

try:
    import speech_recognition as sr
    import pyttsx3
    VOICE_OK = True
except ImportError:
    VOICE_OK = False
    print("[WARN] Voice deps missing — gesture-only mode")

# ═══════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════
CAM_W, CAM_H = 1280, 720
RENDER_W      = 640
RENDER_H      = 480
MODEL_PATH    = "car.obj"
OLLAMA_MODEL  = "llava-phi3"
ARUCO_DICT    = cv2.aruco.DICT_6X6_250
ARUCO_ID      = 0


# ═══════════════════════════════════════════════════════════════════════
#  OBJ LOADER
# ═══════════════════════════════════════════════════════════════════════
def load_obj(path):
    verts, tris = [], []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                p = line.strip().split()
                if not p: continue
                if p[0] == "v":
                    verts.append((float(p[1]), float(p[2]), float(p[3])))
                elif p[0] == "f":
                    idxs = [int(x.split("/")[0])-1 for x in p[1:]]
                    for i in range(1, len(idxs)-1):
                        tris.append((verts[idxs[0]], verts[idxs[i]], verts[idxs[i+1]]))
        print(f"[OBJ] {path}: {len(verts)} verts, {len(tris)} tris")
    except Exception as e:
        print(f"[OBJ] {e}")
    return tris


def compute_normals(tris):
    normals = []
    for (a,b,c) in tris:
        ab=(b[0]-a[0],b[1]-a[1],b[2]-a[2])
        ac=(c[0]-a[0],c[1]-a[1],c[2]-a[2])
        nx=ab[1]*ac[2]-ab[2]*ac[1]; ny=ab[2]*ac[0]-ab[0]*ac[2]; nz=ab[0]*ac[1]-ab[1]*ac[0]
        L=math.sqrt(nx*nx+ny*ny+nz*nz) or 1e-9
        normals.append((nx/L,ny/L,nz/L))
    return normals


def normalise_mesh(tris):
    if not tris: return tris
    all_v=[v for t in tris for v in t]
    xs=[v[0] for v in all_v]; ys=[v[1] for v in all_v]; zs=[v[2] for v in all_v]
    cx=(max(xs)+min(xs))/2; cy=(max(ys)+min(ys))/2; cz=(max(zs)+min(zs))/2
    sc=max(max(xs)-min(xs),max(ys)-min(ys),max(zs)-min(zs)) or 1
    def s(v): return ((v[0]-cx)/sc,(v[1]-cy)/sc,(v[2]-cz)/sc)
    return [(s(a),s(b),s(c)) for (a,b,c) in tris]


def make_placeholder_tris():
    tris=[]
    def box(x0,y0,z0,x1,y1,z1):
        faces=[[(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],
               [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
               [(x0,y0,z0),(x0,y0,z1),(x0,y1,z1),(x0,y1,z0)],
               [(x1,y0,z0),(x1,y0,z1),(x1,y1,z1),(x1,y1,z0)],
               [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],
               [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)]]
        for f in faces: tris.append((f[0],f[1],f[2])); tris.append((f[0],f[2],f[3]))
    def sphere(cx,cy,cz,r,st=6,sl=8):
        for i in range(st):
            la0=math.pi*(-0.5+i/st); la1=math.pi*(-0.5+(i+1)/st)
            for j in range(sl):
                lg0=2*math.pi*j/sl; lg1=2*math.pi*(j+1)/sl
                def pt(la,lg): return (cx+r*math.cos(la)*math.cos(lg),cy+r*math.cos(la)*math.sin(lg),cz+r*math.sin(la))
                a,b,c_,d=pt(la0,lg0),pt(la0,lg1),pt(la1,lg1),pt(la1,lg0)
                tris.append((a,b,c_)); tris.append((a,c_,d))
    box(-1,-0.5,-0.05,1,0.5,0.55); box(-0.6,-0.48,0.55,0.6,0.48,1.0)
    for wx,wy in [(-0.7,-0.55),(0.7,-0.55),(-0.7,0.55),(0.7,0.55)]: sphere(wx,wy,-0.05,0.22)
    return tris


# ═══════════════════════════════════════════════════════════════════════
#  OPENGL RENDERER
# ═══════════════════════════════════════════════════════════════════════
class CarRenderer:
    def __init__(self, width=RENDER_W, height=RENDER_H, model_path=MODEL_PATH):
        self.w=width; self.h=height
        self.yaw=0.0; self.scale=1.0; self.interior=False
        pygame.init()
        os.environ.setdefault("SDL_VIDEODRIVER","windib")
        self._surface=pygame.display.set_mode((width,height),DOUBLEBUF|OPENGL|NOFRAME)
        pygame.display.set_caption("AR-GL")

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
        glBindFramebuffer(GL_FRAMEBUFFER,0)

        glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0); glEnable(GL_LIGHT1); glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)
        glLightfv(GL_LIGHT0,GL_POSITION,[2,3,4,1])
        glLightfv(GL_LIGHT0,GL_DIFFUSE,[1,0.95,0.9,1])
        glLightfv(GL_LIGHT1,GL_POSITION,[-2,-1,2,1])
        glLightfv(GL_LIGHT1,GL_DIFFUSE,[0.3,0.3,0.35,1])
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT,[0.25,0.25,0.28,1])

        raw = load_obj(model_path) if os.path.exists(model_path) else make_placeholder_tris()
        self._tris=normalise_mesh(raw); self._normals=compute_normals(self._tris)
        self._dl=self._build_dl()

        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(55,width/height,0.01,100); glMatrixMode(GL_MODELVIEW)

    def _build_dl(self):
        dl=glGenLists(1); glNewList(dl,GL_COMPILE)
        glColor3f(0.85,0.12,0.12); glBegin(GL_TRIANGLES)
        for (a,b,c),n in zip(self._tris,self._normals):
            glNormal3f(*n); glVertex3f(*a); glVertex3f(*b); glVertex3f(*c)
        glEnd(); glEndList(); return dl

    def update(self,gesture,sd,rd):
        self.scale=max(0.3,min(3.5,self.scale+sd)); self.yaw+=rd
        if gesture=="interior_view" and not self.interior: self.interior=True
        elif gesture=="reset": self.interior=False; self.scale=1.0; self.yaw=0.0

    def render_rgba(self):
        glBindFramebuffer(GL_FRAMEBUFFER,self._fbo)
        glViewport(0,0,self.w,self.h)
        glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        if self.interior:
            gluLookAt(0.1,-0.2,0.3,0.5,0.8,0.3,0,0,1)
        else:
            d=3.5/max(self.scale,0.3); r=math.radians(self.yaw)
            gluLookAt(d*math.sin(r),-d*math.cos(r),d*0.4,0,0,0,0,0,1)
        glScalef(self.scale,self.scale,self.scale)
        glCallList(self._dl)
        glPixelStorei(GL_PACK_ALIGNMENT,1)
        raw=glReadPixels(0,0,self.w,self.h,GL_RGBA,GL_UNSIGNED_BYTE)
        arr=np.frombuffer(raw,dtype=np.uint8).reshape(self.h,self.w,4)[::-1]
        glBindFramebuffer(GL_FRAMEBUFFER,0)
        pygame.event.pump()
        return arr

    def cleanup(self):
        glDeleteLists(self._dl,1)
        glDeleteFramebuffers(1,[self._fbo])
        glDeleteRenderbuffers(1,[self._crb])
        glDeleteRenderbuffers(1,[self._drb])
        pygame.quit()


# ═══════════════════════════════════════════════════════════════════════
#  GESTURE ENGINE
# ═══════════════════════════════════════════════════════════════════════
class GestureEngine:
    def __init__(self):
        self.mp_hands=mp.solutions.hands
        self.hands=self.mp_hands.Hands(max_num_hands=2,
            min_detection_confidence=0.75,min_tracking_confidence=0.75)
        self.draw=mp.solutions.drawing_utils; self.prev_pinch=None

    @staticmethod
    def _dist(a,b): return math.hypot(a.x-b.x,a.y-b.y)

    @staticmethod
    def _fingers_up(lm):
        return sum(lm[t].y<lm[j].y for t,j in zip([8,12,16,20],[6,10,14,18]))

    def process(self,frame):
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); res=self.hands.process(rgb)
        gesture,sd,rd="none",0.0,0.0
        if not res.multi_hand_landmarks: self.prev_pinch=None; return gesture,sd,rd
        for hlm in res.multi_hand_landmarks:
            lm=hlm.landmark
            self.draw.draw_landmarks(frame,hlm,self.mp_hands.HAND_CONNECTIONS,
                self.draw.DrawingSpec(color=(0,200,150),thickness=2),
                self.draw.DrawingSpec(color=(0,255,100),thickness=1))
            pinch=self._dist(lm[4],lm[8]); fingers=self._fingers_up(lm)
            if self.prev_pinch is not None:
                d=pinch-self.prev_pinch
                if abs(d)>0.005: sd=d*1.8
            self.prev_pinch=pinch
            if fingers>=4: gesture="reset"
            elif fingers==2: gesture="interior_view"
            elif pinch<0.06: gesture="pinch_close"
            elif pinch>0.20: gesture="pinch_open"
            wx=lm[0].x
            if wx<0.30: rd=-4.0
            elif wx>0.70: rd=+4.0
        return gesture,sd,rd


# ═══════════════════════════════════════════════════════════════════════
#  ARUCO TRACKER
# ═══════════════════════════════════════════════════════════════════════
class ArucoTracker:
    def __init__(self):
        self.adict=cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.detector=cv2.aruco.ArucoDetector(self.adict,cv2.aruco.DetectorParameters())
        if not os.path.exists("aruco_marker.png"):
            img=cv2.aruco.generateImageMarker(self.adict,ARUCO_ID,300)
            cv2.imwrite("aruco_marker.png",img)
            print("[ArUco] aruco_marker.png generated — print it!")

    def detect(self,frame):
        grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        corners,ids,_=self.detector.detectMarkers(grey)
        if ids is None: return None,1.0
        for i,mid in enumerate(ids.flatten()):
            if mid!=ARUCO_ID: continue
            c=corners[i][0]; cx=int(c[:,0].mean()); cy=int(c[:,1].mean())
            side=np.linalg.norm(c[0]-c[1])
            cv2.aruco.drawDetectedMarkers(frame,corners,ids)
            return (cx,cy),max(0.3,min(3.0,side/100))
        return None,1.0


# ═══════════════════════════════════════════════════════════════════════
#  COMPOSITOR
# ═══════════════════════════════════════════════════════════════════════
def composite(bg,car_rgba,anchor=None,scale=1.0):
    hb,wb=bg.shape[:2]; hc,wc=car_rgba.shape[:2]
    nw=max(10,min(int(wc*scale),wb)); nh=max(10,min(int(hc*scale),hb))
    scaled=cv2.resize(car_rgba,(nw,nh))
    x0=(anchor[0]-nw//2 if anchor else (wb-nw)//2)
    y0=(anchor[1]-nh//2 if anchor else (hb-nh)//2)
    x0=max(0,min(x0,wb-nw)); y0=max(0,min(y0,hb-nh))
    roi=bg[y0:y0+nh,x0:x0+nw].astype(np.float32)
    fg=scaled[:,:,:3].astype(np.float32)[:,:,::-1]
    a=scaled[:,:,3:4].astype(np.float32)/255.0
    out=bg.copy(); out[y0:y0+nh,x0:x0+nw]=(fg*a+roi*(1-a)).astype(np.uint8)
    return out


# ═══════════════════════════════════════════════════════════════════════
#  CONVERSATIONAL AI
# ═══════════════════════════════════════════════════════════════════════
import threading
import queue
import ollama
import pyttsx3


class ConversationalAI:
    SYSTEM = (
        "You are an enthusiastic AR car guide with deep automotive knowledge. "
        "Keep every reply to 1-2 sentences maximum. "
        "Be vivid, specific, and friendly. No markdown, no bullet points."
    )

    GESTURE_PROMPTS = {
        "interior_view": "Describe the interior view of a modern performance car cabin.",
        "pinch_open": "What does scaling up a car's size do to its aerodynamics?",
        "reset": "Give me one surprising fact about iconic car design.",
    }

    def __init__(self, model=OLLAMA_MODEL):
        self.model = model
        self.history = []
        self.reply = "System ready."
        self.busy = False
        self.lock = threading.Lock()

        # Dedicated TTS Queue and Worker Thread
        self.tts_queue = queue.Queue()
        threading.Thread(target=self._tts_worker, daemon=True).start()

    def _tts_worker(self):
        """Persistent worker thread to handle speech sequentially."""
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 168)
            engine.setProperty("volume", 0.9)
            while True:
                text = self.tts_queue.get()
                if text:
                    engine.say(text)
                    engine.runAndWait()
                self.tts_queue.task_done()
        except Exception as e:
            print(f"[TTS Worker Error] {e}")

    def chat(self, user_text, speak_reply=True):
        if self.busy: return
        self.busy = True
        # Fire off the query in a background thread
        threading.Thread(target=self._run, args=(user_text, speak_reply), daemon=True).start()

    def _run(self, user_text, speak_reply):
        with self.lock:
            self.history.append({"role": "user", "content": user_text})

        try:
            # Query Ollama
            res = ollama.chat(
                model=self.model,
                messages=[{"role": "system", "content": self.SYSTEM}] + self.history[-10:]
            )
            reply_text = res["message"]["content"].strip()
        except Exception as e:
            reply_text = "I'm having trouble connecting to the brain."
            print(f"[Ollama Error] {e}")

        with self.lock:
            self.history.append({"role": "assistant", "content": reply_text})
            self.reply = reply_text

        # Push to queue instead of calling pyttsx3 directly
        if speak_reply and VOICE_OK:
            self.tts_queue.put(reply_text)

        self.busy = False

    def get_reply(self):
        with self.lock: return self.reply

# class ConversationalAI:
#     SYSTEM=(
#         "You are an enthusiastic AR car guide with deep automotive knowledge. "
#         "Keep every reply to 1-2 sentences maximum. "
#         "Be vivid, specific, and friendly. No markdown, no bullet points."
#     )
#     GESTURE_PROMPTS={
#         "interior_view":"Describe the interior view of a modern performance car cabin.",
#         "pinch_open":   "What does scaling up a car's size do to its aerodynamics?",
#         "reset":        "Give me one surprising fact about iconic car design.",
#     }
#
#     def __init__(self, model=OLLAMA_MODEL):
#         self.model = model
#         self.history = []
#         self.reply = "System ready."
#         self.busy = False
#         self.lock = threading.Lock()
#         self.engine = None
#         if VOICE_OK:
#             try:
#                 # Re-init in every thread if necessary,
#                 # but better to handle it globally
#                 self.engine = pyttsx3.init()
#                 self.engine.setProperty("rate", 170)
#             except Exception as e:
#                 print(f"[TTS Init Error] {e}")
#
#     def _speak(self, text):
#         """
#         Creates a fresh engine instance for every utterance to avoid
#         queue blocking issues common on Windows.
#         """
#         try:
#             # Initialize fresh for every call
#             engine = pyttsx3.init()
#             # Restore your preferred settings
#             engine.setProperty("rate", 168)
#             engine.setProperty("volume", 0.9)
#
#             engine.say(text)
#             engine.runAndWait()
#             # Explicitly stop to clear the event queue
#             engine.stop()
#         except Exception as e:
#             print(f"[TTS Error] {e}")
#     def chat(self, user_text, speak_reply=True):
#         if self.busy: return
#         self.busy = True
#         # Fire off the query and speech as a single background task
#         threading.Thread(target=self._run, args=(user_text, speak_reply), daemon=True).start()
#
#     def _run(self, user_text, speak_reply):
#         with self.lock:
#             self.history.append({"role": "user", "content": user_text})
#
#         try:
#             # Ensure ollama is actually running before calling
#             res = ollama.chat(model=self.model, messages=self.history[-10:])
#             reply_text = res["message"]["content"].strip()
#         except Exception as e:
#             reply_text = "I'm having trouble connecting to the brain."
#             print(f"[Ollama Error] {e}")
#
#         with self.lock:
#             self.history.append({"role": "assistant", "content": reply_text})
#             self.reply = reply_text
#
#         if speak_reply and VOICE_OK:
#             self._speak(reply_text)
#
#         self.busy = False
#
#     def get_reply(self):
#         with self.lock: return self.reply


# ═══════════════════════════════════════════════════════════════════════
#  CONTINUOUS VOICE LISTENER
# ═══════════════════════════════════════════════════════════════════════
class ContinuousListener:
    def __init__(self):
        self.q=queue.Queue(); self.stop=threading.Event()

    def start(self):
        if not VOICE_OK: return
        threading.Thread(target=self._loop,daemon=True).start()

    def _loop(self):
        rec=sr.Recognizer(); mic=sr.Microphone()
        rec.dynamic_energy_threshold=True; rec.energy_threshold=3000
        with mic as src: rec.adjust_for_ambient_noise(src,duration=1.0)
        print("[Voice] Listening continuously — just speak!")
        while not self.stop.is_set():
            try:
                with mic as src:
                    audio=rec.listen(src,timeout=2,phrase_time_limit=10)
                text=rec.recognize_google(audio)
                if text.strip(): self.q.put(text); print(f"[Voice] '{text}'")
            except sr.WaitTimeoutError: pass
            except sr.UnknownValueError: pass
            except Exception as e: print(f"[Voice] {e}"); time.sleep(0.5)

    def get_text(self):
        try: return self.q.get_nowait()
        except queue.Empty: return None

    def close(self): self.stop.set()


# ═══════════════════════════════════════════════════════════════════════
#  HUD
# ═══════════════════════════════════════════════════════════════════════
def draw_hud_v(frame,gesture,scale,ai_text,aruco_found,fps,listening):
    h,w=frame.shape[:2]
    ov=frame.copy(); cv2.rectangle(ov,(0,0),(350,115),(0,0,0),-1)
    cv2.addWeighted(ov,0.45,frame,0.55,0,frame)
    def put(t,y,col=(210,255,170),s=0.62,th=1):
        cv2.putText(frame,t,(10,y),cv2.FONT_HERSHEY_SIMPLEX,s,col,th,cv2.LINE_AA)
    labels={"none":"","pinch_close":"Shrinking","pinch_open":"Scaling up",
            "interior_view":"Interior","reset":"Reset"}
    put(f"Gesture : {labels.get(gesture,gesture)}",26)
    put(f"Scale   : {scale:.2f}x",52)
    put(f"ArUco   : {'ANCHORED' if aruco_found else 'free'}",78,
        (100,255,100) if aruco_found else (80,80,220))
    put(f"FPS {fps:.0f}   Q=quit   speak freely",104,(160,160,160))
    if listening:
        r=int(8+4*math.sin(time.time()*6))
        cv2.circle(frame,(w-20,20),r,(0,60,255),-1)
    if ai_text:
        words=ai_text.split(); lines,cur=[],""
        for word in words:
            test=(cur+" "+word).strip()
            if len(test)*9<w-24: cur=test
            else: lines.append(cur); cur=word
        if cur: lines.append(cur)
        lines=lines[:3]; bh=len(lines)*28+16
        bar=frame.copy(); cv2.rectangle(bar,(0,h-bh),(w,h),(10,10,10),-1)
        cv2.addWeighted(bar,0.62,frame,0.38,0,frame)
        for i,line in enumerate(lines):
            cv2.putText(frame,line,(12,h-bh+24+i*28),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,240,140),1,cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("="*58)
    print("  AR Car — Voice + Gesture  (Windows Compatible)")
    print("="*58)
    print("  Just SPEAK — no key press needed.")
    print("  Gestures work simultaneously.")
    print("  Say 'exit' or press Q to quit.")
    print("="*58)

    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAM_H)
    cap.set(cv2.CAP_PROP_FPS,30)

    print("[INIT] Loading renderer…")
    car=CarRenderer()
    print("[INIT] Renderer ready")

    ge=GestureEngine(); ai=ConversationalAI()
    aruco=ArucoTracker(); voice=ContinuousListener()
    voice.start()
    ai.chat("Introduce yourself as the AR car guide in one sentence.",speak_reply=False)

    scale=1.0; last_gesture="none"
    fps_timer=time.time(); fps=0.0; fc=0

    print("\n[READY] AR window open. Speak or gesture.\n")

    while True:
        ret,frame=cap.read()
        if not ret: break
        frame=cv2.flip(frame,1)

        gesture,sd,rd=ge.process(frame)
        scale=max(0.3,min(3.5,scale+sd))

        anchor,aruco_scale=aruco.detect(frame)
        eff=scale*aruco_scale if anchor else scale

        car.update(gesture,sd,rd)
        frame=composite(frame,car.render_rgba(),anchor=anchor,scale=eff*0.7)

        if gesture!=last_gesture:
            p=ConversationalAI.GESTURE_PROMPTS.get(gesture)
            if p: ai.chat(p,speak_reply=True)
        last_gesture=gesture

        spoken=voice.get_text()
        if spoken:
            if spoken.lower().strip() in ("exit","quit","stop","goodbye"): break
            ai.chat(spoken,speak_reply=True)

        fc+=1; now=time.time()
        if now-fps_timer>=1.0: fps=fc/(now-fps_timer); fc=0; fps_timer=now

        draw_hud_v(frame,gesture,scale,ai.get_reply(),
                   anchor is not None,fps,VOICE_OK and not voice.stop.is_set())

        cv2.imshow("AR Car — Voice + Gesture  (Q quit)",frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    voice.close(); car.cleanup(); cap.release(); cv2.destroyAllWindows()
    print("[EXIT] Done.")

if __name__=="__main__":
    main()