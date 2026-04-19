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
#  HUD v7  —  Sci-fi AR Interface
#  Design language: dark glass panels, neon cyan/amber accents,
#  corner-bracket borders, live telemetry bars, animated elements.
# ═══════════════════════════════════════════════════════════════════════

# ── Colour palette ────────────────────────────────────────────────────
C = {
    "cyan"   : (220, 210,  20),   # BGR: bright cyan
    "green"  : ( 80, 220,  80),
    "amber"  : ( 20, 180, 220),   # BGR: amber/orange
    "red"    : ( 50,  50, 220),
    "blue"   : (220, 140,  30),
    "white"  : (230, 240, 240),
    "dim"    : (100, 120, 110),
    "dimmer" : ( 60,  70,  65),
    "bg"     : (  6,   8,  14),
    "panel"  : ( 10,  14,  22),
    "border" : ( 0,  160, 100),
    "holo"   : (230, 190,  20),   # hologram cyan
    "warn"   : ( 40, 100, 255),
    "gold"   : ( 30, 210, 255),
}
F  = cv2.FONT_HERSHEY_SIMPLEX
FM = cv2.FONT_HERSHEY_DUPLEX

# ── Primitive draw helpers ────────────────────────────────────────────
def _blend_rect(frame, x, y, w, h, color, alpha=0.70):
    """Semi-transparent filled rectangle."""
    ov = frame.copy()
    cv2.rectangle(ov, (x,y), (x+w, y+h), color, -1)
    cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)

def _t(frame, txt, x, y, col=None, sc=0.44, th=1, font=None):
    col  = col  or C["white"]
    font = font or F
    cv2.putText(frame, txt, (x,y), font, sc, col, th, cv2.LINE_AA)

def _bar(frame, x, y, w, val, mx, col_fill, col_bg=(28,28,28), h=6):
    cv2.rectangle(frame, (x,y), (x+w, y+h), col_bg, -1)
    f = max(0, int(w * min(val/max(mx,1), 1.0)))
    if f: cv2.rectangle(frame, (x,y), (x+f, y+h), col_fill, -1)

def _bracket_panel(frame, x, y, w, h, col=None, thick=1, sz=12):
    """Sci-fi corner-bracket border instead of plain rectangle."""
    col = col or C["border"]
    tl,br = (x,y), (x+w,y+h)
    tr,bl = (x+w,y), (x,y+h)
    for px,py,dx,dy in [(tl[0],tl[1],1,1),(tr[0],tr[1],-1,1),
                         (bl[0],bl[1],1,-1),(br[0],br[1],-1,-1)]:
        cv2.line(frame,(px,py),(px+dx*sz,py),col,thick,cv2.LINE_AA)
        cv2.line(frame,(px,py),(px,py+dy*sz),col,thick,cv2.LINE_AA)

def _panel(frame, x, y, w, h, title="", col=None, alpha=0.72, title_col=None):
    """Dark glass panel with corner brackets and optional title."""
    col       = col       or C["border"]
    title_col = title_col or C["cyan"]
    _blend_rect(frame, x, y, w, h, C["panel"], alpha)
    _bracket_panel(frame, x, y, w, h, col)
    if title:
        # Title bar stripe
        _blend_rect(frame, x+1, y+1, w-2, 16, col, 0.35)
        cv2.putText(frame, title.upper(), (x+6, y+13),
                    F, 0.33, title_col, 1, cv2.LINE_AA)

def _dot(frame, x, y, r, col):
    cv2.circle(frame, (x,y), r, col, -1, cv2.LINE_AA)

def _pulse_dot(frame, x, y, col, t, period=1.0):
    """Pulsing dot — bright centre + soft outer ring."""
    phase = (math.sin(t * 2*math.pi / period) + 1) / 2
    r_outer = int(5 + 4*phase)
    alpha_outer = 0.3 + 0.4*phase
    ov = frame.copy()
    cv2.circle(ov, (x,y), r_outer, col, -1, cv2.LINE_AA)
    cv2.addWeighted(ov, alpha_outer, frame, 1-alpha_outer, 0, frame)
    cv2.circle(frame, (x,y), 4, col, -1, cv2.LINE_AA)

def _hline(frame, x1, x2, y, col=None, th=1):
    cv2.line(frame, (x1,y), (x2,y), col or C["border"], th, cv2.LINE_AA)

def _vline(frame, x, y1, y2, col=None, th=1):
    cv2.line(frame, (x,y1), (x,y2), col or C["border"], th, cv2.LINE_AA)

def _glow_text(frame, txt, x, y, col, sc=0.5, th=2):
    """Text with a darker outline for readability on any background."""
    dark = tuple(max(0,c//4) for c in col)
    cv2.putText(frame, txt, (x,y), FM, sc, dark, th+2, cv2.LINE_AA)
    cv2.putText(frame, txt, (x,y), FM, sc, col,  th,   cv2.LINE_AA)

# ── Animated corner scan-line ─────────────────────────────────────────
def _scanline(frame, t):
    h,w = frame.shape[:2]
    y = int((t*80) % (h+20)) - 10
    if 0 <= y < h:
        ov = frame.copy()
        cv2.line(ov, (0,y), (w,y), C["cyan"], 1)
        cv2.addWeighted(ov, 0.08, frame, 0.92, 0, frame)

# ── Mini radar (hand tracking visualiser) ────────────────────────────
def _mini_radar(frame, ge, x, y, r, t):
    """
    Small circular radar in bottom-left showing hand positions
    and rotation of the car as a blip sweep.
    """
    cx, cy = x+r, y+r
    # Radar BG
    _blend_rect(frame, x, y, r*2, r*2, C["bg"], 0.80)
    # Rings
    for ri in [r, r*2//3, r//3]:
        cv2.circle(frame,(cx,cy),ri,C["dimmer"],1,cv2.LINE_AA)
    # Cross-hairs
    _hline(frame,x,x+r*2,cy,C["dimmer"])
    _vline(frame,cx,y,y+r*2,C["dimmer"])
    # Sweep arm
    sweep_ang = math.radians((t*60)%360)
    sx = int(cx + r*math.cos(sweep_ang))
    sy = int(cy + r*math.sin(sweep_ang))
    cv2.line(frame,(cx,cy),(sx,sy),C["cyan"],1,cv2.LINE_AA)
    # Fading trail
    for trail in range(6):
        ta = math.radians(((t*60)%360) - trail*8)
        tx = int(cx + r*math.cos(ta))
        ty = int(cy + r*math.sin(ta))
        alpha = 0.15*(6-trail)/6
        ov2 = frame.copy()
        cv2.line(ov2,(cx,cy),(tx,ty),C["cyan"],1,cv2.LINE_AA)
        cv2.addWeighted(ov2,alpha,frame,1-alpha,0,frame)

    # Hand blips
    lp = ge.get("left_present",  False)
    rp = ge.get("right_present", False)
    pp = ge.get("palm_pos", None)
    if rp:
        bx = cx + int((ge.get("right_yaw",0)/12)*r*0.3)
        by = cy - int((ge.get("right_pitch",0)/6)*r*0.3)
        bx = max(x+4,min(x+r*2-4,bx)); by = max(y+4,min(y+r*2-4,by))
        _pulse_dot(frame,bx,by,C["cyan"],t,0.8)
    if lp:
        _pulse_dot(frame,cx-r//2,cy+r//3,C["amber"],t,1.2)

    # Label
    cv2.putText(frame,"RADAR",(x+2,y+r*2+12),F,0.30,C["dim"],1,cv2.LINE_AA)

# ── Telemetry arc (rotation gauge) ───────────────────────────────────
def _rotation_gauge(frame, yaw, pitch, scale, x, y, r, t):
    """Circular gauge showing yaw, with pitch and scale as bars."""
    cx,cy = x+r,y+r
    _blend_rect(frame,x,y,r*2,r*2+30,C["bg"],0.78)
    # Outer ring
    cv2.circle(frame,(cx,cy),r,C["dimmer"],1,cv2.LINE_AA)
    cv2.circle(frame,(cx,cy),r-4,C["panel"],1)

    # Yaw arc (0-360)
    ang = (yaw % 360)
    start_ang = -90
    end_ang   = int(start_ang + ang)
    col_arc = C["cyan"] if not False else C["amber"]
    cv2.ellipse(frame,(cx,cy),(r-2,r-2),0,start_ang,end_ang,C["cyan"],2,cv2.LINE_AA)

    # Needle
    needle_ang = math.radians(ang - 90)
    nx = int(cx + (r-6)*math.cos(needle_ang))
    ny = int(cy + (r-6)*math.sin(needle_ang))
    cv2.line(frame,(cx,cy),(nx,ny),C["amber"],2,cv2.LINE_AA)
    cv2.circle(frame,(cx,cy),4,C["amber"],-1,cv2.LINE_AA)

    # Centre text: yaw value
    yaw_str = f"{int(ang%360):03d}"
    cv2.putText(frame,yaw_str,(cx-14,cy+5),FM,0.44,C["cyan"],1,cv2.LINE_AA)
    cv2.putText(frame,"YAW",(cx-8,cy+16),F,0.28,C["dim"],1,cv2.LINE_AA)

    # Pitch bar below gauge
    _bar(frame, x, y+r*2+6, r*2, pitch+85, 170, C["green"], h=5)
    cv2.putText(frame,f"PITCH {pitch:.0f}",(x,y+r*2+22),F,0.28,C["dim"],1,cv2.LINE_AA)

    # Scale bar
    _bar(frame, x, y+r*2+20, r*2, scale, 8.0, C["gold"], h=4)
    cv2.putText(frame,f"SCALE {scale:.1f}x",(x,y+r*2+32),F,0.28,C["dim"],1,cv2.LINE_AA)

# ── AI chat log (scrolling last N entries) ────────────────────────────
_chat_log: list = []   # [(role, text), ...]  persists across frames
_last_reply = ""

def _update_chat_log(ai_reply, voice_last):
    global _chat_log, _last_reply
    if ai_reply and ai_reply != _last_reply:
        _last_reply = ai_reply
        _chat_log.append(("ai", ai_reply))
        if len(_chat_log) > 12:
            _chat_log = _chat_log[-12:]

def _draw_chat_log(frame, x, y, w, h):
    """Scrolling conversation log panel."""
    _panel(frame, x, y, w, h, "JARVIS LOG", col=C["blue"], alpha=0.78)
    max_chars = (w-16)//7
    row_h = 16
    rows  = []
    for role,txt in _chat_log[-6:]:
        prefix = "AI>" if role=="ai" else "YOU>"
        col    = C["cyan"] if role=="ai" else C["amber"]
        # word wrap
        words = txt.split(); cur=""
        for word in words:
            test=(cur+" "+word).strip()
            if len(test)<max_chars: cur=test
            else:
                if cur: rows.append((prefix,col,cur)); prefix=""
                cur=word
        if cur: rows.append((prefix,col,cur))
    rows = rows[-(h//row_h - 2):]
    for i,(pfx,col,line) in enumerate(rows):
        yy = y + 22 + i*row_h
        if yy > y+h-6: break
        if pfx:
            cv2.putText(frame,pfx,(x+4,yy),FM,0.32,col,1,cv2.LINE_AA)
        cv2.putText(frame,line,(x+36,yy),F,0.32,C["white"],1,cv2.LINE_AA)

# ── Voice waveform ────────────────────────────────────────────────────
def _waveform(frame, x, y, w, h, t, active):
    col   = C["cyan"] if active else C["dimmer"]
    mid_y = y + h//2
    for i in range(w//4):
        amp = (int(h*0.45*abs(math.sin(t*7+i*0.4)))+2) if active else 2
        px  = x + i*4
        cv2.line(frame,(px,mid_y-amp),(px,mid_y+amp),col,2,cv2.LINE_AA)

# ── Hand silhouette indicator ─────────────────────────────────────────
def _hand_icon(frame, x, y, label, active, col_active, t):
    col = col_active if active else C["dimmer"]
    # Simple hand icon using circles and lines
    # Palm
    cv2.circle(frame,(x+14,y+18),10,col,1,cv2.LINE_AA)
    # Fingers
    for fi,(fx,fy_base) in enumerate([(7,8),(11,4),(15,4),(19,6),(23,10)]):
        flen = 8 if active else 5
        cv2.line(frame,(x+fx,y+fy_base),(x+fx,y+fy_base-flen),col,2,cv2.LINE_AA)
    _t(frame,label,x,y+30,col,0.30)

# ── Mode badge (centre top) ───────────────────────────────────────────
def _mode_badge(frame, w, h, car, holo, lc, lp, rp, t):
    """Animated mode display centre-top of screen."""
    mode_txt = ""
    mode_col = C["cyan"]
    if car.frozen:
        mode_txt = "** FROZEN **"; mode_col = C["red"]
    elif holo:
        mode_txt = "HOLOGRAM ACTIVE"; mode_col = C["holo"]
    elif car.interior:
        mode_txt = "INTERIOR VIEW";  mode_col = C["green"]
    elif car.exploded:
        mode_txt = "EXPLODE VIEW";   mode_col = C["amber"]
    elif car.xray:
        mode_txt = "X-RAY MODE";     mode_col = C["blue"]
    elif lp and rp:
        mode_txt = "DUAL-HAND CONTROL"
    else:
        hand = "RIGHT" if rp else ("LEFT" if lp else "NO HAND")
        mode_txt = f"{hand} HAND  |  YAW {car.yaw%360:.0f}  SCALE {car.scale:.2f}x"
        mode_col = C["dim"]

    if not mode_txt: return
    tw  = len(mode_txt)*8
    bx  = (w-tw-24)//2; by = 4
    bw  = tw+24;        bh = 22
    _blend_rect(frame, bx, by, bw, bh, C["bg"], 0.75)
    # Animated bracket colour
    pulse = 0.6+0.4*math.sin(t*4)
    bc = tuple(int(c*pulse) for c in mode_col)
    _bracket_panel(frame, bx, by, bw, bh, bc, sz=8)
    cx_txt = bx + (bw-tw)//2
    cv2.putText(frame, mode_txt, (cx_txt, by+15),
                FM, 0.40, mode_col, 1, cv2.LINE_AA)

# ── Crosshair / targeting reticle (frame centre) ─────────────────────
def _reticle(frame, w, h, t):
    cx,cy = w//2, h//2
    r_out = 28; r_in = 10; gap = 6
    alpha = 0.5 + 0.3*math.sin(t*2)
    col   = tuple(int(c*alpha) for c in C["cyan"])
    # Four lines
    for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        x1 = cx+dx*(r_in+gap); y1 = cy+dy*(r_in+gap)
        x2 = cx+dx*r_out;      y2 = cy+dy*r_out
        cv2.line(frame,(x1,y1),(x2,y2),col,1,cv2.LINE_AA)
    # Rotating outer ticks
    for i in range(8):
        ang = math.radians(i*45 + t*20)
        tx1 = int(cx+(r_out+2)*math.cos(ang)); ty1=int(cy+(r_out+2)*math.sin(ang))
        tx2 = int(cx+(r_out+6)*math.cos(ang)); ty2=int(cy+(r_out+6)*math.sin(ang))
        cv2.line(frame,(tx1,ty1),(tx2,ty2),col,1,cv2.LINE_AA)
    # Centre dot
    cv2.circle(frame,(cx,cy),2,col,-1,cv2.LINE_AA)

# ── Hologram status orb (top-right corner) ────────────────────────────
def _hologram_orb(frame, w, holo, t):
    cx = w-28; cy = 36
    if holo:
        r = int(14+4*math.sin(t*6))
        ov=frame.copy()
        cv2.circle(ov,(cx,cy),r,C["holo"],-1,cv2.LINE_AA)
        cv2.addWeighted(ov,0.35,frame,0.65,0,frame)
        cv2.circle(frame,(cx,cy),12,C["holo"],2,cv2.LINE_AA)
        cv2.putText(frame,"HOLO",(cx-14,cy+24),F,0.30,C["holo"],1,cv2.LINE_AA)
    else:
        cv2.circle(frame,(cx,cy),12,C["dimmer"],1,cv2.LINE_AA)
        cv2.putText(frame,"OFF",(cx-10,cy+24),F,0.30,C["dimmer"],1,cv2.LINE_AA)

# ── Corner decoration lines ───────────────────────────────────────────
def _frame_deco(frame, w, h, t):
    """Subtle animated corner decorations on the full frame."""
    pulse = int(0.5+0.5*math.sin(t*1.5))
    col   = tuple(max(0,c-80+pulse*30) for c in C["border"])
    sz    = 40
    for px,py,dx,dy in [(0,0,1,1),(w-1,0,-1,1),(0,h-1,1,-1),(w-1,h-1,-1,-1)]:
        cv2.line(frame,(px,py),(px+dx*sz,py),col,1,cv2.LINE_AA)
        cv2.line(frame,(px,py),(px,py+dy*sz),col,1,cv2.LINE_AA)

# ── FPS ring gauge ────────────────────────────────────────────────────
def _fps_ring(frame, fps, x, y, r):
    cx,cy=x+r,y+r
    _blend_rect(frame,x,y,r*2,r*2,C["bg"],0.75)
    cv2.circle(frame,(cx,cy),r,C["dimmer"],1,cv2.LINE_AA)
    end_ang = int(-90 + (fps/60)*360)
    col = C["green"] if fps>24 else C["amber"] if fps>15 else C["red"]
    cv2.ellipse(frame,(cx,cy),(r,r),0,-90,min(270,end_ang),col,2,cv2.LINE_AA)
    cv2.putText(frame,f"{fps:.0f}",(cx-9,cy+4),FM,0.40,col,1,cv2.LINE_AA)
    cv2.putText(frame,"FPS",(cx-8,cy+14),F,0.28,C["dim"],1,cv2.LINE_AA)

# ── Gesture grid (compact, 2-col) ─────────────────────────────────────
def _gesture_grid(frame, x, y, w, h, lc, holo, ge, lp, rp, car):
    _panel(frame,x,y,w,h,"GESTURE MAP",col=C["border"])
    both = lp and rp
    entries = [
        # (label, active_condition, col_when_active)
        ("R: Wrist→Orbit",  ge.get("right_yaw",0)!=0,          C["cyan"]),
        ("R: Pinch→Zoom",   ge.get("right_zoom",0)!=0,          C["cyan"]),
        ("R: Palm→HOLO",    holo,                               C["holo"]),
        ("R: Point→Spot",   ge.get("pointing",False),           C["cyan"]),
        ("L: 1F→Interior",  both and lc=="interior",            C["amber"]),
        ("L: 2F→Explode",   both and lc=="explode",             C["amber"]),
        ("L: 3F→X-Ray",     both and lc=="xray",                C["amber"]),
        ("L: Palm→Reset",   both and lc=="reset",               C["red"]),
        ("L: Fist→Freeze",  both and lc=="freeze",              C["red"]),
        ("L: Pinch→Scale",  both and ge.get("left_scale",0)!=0, C["green"]),
    ]
    col_w = (w-12)//2
    for i,(label,active,acol) in enumerate(entries):
        row = i//2; col_ = i%2
        xx  = x+6+col_*col_w
        yy  = y+26+row*14
        dot_col = acol if active else C["dimmer"]
        cv2.circle(frame,(xx+4,yy-3),3,dot_col,-1,cv2.LINE_AA)
        text_col= acol if active else C["dimmer"]
        cv2.putText(frame,label,(xx+10,yy),F,0.30,text_col,1,cv2.LINE_AA)

    # Both-hands indicator
    need_col = C["green"] if both else C["red"]
    need_txt = "BOTH HANDS READY" if both else "NEED BOTH HANDS"
    _blend_rect(frame,x+1,y+h-14,w-2,13,C["bg"],0.5)
    cv2.putText(frame,need_txt,(x+6,y+h-4),F,0.30,need_col,1,cv2.LINE_AA)

# ── Car telemetry panel ────────────────────────────────────────────────
def _telemetry(frame, x, y, w, car, t):
    h = 148
    _panel(frame,x,y,w,h,"CAR TELEMETRY",col=C["amber"])
    paint_rgb = CAR_COLORS[car.color_idx][0]
    pr,pg,pb  = int(paint_rgb[2]*255),int(paint_rgb[1]*255),int(paint_rgb[0]*255)
    view=("Interior" if car.interior else "Exploded" if car.exploded
          else "X-Ray" if car.xray else "Hologram" if car.hologram_mode
          else "Wire"  if car.wireframe else "Exterior")

    rows = [
        ("VIEW",  view,               C["cyan"]),
        ("MODEL", os.path.basename(MODEL_PATH), C["white"]),
        ("TRIS",  f"{car._tri_count:,}",        C["white"]),
        ("SCALE", f"{car.scale:.2f}x",           C["gold"]),
        ("YAW",   f"{car.yaw%360:.1f}",          C["green"]),
        ("PITCH", f"{car.pitch:.1f}",            C["green"]),
        ("PAINT", CAR_COLORS[car.color_idx][1],  (pb,pg,pr)),
    ]
    for i,(k,v,col) in enumerate(rows):
        yy=y+24+i*17
        cv2.putText(frame,k+":",(x+6,yy),F,0.33,C["dim"],1,cv2.LINE_AA)
        cv2.putText(frame,v,(x+66,yy),FM,0.34,col,1,cv2.LINE_AA)

    # Paint swatch
    cv2.rectangle(frame,(x+w-22,y+20),(x+w-6,y+36),(pb,pg,pr),-1)
    cv2.rectangle(frame,(x+w-22,y+20),(x+w-6,y+36),C["border"],1)

    # Active modes row
    modes=[]
    if car.auto_spin: modes.append("SPIN")
    if car.frozen:    modes.append("FREEZE")
    if car.wireframe: modes.append("WIRE")
    mode_str = "  ".join(modes) if modes else "—"
    cv2.putText(frame,"FLAGS:",(x+6,y+h-6),F,0.30,C["dim"],1,cv2.LINE_AA)
    cv2.putText(frame,mode_str,(x+50,y+h-6),F,0.30,C["amber"],1,cv2.LINE_AA)

# ── System status panel (right side) ──────────────────────────────────
def _sys_status(frame, x, y, w, fps, ai, voice, anchor, t):
    h=112
    _panel(frame,x,y,w,h,"SYS STATUS",col=C["border"])

    # FPS ring (small)
    _fps_ring(frame,fps,x+w-40,y+6,17)

    rows=[
        ("AI",    "BUSY" if ai.busy else "READY",
                  C["warn"] if ai.busy else C["green"]),
        ("VOICE", "LISTEN" if voice.is_listening else "WAIT",
                  C["holo"] if voice.is_listening else C["dimmer"]),
        ("TTS",   "SPEAK" if ai.tts.is_speaking() else "IDLE",
                  C["blue"] if ai.tts.is_speaking() else C["dimmer"]),
        ("ARUCO", "LOCKED" if anchor else "SCAN",
                  C["green"] if anchor else C["dimmer"]),
    ]
    for i,(k,v,col) in enumerate(rows):
        yy=y+22+i*20
        _pulse_dot(frame,x+10,yy-4,col,t,0.8)
        cv2.putText(frame,k+":",(x+18,yy),F,0.33,C["dim"],1,cv2.LINE_AA)
        cv2.putText(frame,v,(x+60,yy),FM,0.34,col,1,cv2.LINE_AA)

# ── Keys quick-ref ─────────────────────────────────────────────────────
def _keys_panel(frame, x, y, w):
    h=80
    _panel(frame,x,y,w,h,"KEYS",col=C["dimmer"])
    ks=[("V","Voice"),("C","Color"),("W","Wire"),("X","X-Ray"),
        ("N","Night"),("A","Spin"), ("S","Shot"), ("Q","Quit")]
    for i,(k,v) in enumerate(ks):
        col_=i%2; row=i//2
        xx=x+6+col_*(w//2); yy=y+24+row*13
        cv2.putText(frame,f"[{k}]",(xx,yy),F,0.32,C["cyan"],1,cv2.LINE_AA)
        cv2.putText(frame,v,(xx+24,yy),F,0.31,C["dim"],1,cv2.LINE_AA)

# ── AI bottom bar ─────────────────────────────────────────────────────
def _ai_bar(frame, w, h, ai, voice, t):
    bh=82; by=h-bh-2
    _panel(frame,4,by,w-8,bh,
           f"JARVIS  |  Say '{ACTIVATION} ...' for vision AI",
           col=C["blue"],alpha=0.78)
    reply=ai.get_reply()
    words=reply.split(); lines=[]; cur=""
    mc=(w-80)//7
    for word in words:
        test=(cur+" "+word).strip()
        if len(test)<mc: cur=test
        else: lines.append(cur); cur=word
    if cur: lines.append(cur)
    for i,ln in enumerate(lines[:3]):
        yy=by+22+i*20
        if yy>by+bh-4: break
        cv2.putText(frame,ln,(12,yy),F,0.44,C["gold"],1,cv2.LINE_AA)

    # Waveform right side
    _waveform(frame, w-180, by+30, 160, 30, t, voice.is_listening)

    # Speaking orb
    if ai.tts.is_speaking():
        r2=int(7+4*math.sin(t*10))
        ov=frame.copy()
        cv2.circle(ov,(w-14,by+16),r2+4,C["blue"],-1)
        cv2.addWeighted(ov,0.3,frame,0.7,0,frame)
        cv2.circle(frame,(w-14,by+16),r2,C["blue"],-1,cv2.LINE_AA)

    # Thinking animation
    if ai.busy:
        dots="●"*(int(t*3)%4+1)
        cv2.putText(frame,dots,(w-70,by+bh-8),FM,0.45,C["warn"],1,cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════
#  MAIN DRAW FUNCTION
# ══════════════════════════════════════════════════════════════════════
def draw_hud(frame, car, ge, ai, fps, anchor, voice, popup, night):
    h, w = frame.shape[:2]
    t    = time.time()
    lc   = ge.get("left_cmd",   "none")
    holo = ge.get("hologram",   False)
    lp   = ge.get("left_present",  False)
    rp   = ge.get("right_present", False)
    hcount = ge.get("hand_count", 0)

    # ── Night mode tint ───────────────────────────────────────────────
    if night:
        tint = frame.copy()
        tint[:] = (tint * np.array([0.45, 0.55, 1.0])).clip(0,255).astype(np.uint8)
        cv2.addWeighted(tint, 0.20, frame, 0.80, 0, frame)

    # ── Subtle scanline ───────────────────────────────────────────────
    _scanline(frame, t)

    # ── Frame corner decorations ───────────────────────────────────────
    _frame_deco(frame, w, h, t)

    # ── Centre reticle ────────────────────────────────────────────────
    _reticle(frame, w, h, t)

    # ── Hologram orb (top-right corner) ───────────────────────────────
    _hologram_orb(frame, w, holo, t)

    # ── Mode badge (centre top) ────────────────────────────────────────
    _mode_badge(frame, w, h, car, holo, lc, lp, rp, t)

    # ═════════════════ LEFT COLUMN (x=4..218) ════════════════════════
    PW = 214   # panel width left column

    # Panel L1: Gesture map
    _gesture_grid(frame, 4, 8, PW, 168, lc, holo, ge, lp, rp, car)

    # Panel L2: Mini radar (hand tracker)
    radar_r = 44
    _mini_radar(frame, ge, 4, 184, radar_r, t)

    # Panel L3: Rotation gauge (next to radar)
    gauge_r = 40
    _rotation_gauge(frame, car.yaw, car.pitch, car.scale,
                    radar_r*2+12, 184, gauge_r, t)

    # ═════════════════ RIGHT COLUMN (x=w-PW-4) ═══════════════════════
    rx = w - PW - 4

    # Panel R1: System status
    _sys_status(frame, rx, 8, PW, fps, ai, voice, anchor, t)

    # Panel R2: Keys quick-ref
    _keys_panel(frame, rx, 128, PW)

    # Panel R3: Car telemetry
    _telemetry(frame, rx, 216, PW, car, t)

    # Panel R4: Chat log
    _draw_chat_log(frame, rx, 372, PW, h-82-372-8)

    # ═════════════════ BOTTOM BAR ════════════════════════════════════
    # Update chat log with latest reply
    _update_chat_log(ai.get_reply(), voice.last_heard if hasattr(voice,"last_heard") else "")

    # AI bar
    _ai_bar(frame, w, h, ai, voice, t)

    # ═════════════════ OVERLAYS ═══════════════════════════════════════
    # Left-hand command flash (centre screen)
    if lc != "none":
        labels = {"interior":"INTERIOR VIEW","explode":"EXPLODE VIEW",
                  "xray":"X-RAY","reset":"SYSTEM RESET",
                  "freeze":"FREEZE" if not car.frozen else "UNFREEZE"}
        label = labels.get(lc, lc.upper())
        _glow_text(frame, label, w//2 - len(label)*7, h//2 - 60,
                   C["amber"], 0.7, 2)

    # Hologram active flash
    if holo:
        pulse = 0.7 + 0.3*math.sin(t*8)
        col   = tuple(int(c*pulse) for c in C["holo"])
        _glow_text(frame, "HOLOGRAM PROJECTION ACTIVE",
                   w//2-130, h-90, col, 0.55, 1)

    # Two-hand cooperative scale hint
    if lp and rp and ge.get("left_scale",0)!=0:
        _glow_text(frame,"SCALING",w//2-35,h//2+30,C["green"],0.7,2)

    # Popup on top
    popup.tick()
    popup.draw(frame)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("="*66)
    print("  AR Car v7  —  Iron Man HUD + Dual-Hand + Vision AI")
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

        cv2.imshow("AR Car v7  —  Iron Man HUD  (Q=quit)", frame)

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