import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

class GestureEngine:
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )

    def _dist(self, a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def detect(self, bgr_frame):
        rgb   = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        gesture = "none"
        scale_delta = 0.0
        rotate_delta = 0.0

        if result.multi_hand_landmarks:
            for hand_lm in result.multi_hand_landmarks:
                lm = hand_lm.landmark
                mp_draw.draw_landmarks(bgr_frame, hand_lm,
                                       mp_hands.HAND_CONNECTIONS)

                # Pinch (thumb tip ↔ index tip) → scale
                pinch = self._dist(lm[4], lm[8])
                if pinch < 0.06:
                    gesture = "pinch_close"
                    scale_delta = -0.05
                elif pinch > 0.18:
                    gesture = "pinch_open"
                    scale_delta = +0.05

                # Two fingers up → look inside car
                fingers_up = sum([
                    lm[i].y < lm[i - 2].y
                    for i in [8, 12, 16, 20]
                ])
                if fingers_up == 2:
                    gesture = "interior_view"

                # Open palm → reset
                if fingers_up >= 4:
                    gesture = "reset"

                # Wrist x-movement → orbit camera
                wrist_x = lm[0].x
                if wrist_x < 0.35:
                    rotate_delta = -3.0
                elif wrist_x > 0.65:
                    rotate_delta = +3.0

        return gesture, scale_delta, rotate_delta