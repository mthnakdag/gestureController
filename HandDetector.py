import cv2
import mediapipe as mp

class HandDetector():
    """
	Hand Detector for tracking hand using mediapipe backend
	"""
    def __init__(self,
                 mode=False,
                 maxHands=2,
                 detectionConf=0.5,
                 trackConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True, positions=False):
        h, w = img.shape[:2]
        landMarkList= []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                for id,landMark in enumerate(handLandmarks.landmark):
                    cx, cy = int(landMark.x * w), int(landMark.y *h)
                    if positions:
                        landMarkList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), -1)
                        self.mpDraw.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS)
        if positions:
            return img,landMarkList
        else:
            return img