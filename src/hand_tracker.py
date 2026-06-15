import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.7,
            max_num_hands=2
        )

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb_frame)

    def get_hand_info(self, results):
        hand_data = []
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Landmark 8 = Ujung Telunjuk
                index_tip = np.array([landmarks.landmark[8].x, landmarks.landmark[8].y])
                # Landmark 9 = Tengah Telapak
                center = np.array([landmarks.landmark[9].x, landmarks.landmark[9].y])

                # Logika: Jari telunjuk naik (y8 < y6), jari tengah turun (y12 > y10)
                index_up = landmarks.landmark[8].y < landmarks.landmark[6].y
                middle_down = landmarks.landmark[12].y > landmarks.landmark[10].y
                
                # Mode menggambar aktif jika hanya telunjuk yang naik
                is_drawing = index_up and middle_down
                
                hand_data.append({
                    'center': center,
                    'index': index_tip,
                    'is_drawing': is_drawing
                })
        return hand_data
