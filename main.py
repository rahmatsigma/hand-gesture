import cv2
import numpy as np
from src.hand_tracker import HandTracker
from src.virtual_renderer import VirtualRenderer

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    renderer = VirtualRenderer()
    
    prev_dist = 0
    prev_angle = 0

    print("--- VIRTUAL 3D AIR DRAWING ---")
    print("1 JARI: Menggambar")
    print("2 TANGAN: Geser, Putar, Perbesar")
    print("Tombol 'c': Hapus Layar")
    print("Tombol 'q': Keluar")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        results = tracker.process_frame(frame)
        hands = tracker.get_hand_info(results)

        # ─── LOGIKA 1 TANGAN (DRAWING) ───────────────────
        if len(hands) == 1:
            hand = hands[0]
            ix, iy = int(hand['index'][0] * w), int(hand['index'][1] * h)

            if hand['is_drawing']:
                cv2.putText(frame, "STATUS: DRAWING", (20, 50), 1, 1.5, (0, 0, 255), 2)
                cv2.circle(frame, (ix, iy), 15, (0, 0, 255), -1) # Kursor Merah
                renderer.add_point(hand['index'])
            else:
                cv2.putText(frame, "STATUS: POINTER", (20, 50), 1, 1.5, (0, 255, 0), 2)
                cv2.circle(frame, (ix, iy), 15, (255, 255, 255), 2) # Kursor Putih
                renderer.finish_stroke()
            
            prev_dist = 0

        # ─── LOGIKA 2 TANGAN (MANIPULATION) ─────────────
        elif len(hands) == 2:
            cv2.putText(frame, "STATUS: MANIPULATING (3D)", (20, 50), 1, 1.5, (255, 255, 0), 2)
            renderer.finish_stroke()
            
            p1, p2 = hands[0]['center'], hands[1]['center']
            
            # 1. Geser (Translation)
            renderer.offset = (p1 + p2) / 2

            # 2. Perbesar/Kecil (Scale)
            dist = np.linalg.norm(p1 - p2)
            if prev_dist > 0:
                renderer.scale += (dist - prev_dist) * 2
                renderer.scale = max(0.1, renderer.scale)
            
            # 3. Putar 3D (Rotation)
            # Rotasi Y (yaw) dari sudut horizontal antara tangan
            # Rotasi X (pitch) dari perbedaan tinggi tangan relatif terhadap jarak
            if prev_dist > 0:
                # Rotasi Z (seperti sebelumnya)
                angle_z = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
                renderer.angles[2] += (angle_z - prev_angle)
                
                # Rotasi Y (gerakan maju mundur tangan relatif satu sama lain)
                angle_y = (p2[2] - p1[2]) * 5
                renderer.angles[1] += angle_y * 0.1
                
                # Rotasi X (gerakan naik turun tangan relatif satu sama lain)
                angle_x = (p2[1] - p1[1]) * 2
                renderer.angles[0] += angle_x * 0.01

            prev_dist = dist
            prev_angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            
            # Visualisasi kursor di telapak tangan
            for h_info in hands:
                hx, hy = int(h_info['center'][0] * w), int(h_info['center'][1] * h)
                cv2.circle(frame, (hx, hy), 20, (255, 255, 0), 2)

        # Render semua gambar
        frame = renderer.draw(frame)
        
        cv2.imshow("Air Drawing & Gesture Control", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            renderer.all_strokes = []
            renderer.current_stroke = []

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


