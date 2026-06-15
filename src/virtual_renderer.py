import cv2
import numpy as np

class VirtualRenderer:
    def __init__(self):
        self.all_strokes = []       # List dari list poin (kumpulan garis)
        self.current_stroke = []    # Garis yang sedang dibuat
        self.offset = np.array([0.5, 0.5]) # Posisi pusat gambar (tengah layar)
        self.scale = 1.0
        self.angle = 0.0

    def add_point(self, point):
        """Menyimpan poin dalam model-space (relatif terhadap offset, scale, rotation)"""
        # Invers rotasi
        c, s = np.cos(self.angle), np.sin(self.angle)
        inv_rot = np.array([[c, s], [-s, c]])
        
        # Hitung posisi relatif
        rel_p = (np.array(point) - self.offset) / self.scale
        stored_p = inv_rot @ rel_p
        self.current_stroke.append(stored_p)

    def finish_stroke(self):
        if self.current_stroke:
            self.all_strokes.append(self.current_stroke)
            self.current_stroke = []

    def draw(self, frame):
        h, w = frame.shape[:2]
        
        # Matriks Rotasi
        c, s = np.cos(self.angle), np.sin(self.angle)
        rotation_matrix = np.array([[c, -s], [s, c]])

        # Render semua coretan
        strokes_to_draw = self.all_strokes + ([self.current_stroke] if self.current_stroke else [])

        for stroke in strokes_to_draw:
            points_to_render = []
            for p in stroke:
                # Transformasi balik: Model -> Screen
                p_transformed = (rotation_matrix @ np.array(p)) * self.scale
                p_final = p_transformed + self.offset
                
                px = int(p_final[0] * w)
                py = int(p_final[1] * h)
                points_to_render.append((px, py))

            for i in range(1, len(points_to_render)):
                cv2.line(frame, points_to_render[i-1], points_to_render[i], (0, 255, 0), 4)
        return frame
