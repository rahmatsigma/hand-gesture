import cv2
import numpy as np
from src.shape_detector import ShapeDetector

class VirtualRenderer:
    def __init__(self):
        self.all_strokes = []       
        self.current_stroke = []    
        self.offset = np.array([0.5, 0.5, 0.0]) # Pusat 3D
        self.scale = 1.0
        self.angles = np.array([0.0, 0.0, 0.0]) # Rotasi X, Y, Z
        self.detector = ShapeDetector()

    def get_rotation_matrix(self):
        ax, ay, az = self.angles
        # Rotasi X
        rx = np.array([[1, 0, 0],
                       [0, np.cos(ax), -np.sin(ax)],
                       [0, np.sin(ax), np.cos(ax)]])
        # Rotasi Y
        ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                       [0, 1, 0],
                       [-np.sin(ay), 0, np.cos(ay)]])
        # Rotasi Z
        rz = np.array([[np.cos(az), -np.sin(az), 0],
                       [np.sin(az), np.cos(az), 0],
                       [0, 0, 1]])
        return rz @ ry @ rx

    def add_point(self, point):
        """Menyimpan poin dalam model-space (3D)"""
        rot = self.get_rotation_matrix()
        inv_rot = rot.T # Transpose matrix rotasi adalah inversnya
        
        # Hitung posisi relatif (point berasumsi [x, y, z])
        rel_p = (np.array(point) - self.offset) / self.scale
        stored_p = inv_rot @ rel_p
        self.current_stroke.append(stored_p)

    def finish_stroke(self):
        if self.current_stroke:
            perfected_stroke = self.detector.detect_and_perfect(self.current_stroke)
            self.all_strokes.append(perfected_stroke)
            self.current_stroke = []

    def draw(self, frame):
        h, w = frame.shape[:2]
        rot = self.get_rotation_matrix()
        
        strokes_to_draw = self.all_strokes + ([self.current_stroke] if self.current_stroke else [])

        for stroke in strokes_to_draw:
            points_to_render = []
            for p in stroke:
                # Transformasi Model -> World
                p_world = (rot @ np.array(p)) * self.scale + self.offset
                
                # Proyeksi Perspektif Sederhana
                # z_eff: kedalaman relatif (0.5 adalah tengah)
                z_eff = p_world[2] + 2.0 # Geser sedikit agar tidak di depan kamera
                if z_eff <= 0.1: z_eff = 0.1
                
                # Proyeksi ke layar
                fov = 1.5
                px = int((p_world[0] - 0.5) * fov / z_eff * w + w/2)
                py = int((p_world[1] - 0.5) * fov / z_eff * h + h/2)
                
                points_to_render.append((px, py))

            for i in range(1, len(points_to_render)):
                # Gambar dengan ketebalan yang bervariasi berdasarkan kedalaman?
                # Untuk sekarang tetap konstan
                cv2.line(frame, points_to_render[i-1], points_to_render[i], (0, 255, 0), 4)
        return frame
