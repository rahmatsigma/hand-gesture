import numpy as np
import cv2

class ShapeDetector:
    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def detect_and_perfect(self, points):
        """
        Detects if points form a shape and returns perfected points.
        Handles both 2D and 3D points.
        """
        if len(points) < 15:
            return points

        original_pts = np.array(points, dtype=np.float32)
        is_3d = original_pts.shape[1] == 3
        
        # Gunakan hanya X, Y untuk deteksi
        pts_2d = original_pts[:, :2]
        avg_z = np.mean(original_pts[:, 2]) if is_3d else 0
        
        # 1. Cek apakah tertutup
        dist_start_end = np.linalg.norm(pts_2d[0] - pts_2d[-1])
        closed_pts = pts_2d
        if dist_start_end > 0.1:
             if dist_start_end < 0.3:
                 closed_pts = np.vstack([pts_2d, pts_2d[0]])
             else:
                 return points

        scale_factor = 1000
        scaled_pts = closed_pts * scale_factor
        perimeter = cv2.arcLength(scaled_pts, True)
        approx = cv2.approxPolyDP(scaled_pts, 0.04 * perimeter, True)
        num_vertices = len(approx)
        
        res_2d = None

        if num_vertices == 3:
            res_2d = approx.reshape(-1, 2)
            res_2d = np.vstack([res_2d, res_2d[0]])
            res_2d = res_2d / scale_factor
        
        elif num_vertices == 4:
            rect = cv2.minAreaRect(scaled_pts)
            box = cv2.boxPoints(rect)
            box = np.vstack([box, box[0]])
            res_2d = box / scale_factor
            
        else:
            center, radius = cv2.minEnclosingCircle(scaled_pts)
            area = cv2.contourArea(scaled_pts)
            if perimeter == 0: return points
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > 0.7:
                num_pts = 40
                angles = np.linspace(0, 2 * np.pi, num_pts)
                perfect_circle = []
                for a in angles:
                    px = center[0] + radius * np.cos(a)
                    py = center[1] + radius * np.sin(a)
                    perfect_circle.append([px, py])
                res_2d = np.array(perfect_circle) / scale_factor

        if res_2d is not None:
            if is_3d:
                # Kembalikan ke 3D dengan Z rata-rata
                res_3d = np.zeros((len(res_2d), 3))
                res_3d[:, :2] = res_2d
                res_3d[:, 2] = avg_z
                return res_3d
            return res_2d

        return points
