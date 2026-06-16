import numpy as np
import cv2

class ShapeDetector:
    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def detect_and_perfect(self, points):
        """
        Detects if points form a shape (triangle, rectangle, circle) 
        and returns perfected points if they do.
        """
        if len(points) < 15: # Terlalu sedikit poin untuk disebut bentuk
            return points

        pts = np.array(points, dtype=np.float32)
        
        # 1. Cek apakah tertutup (jarak awal & akhir)
        dist_start_end = np.linalg.norm(pts[0] - pts[-1])
        # Jika tidak tertutup, mungkin bukan bentuk geometris sempurna yang dimaksud
        # Tapi kita coba tutup saja untuk analisis
        closed_pts = pts
        if dist_start_end > 0.1: # Threshold jarak untuk menganggap "ingin menutup"
             # Jika jaraknya tidak terlalu jauh, kita asumsikan ingin menutup
             if dist_start_end < 0.3:
                 closed_pts = np.vstack([pts, pts[0]])
             else:
                 return points # Terlalu terbuka

        # 2. Polygon Approximation
        # Kita scale sementara untuk akurasi cv2
        scale_factor = 1000
        scaled_pts = closed_pts * scale_factor
        
        # Hitung keliling untuk epsilon
        perimeter = cv2.arcLength(scaled_pts, True)
        approx = cv2.approxPolyDP(scaled_pts, 0.04 * perimeter, True)
        
        num_vertices = len(approx)
        
        # 3. Klasifikasi
        if num_vertices == 3:
            # Segitiga Sempurna
            res = approx.reshape(-1, 2)
            res = np.vstack([res, res[0]]) # Tutup segitiga
            return res / scale_factor
        
        elif num_vertices == 4:
            # Kotak/Persegi Panjang Sempurna
            rect = cv2.minAreaRect(scaled_pts)
            box = cv2.boxPoints(rect)
            box = np.vstack([box, box[0]]) # Tutup kotak
            return box / scale_factor
            
        else:
            # Cek Lingkaran
            center, radius = cv2.minEnclosingCircle(scaled_pts)
            
            # Cek seberapa mirip dengan lingkaran (compactness atau variance radius)
            # Area vs Perimeter
            area = cv2.contourArea(scaled_pts)
            if perimeter == 0: return points
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > 0.7: # Threshold kemiripan lingkaran
                # Buat poin lingkaran sempurna
                num_pts = 40
                angles = np.linspace(0, 2 * np.pi, num_pts)
                perfect_circle = []
                for a in angles:
                    px = center[0] + radius * np.cos(a)
                    py = center[1] + radius * np.sin(a)
                    perfect_circle.append([px, py])
                return np.array(perfect_circle) / scale_factor

        return points
