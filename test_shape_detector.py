import numpy as np
from src.shape_detector import ShapeDetector

def test_shape_detector():
    detector = ShapeDetector()

    # 1. Test Rectangle
    # Create a noisy rectangle
    rect_pts = [
        [0.1, 0.1], [0.5, 0.12], [0.9, 0.1], 
        [0.92, 0.5], [0.9, 0.9],
        [0.5, 0.88], [0.1, 0.9],
        [0.08, 0.5], [0.1, 0.1]
    ]
    # Add more points to simulate a real stroke
    interpolated_rect = []
    for i in range(len(rect_pts)-1):
        p1, p2 = np.array(rect_pts[i]), np.array(rect_pts[i+1])
        for t in np.linspace(0, 1, 5):
            interpolated_rect.append(p1 + t*(p2-p1))
    
    perfected_rect = detector.detect_and_perfect(interpolated_rect)
    print(f"Rectangle test: Input points: {len(interpolated_rect)}, Output points: {len(perfected_rect)}")
    if len(perfected_rect) == 5: # 4 vertices + 1 closing
        print("✅ Rectangle detected and perfected!")
    else:
        print("❌ Rectangle detection failed.")

    # 2. Test Circle
    angles = np.linspace(0, 2*np.pi, 30)
    circle_pts = []
    for a in angles:
        # Add some noise
        r = 0.3 + np.random.uniform(-0.02, 0.02)
        circle_pts.append([0.5 + r*np.cos(a), 0.5 + r*np.sin(a)])
    
    perfected_circle = detector.detect_and_perfect(circle_pts)
    print(f"Circle test: Input points: {len(circle_pts)}, Output points: {len(perfected_circle)}")
    if len(perfected_circle) == 40: # Our circle generator uses 40 points
        print("✅ Circle detected and perfected!")
    else:
        print("❌ Circle detection failed.")

    # 3. Test Triangle
    tri_pts = [[0.5, 0.1], [0.9, 0.8], [0.1, 0.8], [0.5, 0.1]]
    interpolated_tri = []
    for i in range(len(tri_pts)-1):
        p1, p2 = np.array(tri_pts[i]), np.array(tri_pts[i+1])
        for t in np.linspace(0, 1, 10):
            interpolated_tri.append(p1 + t*(p2-p1))
    
    perfected_tri = detector.detect_and_perfect(interpolated_tri)
    print(f"Triangle test: Input points: {len(interpolated_tri)}, Output points: {len(perfected_tri)}")
    if len(perfected_tri) == 4: # 3 vertices + 1 closing
        print("✅ Triangle detected and perfected!")
    else:
        print("❌ Triangle detection failed.")

if __name__ == "__main__":
    test_shape_detector()
