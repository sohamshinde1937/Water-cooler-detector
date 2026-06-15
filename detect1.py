import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture(0)  # 0 = laptop camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Increased Confidence: Only look for things the model is 60% sure about
    results = model(frame, conf=0.6)

    annotated_frame = frame.copy()

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get coordinates: x1, y1, x2, y2
            b = box.xyxy[0] 
            x1, y1, x2, y2 = b
            w = x2 - x1 # Width of detected object
            h = y2 - y1 # Height of detected object
            
            # 2. Aspect Ratio Filter:
            # Most water coolers are vertical rectangles (Taller than they are Wide).
            # A human or a square window has a different ratio.
            aspect_ratio = h / w
            
            # 3. Size Filter: 
            # Ignore very small boxes that might be random background noise.
            min_width = 100 
            
            # LOGIC: Only draw the box if it's tall (ratio > 1.2) and large enough
            if aspect_ratio > 1.2 and w > min_width:
                # This plots the box ONLY if it passes our "Not-a-Human/Square" test
                annotated_frame = r.plot()
            else:
                # If it fails, we keep the original frame (no box shown)
                pass

    cv2.imshow("Filtered Water Cooler Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()