import cv2
import time
import os
from ultralytics import YOLO

class AstraVisionPipeline:
    def __init__(self, model_path='models/best.pt', conf_threshold=0.05):
        # Load the YOLOv8 model
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Map YOLO class IDs to the specific route names
        self.class_names = {0: "route_center", 1: "route_left", 2: "route_right"}

    def process_frame(self, frame):
        """Runs inference and formats the output."""
        timestamp = time.time()
        
        # Pass the raw frame but let YOLO safely scale it to 640 natively
        results = self.model.predict(source=frame, imgsz=640, conf=self.conf_threshold, verbose=False)
        
        # Save the output image to see the boxes drawn by YOLO (for debugging)
        results[0].save('data/annotated_output.jpg')
        
        detections = []
        
        # Parse results
        for box in results[0].boxes:
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            
            # Map ID to route string (fallback to raw ID if not in dictionary)
            route_id = self.class_names.get(class_id, f"unknown_{class_id}")
            
            # Append structured output
            detections.append({
                "route_id": route_id,
                "confidence": round(conf, 3),
                "timestamp": timestamp,
                "needs_handoff": conf < 0.75  # Flag for handoff if confidence is marginal
            })
            
        return detections

if __name__ == "__main__":
    import cv2
    
    # Initialize the pipeline
    pipeline = AstraVisionPipeline(conf_threshold=0.05) # 5% confidence for testing
    
    # Explicitly load the image using OpenCV to verify it exists
    img_path = "data/sample_route.jpg"
    frame = cv2.imread(img_path)
    
    if frame is None:
        print(f"Error: Could not read image at {img_path}. Check the filename!")
    else:
        print(f"✅ Image loaded successfully. Image shape: {frame.shape}")
        # Process the frame
        results = pipeline.process_frame(frame)
        print("\n--- Pipeline Output ---")
        for res in results:
            print(res)