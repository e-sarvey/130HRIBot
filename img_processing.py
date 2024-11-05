import cv2
from ultralytics import YOLO
import time

model = YOLO('yolo_models/yolo11s.pt', verbose=False)

vid = cv2.VideoCapture(0)

# Initialize time for FPS calculation
prev_time = 0

# Dictionary of detectable objects, reversed for name-to-ID mapping
object_classes = {
    'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7, 
    'boat': 8, 'traffic light': 9, 'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13, 
    'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21, 
    'zebra': 22, 'giraffe': 23, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 
    'frisbee': 29, 'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34, 
    'baseball glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis racket': 38, 'bottle': 39, 'wine glass': 40, 
    'cup': 41, 'fork': 42, 'knife': 43, 'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48, 
    'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55, 
    'chair': 56, 'couch': 57, 'potted plant': 58, 'bed': 59, 'dining table': 60, 'toilet': 61, 'tv': 62, 
    'laptop': 63, 'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67, 'microwave': 68, 'oven': 69, 
    'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74, 'vase': 75, 'scissors': 76, 
    'teddy bear': 77, 'hair drier': 78, 'toothbrush': 79
}

# Retrieve class ID for 'person' directly from the dictionary
PERSON_ID = object_classes['person']

# Default to detect 'cup' along with 'person'
second_object_name = 'cup'
second_object_id = object_classes.get(second_object_name)

# Function to calculate distance from the center of the bounding box to the center of the frame
def getDistanceFromCenter(x_min, x_max, frame_width):
    frame_center_x = frame_width // 2
    object_center_x = (x_min + x_max) // 2
    distance = object_center_x - frame_center_x
    return distance, object_center_x

def obj_detection(frame):
    global prev_time

    # Resize the frame to 640 pixels wide
    original_height, original_width, _ = frame.shape
    target_width = 640
    target_height = int((target_width / original_width) * original_height)
    resized_frame = cv2.resize(frame, (target_width, target_height))
    resized_frame = cv2.flip(resized_frame, 1) # 1 mirrors image
    # YOLO model detection on the color frame
    results = model(resized_frame, conf=0.6, verbose=False)

    # Filter results to include only people and the specified object
    filtered_boxes = [box for box in results[0].boxes if box.cls in [PERSON_ID, second_object_id]]
    
    # Update results with filtered boxes only
    results[0].boxes = filtered_boxes

    # Annotate the frame with bounding boxes for the specified objects
    annotated_frame = results[0].plot()
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display image size and FPS
    cv2.putText(annotated_frame, f"Size: {target_width}x{target_height}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (target_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Process detections for persons and the selected object, and calculate distance from the center
    for box in filtered_boxes:
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
        distance, object_center_x = getDistanceFromCenter(x_min, x_max, target_width)
        cv2.line(annotated_frame, (target_width // 2, target_height // 2), (object_center_x, target_height // 2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Dist: {distance} px", (object_center_x, target_height // 2 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the annotated frame
    cv2.namedWindow("Annotated Feed", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Annotated Feed", cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow("Annotated Feed", annotated_frame)

# Update second_object_id based on the chosen object name (this can be modified dynamically if needed)
def set_second_object(object_name):
    global second_object_id
    if object_name in object_classes:
        second_object_id = object_classes[object_name]
    else:
        print(f"Object '{object_name}' not found in detectable objects.")

try:
    # Set up initial object detection
    set_second_object(second_object_name)

    # Make the OpenCV window stay on top
    cv2.namedWindow("Annotated Feed", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Annotated Feed", cv2.WND_PROP_TOPMOST, 1)

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        obj_detection(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print('Keyboard interrupted!')
finally:
    vid.release()
    cv2.destroyAllWindows()
