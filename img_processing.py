import cv2
from ultralytics import YOLO
import time

# Initialize variables
current_model = None
current_model_name = ""
vid = cv2.VideoCapture(0)

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

# Default to detect 'bottle' along with 'person'
second_object_name = 'bottle'
second_object_id = object_classes.get(second_object_name)

# Function to dynamically load a model based on name
def load_model(model_name):
    global current_model, current_model_name
    if current_model_name == model_name:
        print(f"Model '{model_name}' is already loaded.")
        return
    model_path = f"yolo_models/{model_name}.pt"
    current_model = YOLO(model_path, verbose=False)
    current_model_name = model_name
    print(f"Model '{model_name}' loaded successfully.")

# Function to process frames for the detection model
def process_detection_frame(frame):
    global current_model

    # Resize and mirror the frame for consistency
    target_width, target_height = 640, int(640 * frame.shape[0] / frame.shape[1])
    resized_frame = cv2.resize(frame, (target_width, target_height))
    resized_frame = cv2.flip(resized_frame, 1)

    # YOLO model detection
    results = current_model(resized_frame, conf=0.6, verbose=False)

    # Filter for person and second selected object
    filtered_boxes = [box for box in results[0].boxes if box.cls in [PERSON_ID, second_object_id]]
    results[0].boxes = filtered_boxes

    # Annotate frame with bounding boxes
    annotated_frame = results[0].plot()
    return annotated_frame

# Function to process frames for the pose model
def process_pose_frame(frame):
    global current_model

    # Resize and mirror the frame for consistency
    target_width, target_height = 640, int(640 * frame.shape[0] / frame.shape[1])
    resized_frame = cv2.resize(frame, (target_width, target_height))
    resized_frame = cv2.flip(resized_frame, 1)

    # YOLO pose model detection
    results = current_model(resized_frame, conf=0.6, verbose=False)

    # Annotate frame with pose-specific information
    annotated_frame = results[0].plot()
    return annotated_frame

# Function to calculate and display FPS and image size on a frame
def annotate_fps_and_size(frame, start_time):
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"Size: {frame.shape[1]}x{frame.shape[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

# Main loop
try:
    # Load the default model initially
    load_model('yolo11s')

    # Main video capture loop
    while True:
        start_time = time.time()
        ret, frame = vid.read()
        if not ret:
            break

        # Process and annotate frame based on current model
        if current_model_name == 'yolo11s':
            annotated_frame = process_detection_frame(frame)
        elif current_model_name == 'yolo11s-pose':
            annotated_frame = process_pose_frame(frame)
        else:
            annotated_frame = frame  # Placeholder if you add other models

        # Annotate FPS and size
        annotated_frame = annotate_fps_and_size(annotated_frame, start_time)

        # Display the frame
        cv2.namedWindow("Annotated Feed", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Annotated Feed", cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow("Annotated Feed", annotated_frame)

        # Listen for keys to change models or quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            load_model('yolo11s')  # Load detection model
        elif key == ord('2'):
            load_model('yolo11s-pose')  # Load pose model
        elif key == ord('q'):
            break

except KeyboardInterrupt:
    print("Process interrupted.")
finally:
    vid.release()
    cv2.destroyAllWindows()
