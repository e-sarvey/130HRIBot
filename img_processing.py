import cv2
from ultralytics import YOLO
import time

model = YOLO('yolo_models/yolo11s.pt', verbose=False)

vid = cv2.VideoCapture(0)

# Initialize time for FPS calculation
prev_time = 0

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

    # YOLO model detection on the resized frame
    results = model(resized_frame, verbose=False)

    # Annotate the resized frame with bounding boxes
    annotated_resized_frame = results[0].plot()
    print(results[0])
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display image size and FPS
    cv2.putText(annotated_resized_frame, f"Size: {target_width}x{target_height}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(annotated_resized_frame, f"FPS: {int(fps)}", (target_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Process detections to find persons and calculate distance from the center
    for result in results[0].boxes:
        # Check if the detected object is a person (class ID for person is 0)
        if result.cls == 0:
            # Get bounding box coordinates
            x_min, y_min, x_max, y_max = map(int, result.xyxy[0])

            # Calculate the distance to the center
            distance, person_center_x = getDistanceFromCenter(x_min, x_max, target_width)

            # Draw line and text on the resized frame
            cv2.line(annotated_resized_frame, (target_width // 2, target_height // 2), (person_center_x, target_height // 2), (0, 255, 0), 2)
            cv2.putText(annotated_resized_frame, f"Dist: {distance} px", (person_center_x, target_height // 2 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the annotated resized frame
    cv2.imshow("Annotated Feed", annotated_resized_frame)

try:
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
