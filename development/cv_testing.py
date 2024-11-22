import cv2
import json
import time
import logging
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import numpy as np
from classes import object_index_json

'''
This program was used to develop the desired computer vision algorithm using the data from the yolo model.
'''


# Configurations and Constants
class Config:
    MQTT_BROKER = "10.243.82.33"  # Replace with actual broker IP
    CONTROL_TOPIC = "bot/motors"
    Kp = 0.1  # Proportional gain
    Kd = 0.0  # Derivative gain
    TARGET_IMAGE_WIDTH = 640  # Consistent image width
    SAFE_ZONE_RADIUS = 5  # Safe zone radius in pixels
    TARGET_FPS = 10  # Target FPS for frame processing

    # Motor speed limits
    MAX_LEFT_SPEED = 200
    MAX_RIGHT_SPEED = 200
    MIN_LEFT_PWM = 50  # Minimum PWM value for left motor
    MIN_RIGHT_PWM = 50  # Minimum PWM value for right motor
    CAPTURE_DEVICE = 0

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Motor Control Class
class MotorControl:
    def __init__(self, mqtt_client, control_topic):
        self.client = mqtt_client
        self.control_topic = control_topic
        self.previous_error = 0
        self.left_pwm = 0
        self.right_pwm = 0

    def _publish_command(self, pwm_values):
        payload = json.dumps({"pwm": pwm_values})
        self.client.publish(self.control_topic, payload)
        logger.info(f"Published motor command: {payload}")

    def move(self, left_pwm, right_pwm):
        self.left_pwm = left_pwm
        self.right_pwm = right_pwm

        # Ensure minimum PWM thresholds are respected
        left_pwm = self._apply_min_pwm(left_pwm, Config.MIN_LEFT_PWM)
        right_pwm = self._apply_min_pwm(right_pwm, Config.MIN_RIGHT_PWM)

        # Saturate motor speeds to the maximum limits
        left_pwm = max(-Config.MAX_LEFT_SPEED, min(Config.MAX_LEFT_SPEED, left_pwm))
        right_pwm = max(-Config.MAX_RIGHT_SPEED, min(Config.MAX_RIGHT_SPEED, right_pwm))

        self._publish_command([left_pwm, right_pwm])

    def _apply_min_pwm(self, pwm, min_pwm):
        """Ensure the PWM value respects the minimum threshold for movement."""
        if pwm > 0:  # Moving forward
            return max(pwm, min_pwm)
        elif pwm < 0:  # Moving backward
            return min(pwm, -min_pwm)
        return 0  # No movement

# Vision System Class
class VisionSystem:
    def __init__(self):
        self.current_model = None
        self.current_model_name = ""

    def load_model(self, model_name):
        if self.current_model_name != model_name:
            model_path = f"yolo_models/{model_name}.pt"
            self.current_model = YOLO(model_path, verbose=False)
            self.current_model_name = model_name
            logger.info(f"Model '{model_name}' loaded successfully.")

    def process_detection_frame(self, frame):
        target_height = int(Config.TARGET_IMAGE_WIDTH * frame.shape[0] / frame.shape[1])
        resized_frame = cv2.resize(frame, (Config.TARGET_IMAGE_WIDTH, target_height))
        return self.current_model(resized_frame, conf=0.6, verbose=False)

# Visual Navigation Function
def visual_navigation(frame, target_object, reference_object):
    """
    Navigate to a specific target object, prioritizing proximity to a reference object.
    
    Parameters:
        frame: The current frame from the video capture.
        target_object: The primary object to navigate to (e.g., "person").
        reference_object: The secondary object used for target selection refinement (e.g., "cup").

    Returns:
        processed_frame: The frame with annotations.
        objects_present: Boolean indicating if both target and reference objects are present.
    """
    target_index = object_index_json.get(target_object)
    reference_index = object_index_json.get(reference_object)

    if target_index is None or reference_index is None:
        logger.error("Both target and reference objects must be provided and valid.")
        return frame, False

    # Process the frame and get YOLO results
    results = vision_system.process_detection_frame(frame)
    detections = results[0].boxes  # YOLO detected boxes

    # Filter detections for target and reference objects
    target_boxes = [box for box in detections if box.cls == target_index]
    reference_boxes = [box for box in detections if box.cls == reference_index]

    # Assign filtered boxes back to the results object for plotting
    results[0].boxes = target_boxes + reference_boxes
    processed_frame = results[0].plot()

    # Image center
    img_center_x = Config.TARGET_IMAGE_WIDTH // 2
    safe_zone_radius = Config.SAFE_ZONE_RADIUS

    # Draw safe zone (vertical lines)
    safe_left = img_center_x - safe_zone_radius
    safe_right = img_center_x + safe_zone_radius
    cv2.line(processed_frame, (safe_left, 0), (safe_left, processed_frame.shape[0]), (0, 255, 0), 2)
    cv2.line(processed_frame, (safe_right, 0), (safe_right, processed_frame.shape[0]), (0, 255, 0), 2)

    objects_present = bool(target_boxes) and bool(reference_boxes)

    if not reference_boxes:
        # Reference object not found
        cv2.putText(
            processed_frame,
            f"No {reference_object} found",
            (10, processed_frame.shape[0] - 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        motor_control.move(0, 0)
        return processed_frame, False

    # Navigation logic
    chosen_target = None
    if target_boxes:
        # Find the target closest to any reference object
        chosen_target = min(
            target_boxes,
            key=lambda tb: min(
                np.linalg.norm(
                    [
                        (tb.xyxy[0][0] + tb.xyxy[0][2]) / 2 - (rb.xyxy[0][0] + rb.xyxy[0][2]) / 2,
                        (tb.xyxy[0][1] + tb.xyxy[0][3]) / 2 - (rb.xyxy[0][1] + rb.xyxy[0][3]) / 2,
                    ]
                )
                for rb in reference_boxes
            ),
        )
    else:
        # Target object not found
        cv2.putText(
            processed_frame,
            f"No {target_object} found",
            (10, processed_frame.shape[0] - 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        motor_control.move(0, 0)
        return processed_frame, False

    # Calculate error and annotate
    box_center_x = int((chosen_target.xyxy[0][0] + chosen_target.xyxy[0][2]) / 2)
    error = box_center_x - img_center_x
    cv2.line(processed_frame, (box_center_x, 0), (box_center_x, processed_frame.shape[0]), (255, 0, 0), 2)
    cv2.putText(processed_frame, f"Error: {error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # PD control for motor adjustment
    derivative = error - motor_control.previous_error
    adjustment = Config.Kp * error + Config.Kd * derivative
    motor_control.previous_error = error

    # Determine motor speeds
    if abs(error) <= safe_zone_radius:
        # If within the safe zone, move forward
        left_pwm = right_pwm = 150  # Forward speed
    else:
        # Adjust motor speeds to turn towards the target
        left_pwm = 100 - adjustment
        right_pwm = 150 + adjustment

    # Send motor commands
    motor_control.move(int(left_pwm), int(right_pwm))

    # Annotate motor PWM values
    cv2.putText(
        processed_frame,
        f"Left PWM: {motor_control.left_pwm}",
        (10, processed_frame.shape[0] - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        processed_frame,
        f"Right PWM: {motor_control.right_pwm}",
        (processed_frame.shape[1] - 300, processed_frame.shape[0] - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    return processed_frame, objects_present

# Main loop for video processing and motor control
def main():
    vid = cv2.VideoCapture(Config.CAPTURE_DEVICE)
    mqtt_client = mqtt.Client()
    mqtt_client.connect(Config.MQTT_BROKER)
    global motor_control, vision_system
    motor_control = MotorControl(mqtt_client, Config.CONTROL_TOPIC)
    vision_system = VisionSystem()

    vision_system.load_model('yolo11s')  # Load the detection model
    previous_time = time.time()

while True:
    ret, frame = vid.read()
    if not ret:
        break

    # Specify navigation targets
    processed_frame, objects_present = visual_navigation(frame, target_object="person", reference_object="cup")

    # Log object presence status
    logger.info(f"Objects Present: {objects_present}")

    # Display the processed frame
    cv2.imshow("Robot Interface", processed_frame)

    # Exit if 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    mqtt_client.loop_stop()
    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()