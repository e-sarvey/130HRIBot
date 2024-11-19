import cv2
import json
import time
import logging
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import numpy as np

# Configurations and Constants
class Config:
    MQTT_BROKER = "10.243.82.33"  # Replace with actual broker IP
    CONTROL_TOPIC = "bot/motors"
    Kp = 0.5  # Proportional gain
    Kd = 0.0  # Derivative gain
    TARGET_IMAGE_WIDTH = 640  # Consistent image width
    SAFE_ZONE_RADIUS = 5  # Safe zone radius in pixels
    TARGET_FPS = 10  # Target FPS for frame processing

    # Motor speed limits
    MAX_LEFT_SPEED = 200
    MAX_RIGHT_SPEED = 200
    MIN_LEFT_PWM = 50  # Minimum PWM value for left motor
    MIN_RIGHT_PWM = 50  # Minimum PWM value for right motor
    
# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Motor Control Class
class MotorControl:
    def __init__(self, mqtt_client, control_topic):
        self.client = mqtt_client
        self.control_topic = control_topic

    def _publish_command(self, pwm_values):
        payload = json.dumps({"pwm": pwm_values})
        self.client.publish(self.control_topic, payload)
        logger.info(f"Published motor command: {payload}")

    def move(self, left_pwm, right_pwm):
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
        self.PERSON_ID = 0  # Assuming detection of 'person'

    def load_model(self, model_name):
        if self.current_model_name != model_name:
            model_path = f"yolo_models/{model_name}.pt"
            self.current_model = YOLO(model_path, verbose=False)
            self.current_model_name = model_name
            logger.info(f"Model '{model_name}' loaded successfully.")

    def process_detection_frame(self, frame):
        target_height = int(Config.TARGET_IMAGE_WIDTH * frame.shape[0] / frame.shape[1])
        resized_frame = cv2.resize(frame, (Config.TARGET_IMAGE_WIDTH, target_height))
        results = self.current_model(resized_frame, conf=0.6, verbose=False)

        people_boxes = [box for box in results[0].boxes if box.cls == self.PERSON_ID]
        
        # Select the target person box (e.g., the largest box)
        target_box = None
        if people_boxes:
            target_box = max(people_boxes, key=lambda box: (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]))

        return results[0].plot(), target_box

# Main loop for video processing and motor control
def main():
    vid = cv2.VideoCapture(1)
    mqtt_client = mqtt.Client()
    mqtt_client.connect(Config.MQTT_BROKER)
    motor_control = MotorControl(mqtt_client, Config.CONTROL_TOPIC)
    vision_system = VisionSystem()

    vision_system.load_model('yolo11s')  # Load the detection model
    previous_time = time.time()
    
    # Initialize PD control variables
    previous_error = 0
    safe_zone_radius = Config.SAFE_ZONE_RADIUS

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        # Process detection frame
        processed_frame, target_box = vision_system.process_detection_frame(frame)
        
        # Image center
        img_center_x = Config.TARGET_IMAGE_WIDTH // 2
        img_center_y = processed_frame.shape[0] // 2

        # Draw safe zone (vertical lines)
        safe_left = img_center_x - safe_zone_radius
        safe_right = img_center_x + safe_zone_radius
        cv2.line(processed_frame, (safe_left, 0), (safe_left, processed_frame.shape[0]), (0, 255, 0), 2)
        cv2.line(processed_frame, (safe_right, 0), (safe_right, processed_frame.shape[0]), (0, 255, 0), 2)

        if target_box:
            # Calculate the center of the target person's bounding box
            box_center_x = int((target_box.xyxy[0][0] + target_box.xyxy[0][2]) / 2)

            # Compute the centering error
            error = box_center_x - img_center_x

            # Annotate target position
            cv2.line(processed_frame, (box_center_x, 0), (box_center_x, processed_frame.shape[0]), (255, 0, 0), 2)
            cv2.putText(
                processed_frame, 
                f"Error: {error}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2
            )

            # PD control for motor adjustment
            derivative = error - previous_error
            adjustment = Config.Kp * error + Config.Kd * derivative
            previous_error = error

            # Determine motor speeds
            if abs(error) <= safe_zone_radius:
                # If within the safe zone, move forward
                left_pwm = right_pwm = 100  # Forward speed
            else:
                # Adjust motor speeds to turn towards the target
                left_pwm = 100 - adjustment
                right_pwm = 100 + adjustment

            # Send motor commands
            motor_control.move(int(left_pwm), int(right_pwm))
        else:
            # Stop the motors if no target is found
            motor_control.move(0, 0)

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
