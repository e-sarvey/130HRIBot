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
    Kp = 0.1  # Proportional gain
    Kd = 0.05  # Derivative gain
    TARGET_IMAGE_WIDTH = 640  # Consistent image width
    SAFE_ZONE_RADIUS = 50  # Safe zone radius in pixels
    TARGET_FPS = 10  # Target FPS for frame processing

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
        self._publish_command([left_pwm, right_pwm])

# Vision System Class
class VisionSystem:
    def __init__(self):
        self.current_model = None
        self.current_model_name = ""
        self.PERSON_ID = 0  # Assuming detection of 'person'
        self.OBJECT_ID = 39  # Assuming detection of 'bottle'

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
        bottle_boxes = [box for box in results[0].boxes if box.cls == self.OBJECT_ID]
        
        target_box = None
        if people_boxes and bottle_boxes:
            bottle_center_x = int((bottle_boxes[0].xyxy[0][0] + bottle_boxes[0].xyxy[0][2]) / 2)
            closest_person = min(
                people_boxes,
                key=lambda person_box: abs(int((person_box.xyxy[0][0] + person_box.xyxy[0][2]) / 2) - bottle_center_x)
            )
            target_box = closest_person  # The closest person to the bottle

        return results[0].plot(), people_boxes, bottle_boxes, target_box

    def process_pose_frame(self, frame):
        target_height = int(Config.TARGET_IMAGE_WIDTH * frame.shape[0] / frame.shape[1])
        resized_frame = cv2.resize(frame, (Config.TARGET_IMAGE_WIDTH, target_height))
        results = self.current_model(resized_frame, conf=0.6, verbose=False)

        person_with_raised_hand = None
        max_box_size = 0

        for person in results[0].keypoints:
            keypoints = person.data[0]  # Access the keypoints tensor

            # Define joint positions based on COCO keypoint indices
            left_shoulder = keypoints[5][:2]  # [x, y]
            right_shoulder = keypoints[6][:2]
            left_elbow = keypoints[7][:2]
            right_elbow = keypoints[8][:2]
            left_wrist = keypoints[9][:2] if keypoints[9][0] > 0 and keypoints[9][1] > 0 else None
            right_wrist = keypoints[10][:2] if keypoints[10][0] > 0 and keypoints[10][1] > 0 else None

            # Calculate angles for both arms only if wrists are present
            left_hand_raised = False
            right_hand_raised = False
            if left_wrist:
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, (left_shoulder[0], left_shoulder[1] - 50))
                left_hand_raised = is_hand_raised(left_elbow, left_shoulder, left_wrist, left_elbow_angle, left_shoulder_angle, angle_offset=15)
                
            if right_wrist:
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, (right_shoulder[0], right_shoulder[1] - 50))
                right_hand_raised = is_hand_raised(right_elbow, right_shoulder, right_wrist, right_elbow_angle, right_shoulder_angle, angle_offset=15)

            # Draw an indicator if a hand is raised and print "Hand raised"
            if left_hand_raised:
                cv2.circle(resized_frame, (int(left_wrist[0]), int(left_wrist[1])), 10, (0, 0, 255), 2)
                print("Hand raised (left)")
            if right_hand_raised:
                cv2.circle(resized_frame, (int(right_wrist[0]), int(right_wrist[1])), 10, (0, 0, 255), 2)
                print("Hand raised (right)")

            # Estimate bounding box size for the person using the range of keypoints
            x_values = [point[0] for point in keypoints if point[0] > 0]
            y_values = [point[1] for point in keypoints if point[1] > 0]
            
            if x_values and y_values:
                box_width = max(x_values) - min(x_values)
                box_height = max(y_values) - min(y_values)
                box_size = box_width * box_height

                # Track the largest person with a raised hand
                if (left_hand_raised or right_hand_raised) and box_size > max_box_size:
                    max_box_size = box_size
                    person_with_raised_hand = person

        # Use the `plot` method to visualize keypoints and skeleton
        return results[0].plot(), person_with_raised_hand

# Additional Helper Functions
def calculate_angle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    norm_product = np.linalg.norm(ba) * np.linalg.norm(bc)
    
    # Check for zero denominator to avoid invalid value for cosine
    if norm_product == 0:
        return 0  # Return 0 or another default value if points are collinear or overlapping

    cosine_angle = np.dot(ba, bc) / norm_product
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))  # Clip to handle any floating-point precision errors
    return angle

def is_hand_raised(elbow, shoulder, wrist, elbow_angle, shoulder_angle, angle_offset=0):
    hand_above_shoulder = wrist[1] < shoulder[1]
    elbow_flexed = elbow_angle < (90 + angle_offset)
    shoulder_positioned_for_raise = shoulder_angle < (60 + angle_offset)
    return hand_above_shoulder and elbow_flexed and shoulder_positioned_for_raise

# MQTT Handler Class
class MQTTHandler:
    def __init__(self, broker):
        self.client = mqtt.Client()
        self.client.connect(broker)
        self.client.loop_start()

# Main loop for video and mode switching
def main():
    vid = cv2.VideoCapture(0)
    mqtt_client = mqtt.Client()
    mqtt_client.connect(Config.MQTT_BROKER)
    motor_control = MotorControl(mqtt_client, Config.CONTROL_TOPIC)
    vision_system = VisionSystem()
    mqtt_handler = MQTTHandler(Config.MQTT_BROKER)

    mode = 'detection'
    vision_system.load_model('yolo11s' if mode == 'detection' else 'yolo11s-pose')
    previous_time = time.time()

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        if mode == 'detection':
            processed_frame, people_boxes, bottle_boxes, target_box = vision_system.process_detection_frame(frame)
        else:
            processed_frame, person_with_raised_hand = vision_system.process_pose_frame(frame)
            target_box = person_with_raised_hand

        cv2.imshow("Robot Interface", processed_frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('m'):
            mode = 'pose' if mode == 'detection' else 'detection'
            vision_system.load_model('yolo11s' if mode == 'detection' else 'yolo11s-pose')

    mqtt_client.loop_stop()
    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
