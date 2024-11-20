from enum import Enum
import cv2
import json
import time
import logging
from ultralytics import YOLO
import paho.mqtt.client as mqtt

# Configurations and Constants
class Config:
    # MQTT_BROKER = "10.0.0.25"
    MQTT_BROKER = "10.243.82.33"  # Replace with actual broker IP
    STATUS_TOPIC = "bot/state"
    SENSORS_TOPIC = "bot/sensors"
    CONTROL_TOPIC = "bot/motors"
    Kp = 0.5  # Proportional gain
    Kd = 0.0  # Derivative gain
    TARGET_IMAGE_WIDTH = 640  # Consistent image width
    SAFE_ZONE_RADIUS = 5  # Safe zone radius in pixels
    TARGET_FPS = 10  # Target FPS for frame processing
    MAX_LEFT_SPEED = 200
    MAX_RIGHT_SPEED = 200
    MIN_LEFT_PWM = 50  # Minimum PWM value for left motor
    MIN_RIGHT_PWM = 50  # Minimum PWM value for right motor
    STOP_DISTANCE_THRESHOLD = 50  # Lidar distance threshold (cm)
    DETECTION_TIMEOUT = 45  # Timeout for detecting a target (seconds)
    NAVIGATION_TIMEOUT = 3  # Timeout for losing a target during navigation (seconds)

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State Enum
class State(Enum):
    BOOT_UP = "boot_up"
    AWAIT_ACTIVATION = "await_activation"
    DETECT_PICKUP_LOCATION = "detect_pickup_location"
    RETRIEVAL = "retrieval"
    PROCUREMENT = "procurement"
    DETECT_DELIVERY_LOCATION = "detect_delivery_location"
    SHIPPING = "shipping"
    DELIVERY = "delivery"

# Motor Control Class
class MotorControl:
    def __init__(self, mqtt_client, control_topic):
        self.client = mqtt_client
        self.control_topic = control_topic
        self.previous_error = 0

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

    def stop(self):
        self._publish_command([0, 0])

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
        self.object_id = 41  # Secondary object (e.g., 'cup')
        self.load_model('yolo11s')  # Load the model at initialization

    def load_model(self, model_name):
        try:
            if self.current_model_name != model_name:
                model_path = f"yolo_models/{model_name}.pt"
                self.current_model = YOLO(model_path, verbose=False)
                self.current_model_name = model_name
                logger.info(f"Model '{model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            self.current_model = None

    def process_detection_frame(self, frame, detect_cup=False):
        if not self.current_model:
            logger.error("YOLO model not loaded. Ensure the model is available at boot-up.")
            raise RuntimeError("YOLO model not loaded. Ensure it is available at boot-up.")

        target_height = int(Config.TARGET_IMAGE_WIDTH * frame.shape[0] / frame.shape[1])
        resized_frame = cv2.resize(frame, (Config.TARGET_IMAGE_WIDTH, target_height))
        results = self.current_model(resized_frame, conf=0.6, verbose=False)

        # Filter for person and optional secondary object
        valid_classes = [self.PERSON_ID]
        if detect_cup and self.object_id is not None:
            valid_classes.append(self.object_id)

        filtered_boxes = [box for box in results[0].boxes if box.cls in valid_classes]

        # Select the largest target box
        target_box = None
        if filtered_boxes:
            target_box = max(filtered_boxes, key=lambda box: (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]))

        return results[0].plot(), target_box

# Robot State Machine
class RobotStateMachine:
    def __init__(self, motor_control, vision_system, mqtt_client):
        self.current_state = State.BOOT_UP
        self.motor_control = motor_control
        self.vision_system = vision_system
        self.printed_states = set()
        self.state_start_time = None  # Track when the state started
        self.mqtt_client = mqtt_client

        self.state_actions = {
            State.BOOT_UP: self._boot_up,
            State.AWAIT_ACTIVATION: self._await_activation,
            State.DETECT_PICKUP_LOCATION: self._detect_pickup_location,
            State.RETRIEVAL: self._retrieval,
            State.PROCUREMENT: self._procurement,
            State.DETECT_DELIVERY_LOCATION: self._detect_delivery_location,
            State.SHIPPING: self._shipping,
            State.DELIVERY: self._delivery
        }
    def handle_event(self, event):
        logger.info(f"Received event: {event}, Current state: {self.current_state}")
        if self.current_state == State.BOOT_UP and event == "robot_initialized":
            self.transition_to_state(State.AWAIT_ACTIVATION)
        elif self.current_state == State.AWAIT_ACTIVATION and event == "imu_tap_detected":
            self.transition_to_state(State.DETECT_PICKUP_LOCATION)
        elif self.current_state == State.DETECT_PICKUP_LOCATION and event == "person_detected":
            logger.info("Person detected. Stopping motors and transitioning to retrieval.")
            self.motor_control.stop()
            self.transition_to_state(State.RETRIEVAL)
        elif self.current_state == State.RETRIEVAL and event == "lidar_threshold_met":
            self.transition_to_state(State.PROCUREMENT)
        elif self.current_state == State.SHIPPING and event == "lidar_threshold_met":
            self.transition_to_state(State.DELIVERY)
        elif self.current_state == State.PROCUREMENT and event == "cup_placed":
            self.transition_to_state(State.DETECT_DELIVERY_LOCATION)
        elif self.current_state == State.DELIVERY and event == "cup_removed":
            self.transition_to_state(State.AWAIT_ACTIVATION)

    def transition_to_state(self, next_state):
        logger.info(f"Transitioning to state: {next_state.name}")
        self.mqtt_client.publish(Config.STATUS_TOPIC, json.dumps({"state": f"start_{next_state.value}"}))
        self.current_state = next_state
        self.state_start_time = time.time()
        self.printed_states.clear()

    def execute_current_state(self, frame):
        if self.current_state in self.state_actions:
            return self.state_actions[self.current_state](frame)
        return frame

    #######################
    ## STATE DEFINITIONS ##
    #######################
    def _boot_up(self, frame):
        if State.BOOT_UP not in self.printed_states:
            logger.info("Booting up... Waiting for initialization confirmation.")
            self.printed_states.add(State.BOOT_UP)
        return frame

    def _await_activation(self, frame):
        if State.AWAIT_ACTIVATION not in self.printed_states:
            logger.info("Awaiting activation...")
            self.printed_states.add(State.AWAIT_ACTIVATION)
        return frame
    
    def _detect_pickup_location(self, frame):
        if State.DETECT_PICKUP_LOCATION not in self.printed_states:
            logger.info("Detecting pickup location: Slowly turning left.")
            self.printed_states.add(State.DETECT_PICKUP_LOCATION)
            self.motor_control.move(-70, 70)  # Slow turn left

        processed_frame, target_box = self.vision_system.process_detection_frame(frame, detect_cup=True)
        if target_box:
            logger.info("Person and cup detected in pickup location.")
            self.motor_control.stop()
            self.mqtt_client.publish(Config.SENSORS_TOPIC, json.dumps({"event": "person_detected"}))
            logger.info("Event published: person_detected")
        elif time.time() - self.state_start_time > Config.DETECTION_TIMEOUT:
            logger.warning("Timeout in pickup location. Returning to await activation.")
            self.transition_to_state(State.AWAIT_ACTIVATION)
        return processed_frame

    def _retrieval(self, frame):
        if State.RETRIEVAL not in self.printed_states:
            logger.info("Retrieval: Navigating to target.")
            self.printed_states.add(State.RETRIEVAL)

        processed_frame, target_box = self.vision_system.process_detection_frame(frame)
        if target_box:
            box_center_x = int((target_box.xyxy[0][0] + target_box.xyxy[0][2]) / 2)
            img_center_x = Config.TARGET_IMAGE_WIDTH // 2
            error = box_center_x - img_center_x

            derivative = error - self.motor_control.previous_error
            adjustment = Config.Kp * error + Config.Kd * derivative
            self.motor_control.previous_error = error

            left_pwm = 100 - adjustment
            right_pwm = 100 + adjustment
            self.motor_control.move(int(left_pwm), int(right_pwm))
        else:
            logger.warning("Target lost in retrieval.")
            if time.time() - self.state_start_time > Config.NAVIGATION_TIMEOUT:
                logger.warning("Target lost timeout. Returning to detect pickup location.")
                self.transition_to_state(State.DETECT_PICKUP_LOCATION)

        return processed_frame

    def _procurement(self, frame):
        if State.PROCUREMENT not in self.printed_states:
            logger.info("Procurement: Waiting for cup placement.")
            self.printed_states.add(State.PROCUREMENT)
            self.motor_control.stop()
        return frame

    def _detect_delivery_location(self, frame):
        if State.DETECT_DELIVERY_LOCATION not in self.printed_states:
            logger.info("Detecting delivery location: Slowly turning left.")
            self.printed_states.add(State.DETECT_DELIVERY_LOCATION)
            self.motor_control.move(-70, 70)  # Slow turn left

        processed_frame, target_box = self.vision_system.process_detection_frame(frame)
        if target_box:
            logger.info("Person detected in delivery location.")
            self.motor_control.stop()
            self.transition_to_state(State.SHIPPING)
        elif time.time() - self.state_start_time > Config.DETECTION_TIMEOUT:
            logger.warning("Timeout in delivery location. Returning to await activation.")
            self.transition_to_state(State.AWAIT_ACTIVATION)
        return processed_frame

    def _shipping(self, frame):
        if State.SHIPPING not in self.printed_states:
            logger.info("Shipping: Navigating to delivery point.")
            self.printed_states.add(State.SHIPPING)

        processed_frame, target_box = self.vision_system.process_detection_frame(frame)
        if target_box:
            box_center_x = int((target_box.xyxy[0][0] + target_box.xyxy[0][2]) / 2)
            img_center_x = Config.TARGET_IMAGE_WIDTH // 2
            error = box_center_x - img_center_x

            derivative = error - self.motor_control.previous_error
            adjustment = Config.Kp * error + Config.Kd * derivative
            self.motor_control.previous_error = error

            left_pwm = 100 - adjustment
            right_pwm = 100 + adjustment
            self.motor_control.move(int(left_pwm), int(right_pwm))
        else:
            logger.warning("Target lost in shipping.")
            if time.time() - self.state_start_time > Config.NAVIGATION_TIMEOUT:
                logger.warning("Target lost timeout. Returning to detect delivery location.")
                self.transition_to_state(State.DETECT_DELIVERY_LOCATION)

        return processed_frame

    def _delivery(self, frame):
        if State.DELIVERY not in self.printed_states:
            logger.info("Delivery: Waiting for cup removal.")
            self.printed_states.add(State.DELIVERY)
            self.motor_control.stop()
        return frame

# MQTT Handler
class MQTTHandler:
    def __init__(self, broker, state_machine):
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.state_machine = state_machine
        self.client.connect(broker)
        self.client.loop_start()

    def _on_connect(self, client, userdata, flags, rc):
        logger.info("Connected to MQTT broker.")
        # Subscribe to necessary topics
        self.client.subscribe([(Config.STATUS_TOPIC, 0), (Config.SENSORS_TOPIC, 0)])

    def _on_message(self, client, userdata, msg):
        payload = json.loads(msg.payload.decode())
        logger.info(f"Message received on topic {msg.topic}: {payload}")

        # Handle events or commands based on the topic
        if "event" in payload:
            # Handle event messages from the ESP
            self.state_machine.handle_event(payload["event"])
        elif "state" in payload:
            state_value = payload["state"]
            # Ignore confirmation messages (ending with "_started")
            if state_value.endswith("_started"):
                logger.info(f"State confirmation received: {state_value}")
                return
            # Handle new state commands
            logger.info(f"State change command received: {state_value}")
            self.state_machine.handle_event(state_value)


# Annotate Frame Function
def annotate_frame(frame, state_text=None, fps_text=None):
    if state_text:
        cv2.putText(frame, f"State: {state_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if fps_text:
        text_size, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_x = frame.shape[1] - text_size[0] - 10
        cv2.putText(frame, fps_text, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

# Main Loop
def main():
    vid = cv2.VideoCapture(0)
    mqtt_client = mqtt.Client()
    mqtt_client.connect(Config.MQTT_BROKER)
    motor_control = MotorControl(mqtt_client, Config.CONTROL_TOPIC)
    vision_system = VisionSystem()

    state_machine = RobotStateMachine(motor_control, vision_system, mqtt_client)
    mqtt_handler = MQTTHandler(Config.MQTT_BROKER, state_machine)

    cv2.namedWindow("Robot Interface", cv2.WINDOW_NORMAL)

    previous_time = time.time()

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        #cv2.setWindowProperty("Robot Interface", cv2.WND_PROP_TOPMOST, 1)

        frame = state_machine.execute_current_state(frame)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        frame = annotate_frame(frame, state_text=state_machine.current_state.name, fps_text=f"FPS: {int(fps)}")

        cv2.imshow("Robot Interface", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()