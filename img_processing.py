import cv2
import json
import time
import random
import logging
import paho.mqtt.client as mqtt
from enum import Enum, auto
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration class for constants and MQTT topics
class Config:
    MQTT_BROKER = "10.0.0.25"  # Example IP
    STATUS_TOPIC = "bot/state"
    SENSORS_TOPIC = "bot/sensors"
    CONTROL_TOPIC = "bot/motors"
    Kp = 0.1
    Kd = 0.05
    SAFE_ZONE_RADIUS = 5
    STOP_DISTANCE_THRESHOLD = 50
    VIDEO_WIDTH = 640
    VIDEO_HEIGHT = 480

# Enum for defining robot states
class State(Enum):
    BOOT_UP = auto()
    AWAIT_ACTIVATION = auto()
    DETECT_PICKUP_LOCATION = auto()
    RETRIEVAL = auto()
    PROCUREMENT = auto()
    DETECT_DELIVERY_LOCATION = auto()
    SHIPPING = auto()
    DELIVERY = auto()

# MQTT handler class
class MQTTHandler:
    def __init__(self, broker, state_callback, sensor_callback):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(broker)
        self.state_callback = state_callback
        self.sensor_callback = sensor_callback
        self.client.loop_start()

    def on_connect(self, client, userdata, flags, rc):
        logger.info("Connected to MQTT broker")
        client.subscribe([(Config.STATUS_TOPIC, 0), (Config.SENSORS_TOPIC, 0)])

    def on_message(self, client, userdata, msg):
        payload = json.loads(msg.payload.decode())
        if msg.topic == Config.STATUS_TOPIC:
            self.state_callback(payload)
        elif msg.topic == Config.SENSORS_TOPIC:
            self.sensor_callback(payload)

    def publish(self, topic, message):
        self.client.publish(topic, json.dumps(message))

    def stop(self):
        self.client.loop_stop()

# Motor control class for abstracting motor commands
class MotorControl:
    def __init__(self, mqtt_handler):
        self.mqtt_handler = mqtt_handler

    def spin(self, dir="left", speed=100):
        left_pwm = -speed if dir == "left" else speed
        right_pwm = speed if dir == "left" else -speed
        self._publish_command([left_pwm, right_pwm])

    def stop(self):
        self._publish_command([0, 0])

    def move(self, left_pwm, right_pwm):
        self._publish_command([left_pwm, right_pwm])

    def _publish_command(self, pwm_values):
        self.mqtt_handler.publish(Config.CONTROL_TOPIC, {"pwm": pwm_values})
        logger.debug(f"Published motor command: {pwm_values}")

# Vision system class for image processing and model management
class VisionSystem:
    def __init__(self):
        self.current_model = None
        self.current_model_name = ""
        self.object_classes = {
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
        self.PERSON_ID = self.object_classes['person']
        self.second_object_id = self.object_classes['bottle']

    def load_model(self, model_name):
        if self.current_model_name == model_name:
            logger.debug(f"Model '{model_name}' is already loaded.")
            return
        self.current_model = YOLO(f"yolo_models/{model_name}.pt", verbose=False)
        self.current_model_name = model_name
        logger.info(f"Model '{model_name}' loaded successfully.")

    def process_detection_frame(self, frame):
        target_width, target_height = Config.VIDEO_WIDTH, int(Config.VIDEO_WIDTH * frame.shape[0] / frame.shape[1])
        resized_frame = cv2.resize(frame, (target_width, target_height))
        resized_frame = cv2.flip(resized_frame, 1)
        results = self.current_model(resized_frame, conf=0.6, verbose=False)
        filtered_boxes = [box for box in results[0].boxes if box.cls in [self.PERSON_ID, self.second_object_id]]
        results[0].boxes = filtered_boxes
        return results[0].plot()

# State machine for managing states and executing relevant functions
class RobotStateMachine:
    def __init__(self, motor_control, vision_system):
        self.current_state = State.BOOT_UP
        self.motor_control = motor_control
        self.vision_system = vision_system
        self.previous_error = 0

        # Dictionary to map states to functions
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

    def transition_to_state(self, state):
        logger.info(f"Transitioning to state: {state.name}")
        self.current_state = state

    def execute_current_state(self, frame):
        # Execute the function for the current state
        return self.state_actions.get(self.current_state, lambda frame: frame)(frame)

    def _boot_up(self, frame):
        logger.info("Boot-up: Initializing system and checking connections.")
        return annotate_frame(frame, "Boot-up")

    def _await_activation(self, frame):
        logger.info("Awaiting activation: Monitoring IMU for device tap.")
        return annotate_frame(frame, "Awaiting Activation")

    def _detect_pickup_location(self, frame):
        self.vision_system.load_model('yolo11s')
        self.motor_control.spin(dir="left", speed=100)
        return self.vision_system.process_detection_frame(frame)

    def _retrieval(self, frame):
        self.vision_system.load_model('yolo11s')
        self._navigate_to_target(frame)
        return frame

    def _procurement(self, frame):
        self.motor_control.stop()
        logger.info("Procurement: Awaiting package placement.")
        return annotate_frame(frame, "Procurement")

    def _detect_delivery_location(self, frame):
        self.vision_system.load_model('yolo11s-pose')
        self.motor_control.spin(dir="right", speed=100)
        return self.vision_system.process_detection_frame(frame)

    def _shipping(self, frame):
        self.vision_system.load_model('yolo11s-pose')
        self._navigate_to_target(frame)
        return frame

    def _delivery(self, frame):
        self.motor_control.stop()
        logger.info("Delivery: Awaiting package removal.")
        return annotate_frame(frame, "Delivery")

    def _navigate_to_target(self, frame):
        results = self.vision_system.current_model(frame, conf=0.6, verbose=False)
        target_found = False
        for box in results[0].boxes:
            if box.cls == self.vision_system.PERSON_ID or box.cls == self.vision_system.second_object_id:
                x_min, x_max = map(int, box.xyxy[0][0:2])
                target_center_x = (x_min + x_max) // 2
                frame_center_x = frame.shape[1] // 2
                error = target_center_x - frame_center_x
                target_found = True
                break

        if target_found:
            error_derivative = error - self.previous_error
            control_signal = Config.Kp * error + Config.Kd * error_derivative
            self.previous_error = error
            base_speed = 150
            left_pwm = base_speed + control_signal
            right_pwm = base_speed - control_signal
            left_pwm = max(min(int(left_pwm), 255), -255)
            right_pwm = max(min(int(right_pwm), 255), -255)
            self.motor_control.move(left_pwm, right_pwm)
        else:
            self.motor_control.stop()

        if get_lidar_distance() <= Config.STOP_DISTANCE_THRESHOLD:
            self.motor_control.stop()

# Annotation function for adding text to frames
def annotate_frame(frame, text):
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

# Simulated LIDAR distance reading
def get_lidar_distance():
    return random.randint(30, 100)

# Handle state callback for MQTT
def handle_state(payload):
    state_id = payload.get("state", None)
    if state_id is not None and state_id in State:
        state_machine.transition_to_state(State(state_id))

# Main setup and execution
mqtt_handler = MQTTHandler(Config.MQTT_BROKER, handle_state, lambda payload: logger.info("Sensor data received"))
motors = MotorControl(mqtt_handler)
vision = VisionSystem()
state_machine = RobotStateMachine(motors, vision)
previous_loop_time = time.time()

try:
    vision.load_model('yolo11s')
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        processed_frame = state_machine.execute_current_state(frame)
        current_time = time.time()
        fps = 1 / (current_time - previous_loop_time)
        previous_loop_time = current_time
        cv2.putText(processed_frame, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Robot Feed", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    logger.info("Process interrupted.")
finally:
    mqtt_handler.stop()
    vid.release()
    cv2.destroyAllWindows()
