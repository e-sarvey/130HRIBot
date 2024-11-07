import cv2
import json
import time
from ultralytics import YOLO
import paho.mqtt.client as mqtt

# MQTT Configurations
MQTT_BROKER = "10.0.0.25"
STATUS_TOPIC = "bot/state"
SENSORS_TOPIC = "bot/sensors"
CONTROL_TOPIC = "bot/motors"

# Initialize MQTT Client
client = mqtt.Client()

# Global variables for PD control
Kp = 0.1
Kd = 0.05
SAFE_ZONE_RADIUS = 5
STOP_DISTANCE_THRESHOLD = 50
current_state = 0
previous_error = 0
vid = cv2.VideoCapture(0)

# Detection model and motor control initialization
current_model = None
current_model_name = ""
object_classes = {'person': 0, 'bottle': 39}
PERSON_ID = object_classes['person']
second_object_id = object_classes['bottle']

class MotorControl:
    def __init__(self, mqtt_client, control_topic):
        self.client = mqtt_client
        self.control_topic = control_topic

    def _publish_command(self, pwm_values):
        payload = json.dumps({"pwm": pwm_values})
        self.client.publish(self.control_topic, payload)

    def spin(self, dir="left", speed=100):
        left_pwm = -speed if dir == "left" else speed
        right_pwm = speed if dir == "left" else -speed
        self._publish_command([left_pwm, right_pwm])

    def stop(self):
        self._publish_command([0, 0])

    def move(self, left_pwm, right_pwm):
        self._publish_command([left_pwm, right_pwm])

motors = MotorControl(client, CONTROL_TOPIC)

def load_model(model_name):
    global current_model, current_model_name
    if current_model_name == model_name:
        return
    model_path = f"yolo_models/{model_name}.pt"
    current_model = YOLO(model_path, verbose=False)
    current_model_name = model_name

def process_detection_frame(frame):
    target_width, target_height = 640, int(640 * frame.shape[0] / frame.shape[1])
    resized_frame = cv2.resize(frame, (target_width, target_height))
    resized_frame = cv2.flip(resized_frame, 1)
    results = current_model(resized_frame, conf=0.6, verbose=False)
    filtered_boxes = [box for box in results[0].boxes if box.cls in [PERSON_ID, second_object_id]]
    results[0].boxes = filtered_boxes
    return results[0].plot()

def annotate_fps_and_state(frame, start_time, state):
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"State: {state}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

def state_boot_up(frame):
    print("Boot-up: Initializing system and checking connections.")
    return annotate_fps_and_state(frame, time.time(), "Boot-up")

def state_await_activation(frame):
    print("Awaiting activation: Monitoring IMU for device tap.")
    return annotate_fps_and_state(frame, time.time(), "Awaiting Activation")

def state_detect_pickup_location(frame):
    load_model('yolo11s')
    motors.spin(dir="left", speed=100)
    return process_detection_frame(frame)

def state_retrieval(frame):
    load_model('yolo11s')
    navigate_to_target(frame)
    return frame

def state_procurement(frame):
    motors.stop()
    print("Procurement: Awaiting package placement.")
    return annotate_fps_and_state(frame, time.time(), "Procurement")

def state_detect_delivery_location(frame):
    load_model('yolo11s-pose')
    motors.spin(dir="right", speed=100)
    return process_detection_frame(frame)

def state_shipping(frame):
    load_model('yolo11s-pose')
    navigate_to_target(frame)
    return frame

def state_delivery(frame):
    motors.stop()
    print("Delivery: Awaiting package removal.")
    return annotate_fps_and_state(frame, time.time(), "Delivery")

state_functions = {
    0: state_boot_up,
    1: state_await_activation,
    2: state_detect_pickup_location,
    3: state_retrieval,
    4: state_procurement,
    5: state_detect_delivery_location,
    6: state_shipping,
    7: state_delivery,
}

def handle_state(payload):
    global current_state
    state = payload.get("state", None)
    if state is not None and state in state_functions:
        current_state = state

def navigate_to_target(frame):
    global previous_error
    results = current_model(frame, conf=0.6, verbose=False)
    target_found = False
    for box in results[0].boxes:
        if box.cls == PERSON_ID or box.cls == second_object_id:
            x_min, x_max = map(int, box.xyxy[0][0:2])
            target_center_x = (x_min + x_max) // 2
            frame_center_x = frame.shape[1] // 2
            error = target_center_x - frame_center_x
            target_found = True
            break

    if target_found:
        if abs(error) <= SAFE_ZONE_RADIUS:
            left_pwm = right_pwm = 150
        else:
            error_derivative = error - previous_error
            control_signal = Kp * error + Kd * error_derivative
            previous_error = error
            base_speed = 150
            left_pwm = base_speed + control_signal
            right_pwm = base_speed - control_signal

        left_pwm = max(min(int(left_pwm), 255), -255)
        right_pwm = max(min(int(right_pwm), 255), -255)
        motors.move(left_pwm, right_pwm)
    else:
        motors.stop()

    if get_lidar_distance() <= STOP_DISTANCE_THRESHOLD:
        motors.stop()

def get_lidar_distance():
    return random.randint(30, 100)

def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker")
    client.subscribe([(STATUS_TOPIC, 0), (SENSORS_TOPIC, 0)])

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    if msg.topic == STATUS_TOPIC:
        handle_state(payload)
    elif msg.topic == SENSORS_TOPIC:
        print("Sensor data received")

client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER)
client.loop_start()

try:
    load_model('yolo11s')
    while True:
        start_time = time.time()
        ret, frame = vid.read()
        if not ret:
            break

        # Process the frame based on the current state
        if current_state in state_functions:
            processed_frame = state_functions[current_state](frame)
        else:
            processed_frame = annotate_fps_and_state(frame, start_time, current_state)

        # Display the frame
        cv2.imshow("Robot Feed", processed_frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Process interrupted.")
finally:
    client.loop_stop()
    vid.release()
    cv2.destroyAllWindows()
