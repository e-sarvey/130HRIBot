import json
import time
import random
import threading
import dearpygui.dearpygui as dpg
import paho.mqtt.client as mqtt

# MQTT Configurations
MQTT_BROKER = "10.0.0.25"  # Example IP
STATUS_TOPIC = "bot/state"
SENSORS_TOPIC = "bot/sensors"
CONTROL_TOPIC = "bot/motors"

# Initialize MQTT Client
client = mqtt.Client()

# Global variables to hold incoming data for display
current_state = 0
imu_data = {"x": 0, "y": 0, "z": 0}
lidar_data = 0
motor_pwm = [0, 0]

# MQTT Callbacks
def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker with result code " + str(rc))
    client.subscribe([(CONTROL_TOPIC, 0)])

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    if msg.topic == CONTROL_TOPIC:
        handle_motor_data(payload)

def handle_motor_data(payload):
    global motor_pwm
    motor_pwm = payload.get("pwm", [0, 0])
    update_motor_gui()

# Publish IMU and LIDAR data to SENSORS_TOPIC
def publish_sensor_data():
    payload = json.dumps({"imu": imu_data, "lidar": lidar_data})
    client.publish(SENSORS_TOPIC, payload)
    print(f"Published sensor data: {payload}")

# Publish a new state to STATUS_TOPIC
def publish_state(state):
    global current_state
    current_state = state
    payload = json.dumps({"state": state})
    client.publish(STATUS_TOPIC, payload)
    print(f"Published new state: {state}")

# Update motor PWM display in GUI
def update_motor_gui():
    dpg.set_value("left_pwm_display", motor_pwm[0])
    dpg.set_value("right_pwm_display", motor_pwm[1])

# Update IMU and LIDAR data in response to user input and publish it
def update_imu_data(tag, value):
    imu_data[tag] = value
    publish_sensor_data()

def update_lidar_data(sender, value):
    global lidar_data
    lidar_data = round(value, 5)  # Round to 5 decimal places
    publish_sensor_data()

# Set up the MQTT client
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER)

# Run MQTT client in a separate thread
def mqtt_loop():
    client.loop_forever()

mqtt_thread = threading.Thread(target=mqtt_loop)
mqtt_thread.daemon = True
mqtt_thread.start()

# DearPyGui interface setup
dpg.create_context()

# Main Window
with dpg.window(label="Robot Control Interface", width=900, height=300):

    # State Control
    dpg.add_text("State Control")
    with dpg.group(horizontal=True):
        dpg.add_button(label="Boot Up", callback=lambda: publish_state(0))
        dpg.add_button(label="Await Activation", callback=lambda: publish_state(1))
        dpg.add_button(label="Detect Pickup Location", callback=lambda: publish_state(2))
        dpg.add_button(label="Retrieval", callback=lambda: publish_state(3))
        dpg.add_button(label="Procurement", callback=lambda: publish_state(4))
        dpg.add_button(label="Detect Delivery Location", callback=lambda: publish_state(5))
        dpg.add_button(label="Shipping", callback=lambda: publish_state(6))
        dpg.add_button(label="Delivery", callback=lambda: publish_state(7))

    # Sensor Data Entry
    dpg.add_text("Sensor Data Input")
    dpg.add_text("IMU Data (x, y, z):")
    with dpg.group(horizontal=True):
        dpg.add_input_float(label="X", tag="imu_x", width=75, callback=lambda s, d: update_imu_data("x", d))
        dpg.add_input_float(label="Y", tag="imu_y", width=75, callback=lambda s, d: update_imu_data("y", d))
        dpg.add_input_float(label="Z", tag="imu_z", width=75, callback=lambda s, d: update_imu_data("z", d))
    dpg.add_text("LIDAR Distance:")
    dpg.add_input_float(label="Distance", tag="lidar_value", callback=update_lidar_data)

    # Motor PWM Display
    dpg.add_text("Motor PWM Output")
    with dpg.group(horizontal=True):
        dpg.add_input_int(label="Left PWM", tag="left_pwm_display", readonly=True)
        dpg.add_input_int(label="Right PWM", tag="right_pwm_display", readonly=True)

# Finalizing and running the interface
dpg.create_viewport(title='Robot Control Interface', width=900, height=300)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()

client.loop_stop()  # Stop MQTT loop when GUI is closed
