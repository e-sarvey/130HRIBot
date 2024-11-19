import json
import threading
import dearpygui.dearpygui as dpg
import paho.mqtt.client as mqtt

# MQTT Configurations
MQTT_BROKER = "10.243.82.33"
STATUS_TOPIC = "bot/state"
SENSORS_TOPIC = "bot/sensors"
CONTROL_TOPIC = "bot/motors"

# Global Variables
state_messages = []
sensor_messages = []

# MQTT Callbacks
def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")
    client.subscribe([(STATUS_TOPIC, 0), (SENSORS_TOPIC, 0), (CONTROL_TOPIC, 0)])

def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    if msg.topic == STATUS_TOPIC:
        state_messages.append(payload)
        dpg.set_value("state_log", "\n".join(state_messages[-10:]))  # Show last 10 messages
    elif msg.topic == SENSORS_TOPIC:
        sensor_messages.append(payload)
        dpg.set_value("sensor_log", "\n".join(sensor_messages[-10:]))
    elif msg.topic == CONTROL_TOPIC:
        print(f"Motor Command: {payload}")  # Print motor messages

# MQTT Setup
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER)

# Run MQTT client in a separate thread
def mqtt_loop():
    client.loop_forever()

mqtt_thread = threading.Thread(target=mqtt_loop)
mqtt_thread.daemon = True
mqtt_thread.start()

# Publish Simulated Events
def publish_event(event_name):
    client.publish(SENSORS_TOPIC, json.dumps({"event": event_name}))
    print(f"Simulated Event Published: {event_name}")

# DearPyGui Setup
dpg.create_context()

with dpg.window(label="State Transition Tester", width=600, height=400):
    dpg.add_text("Simulate Events")
    with dpg.group(horizontal=True):
        dpg.add_button(label="Robot Initialized", callback=lambda: publish_event("robot_initialized"))
        dpg.add_button(label="IMU Tap", callback=lambda: publish_event("imu_tap_detected"))
        dpg.add_button(label="Lidar Threshold Met", callback=lambda: publish_event("lidar_threshold_met"))
        dpg.add_button(label="Cup Placed", callback=lambda: publish_event("cup_placed"))
        dpg.add_button(label="Cup Removed", callback=lambda: publish_event("cup_removed"))

    dpg.add_separator()
    dpg.add_text("State Messages")
    dpg.add_input_text(tag="state_log", multiline=True, readonly=True, width=550, height=100)

    dpg.add_separator()
    dpg.add_text("Sensor Messages")
    dpg.add_input_text(tag="sensor_log", multiline=True, readonly=True, width=550, height=100)

# Finalizing and Running the Interface
dpg.create_viewport(title="State Transition Tester", width=600, height=400)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()

client.loop_stop()  # Stop MQTT loop when GUI is closed