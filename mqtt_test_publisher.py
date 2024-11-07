import time
import json
import paho.mqtt.client as mqtt

# MQTT Configuration
MQTT_BROKER = "10.243.82.33"
STATUS_TOPIC = "bot/state"
SENSORS_TOPIC = "bot/sensors"
CONTROL_TOPIC = "bot/motors"

# Initialize MQTT Client
client = mqtt.Client()

# Callback function for receiving messages
def on_message(client, userdata, msg):
    # Parse the JSON payload
    payload = json.loads(msg.payload.decode())
    if msg.topic == CONTROL_TOPIC:
        print(f"Received motor command: {payload}")

# Connect callback
def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker with result code " + str(rc))
    client.subscribe(CONTROL_TOPIC)  # Subscribe to the motor topic

# Initialize client and set callbacks
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER)

# Start MQTT loop in a separate thread
client.loop_start()

# Function to publish mock data to STATUS_TOPIC and SENSORS_TOPIC
def publish_test_data():
    state_data = {"state": "1"}
    sensor_data = {
        "imu": {
            "gyro_x": 1.23,
            "gyro_y": -0.5,
            "gyro_z": 0.0,
            "accel_x": 0.12,
            "accel_y": 0.34,
            "accel_z": 9.81
        },
        "lidar": 150.0
    }
    
    while True:
        # Publish to state topic
        client.publish(STATUS_TOPIC, json.dumps(state_data))
        print(f"Published to {STATUS_TOPIC}: {state_data}")
        
        # Publish to sensors topic
        client.publish(SENSORS_TOPIC, json.dumps(sensor_data))
        print(f"Published to {SENSORS_TOPIC}: {sensor_data}")
        
        # Wait 1 second before next publish
        time.sleep(1)

try:
    # Start publishing test data
    publish_test_data()

except KeyboardInterrupt:
    print("Test program interrupted.")

finally:
    client.loop_stop()  # Stop the MQTT loop
