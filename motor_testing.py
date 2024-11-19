import paho.mqtt.client as mqtt

# MQTT Broker details
broker = "10.243.82.33"
port = 1883
topic = "bot/motors"

# Initialize the MQTT client
client = mqtt.Client("MotorController")
client.connect(broker, port, 60)

def send_motor_command(pwm1, pwm2):
    """Send a motor command as a JSON array [pwm1, pwm2]."""
    message = f"[{pwm1},{pwm2}]"
    client.publish(topic, message)
    print(f"Sent command: {message}")

print("Enter motor commands in the format 'w', 's', 'a', 'd' or 'q' to quit:")
print("  W: Forward")
print("  S: Reverse")
print("  A: Turn left")
print("  D: Turn right")
print("  Q: Quit")

try:
    while True:
        command = input("Enter command: ").strip().lower()

        if command == 'w':
            send_motor_command(200, 200)  # Forward
        elif command == 's':
            send_motor_command(-200, -200)  # Reverse
        elif command == 'a':
            send_motor_command(-200, 200)  # Turn left
        elif command == 'd':
            send_motor_command(200, -200)  # Turn right
        elif command == 'q':
            send_motor_command(0, 0)  # Stop motors
            print("Exiting...")
            break
        else:
            print("Invalid command. Use 'w', 's', 'a', 'd', or 'q'.")
finally:
    client.disconnect()
