#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_SSD1306.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include "Adafruit_VL53L1X.h"
#include <math.h>

// Wi-Fi and MQTT Settings
const char* ssid = "Tufts_Robot";
const char* password = "";
const char* mqttServer = "10.243.82.33";
const int mqttPort = 1883;
const char* stateTopic = "bot/state";      // Topic to receive state change commands
const char* sensorsTopic = "bot/sensors"; // Topic to publish sensor-based events
const char* motorTopic = "bot/motors";    // Topic to receive motor control commands

// OLED Display Settings
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire);

// IMU (MPU6050) Settings
#define MPU 0x68  // MPU6050 I2C address
unsigned long tapDisplayTime = 0;  // Track how long "TAPPED" message is displayed

// Lidar Settings
#define IRQ_PIN 17
#define XSHUT_PIN 16
Adafruit_VL53L1X vl53 = Adafruit_VL53L1X(XSHUT_PIN, IRQ_PIN);
const int lidarThreshold = 300;  // Distance in mm to trigger an event

// Motor Settings
const int motor1Pin1 = 0; // Motor 1 forward
const int motor1Pin2 = 4; // Motor 1 reverse
const int motor2Pin1 = 2; // Motor 2 forward
const int motor2Pin2 = 15; // Motor 2 reverse
const int pwmFreq = 1000;  // PWM frequency
const int pwmResolution = 8; // PWM resolution (8-bit: 0-255)
const int pwmChannel1Fwd = 0;
const int pwmChannel1Rev = 1;
const int pwmChannel2Fwd = 2;
const int pwmChannel2Rev = 3;

// Photoresistor (Cup Detection) Settings
const int photoPin = 36; // ADC0 pin for photoresistor

// Failsafe Timeout (Stops motors if no commands received for 45 seconds)
const unsigned long FAILSAFE_TIMEOUT = 45000;
unsigned long lastMotorCommandTime = 0;

// MQTT Client
WiFiClient espClient;
PubSubClient mqttClient(espClient);

// State Machine Enum
enum State {
    BOOT_UP,                  // Robot is initializing
    AWAIT_ACTIVATION,         // Waiting for an IMU tap event
    DETECT_PICKUP_LOCATION,   // Searching for pickup target
    RETRIEVAL,                // Navigating to pickup target
    PROCUREMENT,              // Waiting for cup placement
    DETECT_DELIVERY_LOCATION, // Searching for delivery target
    SHIPPING,                 // Navigating to delivery target
    DELIVERY                  // Waiting for cup removal
};
State currentState = BOOT_UP; // Initial state is BOOT_UP

// Function Prototypes
void connectToWiFi();
void connectToMQTT();
void mqttCallback(char* topic, byte* payload, unsigned int length);
void setMotors(int pwm1, int pwm2);
void stopMotors();
void updateOLED(const String& line1, const String& line2);
bool tapDetected();
String cupDetected();
void handleLidar();
void publishEvent(const char* event);

// ============================ MQTT Callback ============================
// This function processes incoming MQTT messages
void mqttCallback(char* topic, byte* payload, unsigned int length) {
    JsonDocument doc = StaticJsonDocument<128>(); // Explicitly specify `StaticJsonDocument`
    DeserializationError error = deserializeJson(doc, payload, length);

    if (error) {
        Serial.println("Failed to parse MQTT message.");
        return;
    }

    const char* stateCommand = doc["state"];
    if (stateCommand) {
        if (String(stateCommand).startsWith("start_")) {
            String newState = String(stateCommand).substring(6);
            if (newState == "await_activation") currentState = AWAIT_ACTIVATION;
            else if (newState == "detect_pickup_location") currentState = DETECT_PICKUP_LOCATION;
            else if (newState == "retrieval") currentState = RETRIEVAL;
            else if (newState == "procurement") currentState = PROCUREMENT;
            else if (newState == "detect_delivery_location") currentState = DETECT_DELIVERY_LOCATION;
            else if (newState == "shipping") currentState = SHIPPING;
            else if (newState == "delivery") currentState = DELIVERY;

            updateOLED("State Changed:", newState);
            publishEvent((newState + "_started").c_str());
        }
    }
}

// ============================ Wi-Fi Connection ============================
// Connects to the specified Wi-Fi network
void connectToWiFi() {
    Serial.print("Connecting to Wi-Fi...");
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println(" Connected.");
}

// ============================ MQTT Connection ============================
// Connects to the MQTT broker and subscribes to necessary topics
void connectToMQTT() {
    while (!mqttClient.connected()) {
        Serial.print("Connecting to MQTT...");
        if (mqttClient.connect("ESP32Client")) {
            Serial.println(" Connected.");
            mqttClient.subscribe(stateTopic);
        } else {
            Serial.println(" Failed. Retrying...");
            delay(5000);
        }
    }
}

// ============================ OLED Update ============================
// Updates the OLED display with two lines of text
void updateOLED(const String& line1, const String& line2) {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.print(line1);
    display.setCursor(0, 16);
    display.print(line2);
    display.display();
}

// ============================ IMU Tap Detection ============================
// Detects a tap event using the MPU6050 accelerometer
bool tapDetected() {
    int16_t ax, ay, az;
    Wire.beginTransmission(MPU);
    Wire.write(0x3B);  // Register address for accelerometer data
    Wire.endTransmission(false);
    Wire.requestFrom(MPU, 6);
    ax = (Wire.read() << 8 | Wire.read());
    ay = (Wire.read() << 8 | Wire.read());
    az = (Wire.read() << 8 | Wire.read());

    float accelMagnitude = sqrt(ax * ax + ay * ay + az * az) / 16384.0;

    if (accelMagnitude > 2.2) {  // Tap threshold
        tapDisplayTime = millis();
        return true;
    }
    return false;
}

// ============================ Cup Detection ============================
// Detects if a cup is placed based on the photoresistor reading
String cupDetected() {
    int sensorValue = analogRead(photoPin);
    float voltage = sensorValue * (3.3 / 4095.0);
    return voltage < 0.5 ? "Cup Detected" : "Place Cup";
}

// ============================ Lidar Handling ============================
// Monitors distance using the Lidar and publishes an event if below threshold
void handleLidar() {
    if (vl53.dataReady()) {
        int16_t distance = vl53.distance();
        if (distance != -1 && distance < lidarThreshold) {
            publishEvent("lidar_threshold_met");
        }
        vl53.clearInterrupt();
    }
}

// ============================ Publish Event ============================
// Publishes an event to the MQTT sensors topic
void publishEvent(const char* event) {
    JsonDocument doc = StaticJsonDocument<128>(); // Explicitly specify `StaticJsonDocument`
    doc["event"] = event;
    char buffer[128];
    size_t len = serializeJson(doc, buffer);
    mqttClient.publish(sensorsTopic, buffer, len);
}

// ============================ Motor Control ============================
// Sets motor PWM values
void setMotors(int pwm1, int pwm2) {
    // Motor 1
    if (pwm1 > 0) {
        ledcWrite(pwmChannel1Fwd, pwm1);
        ledcWrite(pwmChannel1Rev, 0);
    } else {
        ledcWrite(pwmChannel1Fwd, 0);
        ledcWrite(pwmChannel1Rev, -pwm1);
    }

    // Motor 2
    if (pwm2 > 0) {
        ledcWrite(pwmChannel2Fwd, pwm2);
        ledcWrite(pwmChannel2Rev, 0);
    } else {
        ledcWrite(pwmChannel2Fwd, 0);
        ledcWrite(pwmChannel2Rev, -pwm2);
    }
}

// Stops both motors
void stopMotors() {
    setMotors(0, 0);
}

// ============================ Setup ============================
// Initializes all components and connections
void setup() {
    Serial.begin(115200);

    // Initialize Wi-Fi and MQTT
    connectToWiFi();
    mqttClient.setServer(mqttServer, mqttPort);
    mqttClient.setCallback(mqttCallback);

    // Initialize OLED
    if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
        Serial.println(F("OLED initialization failed"));
        while (true);
    }
    display.clearDisplay();
    display.display();

    // Initialize MPU6050
    Wire.begin();
    Wire.setClock(400000);
    Wire.beginTransmission(MPU);
    Wire.write(0x6B);  // Power management register
    Wire.write(0x00);  // Wake up MPU6050
    Wire.endTransmission(true);

    // Initialize Lidar
    if (!vl53.begin(0x29, &Wire)) {
        Serial.println(F("Lidar initialization failed."));
        while (true);
    }
    vl53.setTimingBudget(50);
    vl53.startRanging();

    // Initialize Motor PWM
    ledcSetup(pwmChannel1Fwd, pwmFreq, pwmResolution);
    ledcAttachPin(motor1Pin1, pwmChannel1Fwd);
    ledcSetup(pwmChannel1Rev, pwmFreq, pwmResolution);
    ledcAttachPin(motor1Pin2, pwmChannel1Rev);
    ledcSetup(pwmChannel2Fwd, pwmFreq, pwmResolution);
    ledcAttachPin(motor2Pin1, pwmChannel2Fwd);
    ledcSetup(pwmChannel2Rev, pwmFreq, pwmResolution);
    ledcAttachPin(motor2Pin2, pwmChannel2Rev);

    // Publish initialization complete
    publishEvent("robot_initialized");
    Serial.println("Setup complete.");
}

// ============================ Loop ============================
// Main program logic
void loop() {
    if (!mqttClient.connected()) {
        connectToMQTT();
    }
    mqttClient.loop();

    // State-specific behavior
    switch (currentState) {
        case AWAIT_ACTIVATION:
            if (tapDetected()) {
                publishEvent("imu_tap_detected");
            }
            break;

        case DETECT_PICKUP_LOCATION:
        case DETECT_DELIVERY_LOCATION:
            handleLidar();
            break;

        case PROCUREMENT:
        case DELIVERY:
            if (cupDetected() == "Cup Detected") {
                publishEvent(currentState == PROCUREMENT ? "cup_placed" : "cup_removed");
            }
            break;

        default:
            break;
    }

    // Failsafe: Stop motors if no commands for 45 seconds
    if (millis() - lastMotorCommandTime > FAILSAFE_TIMEOUT) {
        stopMotors();
        lastMotorCommandTime = millis();
    }
}