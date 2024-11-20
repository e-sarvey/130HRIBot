#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_SSD1306.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <Adafruit_VL53L1X.h>
#include <ArduinoJson.h>
#include <math.h>

// Wi-Fi and MQTT Settings
// const char* ssid = "53CurtisAve2";
// const char* password = "Apt2@53CurtisAve";
// const char* mqttServer = "10.0.0.25"; //"10.243.82.33";
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
void tapDetected();
void cupDetected();
void handleLidar();
void publishEvent(const char* event);

// ============================ MQTT Callback ============================
// ============================ MQTT Callback ============================
void mqttCallback(char* topic, byte* payload, unsigned int length) {
    char message[length + 1];
    strncpy(message, (char*)payload, length);
    message[length] = '\0';
        // Suppress printing for bot/motors topic
    if (String(topic) == motorTopic) {
        return;
    }
    // Debug: Log received message
    // Serial.print("MQTT Message received on topic: ");
    // Serial.println(topic);
    // Serial.print("Payload: ");
    // Serial.println(message);

    // Check if the topic is for state changes
    if (String(topic) == stateTopic) {
        // Parse the JSON message
        StaticJsonDocument<128> doc; // Adjust size if needed
        DeserializationError error = deserializeJson(doc, message);

        if (error) {
            Serial.print("JSON parsing failed: ");
            Serial.println(error.c_str());
            return;
        }

        // Extract the "state" field
        const char* stateCommand = doc["state"];
        if (stateCommand == nullptr) {
            Serial.println("No 'state' field found in JSON message.");
            return;
        }

        // Debug: Log the extracted state command
        Serial.print("Recieved state command: ");
        Serial.println(stateCommand);

        // Handle state change commands
        if (String(stateCommand).startsWith("start_")) {
            String newState = String(stateCommand).substring(6);

            // Validate and update the state
            if (newState == "await_activation") currentState = AWAIT_ACTIVATION;
            else if (newState == "detect_pickup_location") currentState = DETECT_PICKUP_LOCATION;
            else if (newState == "retrieval") currentState = RETRIEVAL;
            else if (newState == "procurement") currentState = PROCUREMENT;
            else if (newState == "detect_delivery_location") currentState = DETECT_DELIVERY_LOCATION;
            else if (newState == "shipping") currentState = SHIPPING;
            else if (newState == "delivery") currentState = DELIVERY;
            else {
                Serial.println("Invalid state received. Ignoring...");
                return;
            }

            // Debug: Confirm state update
            Serial.print("State successfully updated to: ");
            Serial.println(newState);

            // Update OLED display
            updateOLED("CURRENT STATE", newState);

            // Publish confirmation message
            String confirmationMessage = "{\"state\": \"" + newState + "_started\"}";
            mqttClient.publish(stateTopic, confirmationMessage.c_str());

            // Debug: Log confirmation message
            Serial.print("Confirmation message published: ");
            Serial.println(confirmationMessage);
        } else {
            //Serial.println("State command does not start with 'start_'. Ignoring...");
        }
    }
}


// ============================ Wi-Fi Connection ============================
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
void connectToMQTT() {
    while (!mqttClient.connected()) {
        Serial.print("Connecting to MQTT...");
        if (mqttClient.connect("ESP32Client")) {
            Serial.println(" Connected.");
            mqttClient.subscribe(stateTopic);
            mqttClient.subscribe(motorTopic);
        } else {
            Serial.println(" Failed. Retrying...");
            delay(5000);
        }
    }
}

// ============================ OLED Update ============================
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
void tapDetected() {
    int16_t ax, ay, az;
    Wire.beginTransmission(MPU);
    Wire.write(0x3B);  // Register address for accelerometer data
    Wire.endTransmission(false);
    Wire.requestFrom(MPU, 6);
    ax = (Wire.read() << 8 | Wire.read());
    ay = (Wire.read() << 8 | Wire.read());
    az = (Wire.read() << 8 | Wire.read());

    float accelMagnitude = sqrt(ax * ax + ay * ay + az * az) / 16384.0;

    // Log the IMU readings for debugging
    // Serial.print("Accel Magnitude: "); // Uncomment for debugging
    // Serial.println(accelMagnitude);

    // Check if the magnitude exceeds the tap threshold
    if (accelMagnitude > 2.2) {  // Tap threshold
        mqttClient.publish(sensorsTopic, "{\"event\": \"imu_tap_detected\"}");
        Serial.println("Event published: imu_tap_detected");
    }
}


// ============================ Cup Detection ============================
void cupDetected() {
    int sensorValue = analogRead(photoPin);
    float voltage = sensorValue * (3.3 / 4095.0);
    // Serial.print("Photoresistor Voltage: ");
    // Serial.println(voltage);

    // Check the state to determine action
    if (currentState == PROCUREMENT && voltage < 1.8) {
        // Cup placed in PROCUREMENT state
        mqttClient.publish(sensorsTopic, "{\"event\": \"cup_placed\"}");
        Serial.println("Event published: cup_placed");
    } else if (currentState == DELIVERY && voltage > 1.8) {
        // Cup removed in DELIVERY state
        mqttClient.publish(sensorsTopic, "{\"event\": \"cup_removed\"}");
        Serial.println("Event published: cup_removed");
    }
}

// ============================ Lidar Handling ============================
void handleLidar() {
    // Only use Lidar in RETRIEVAL and SHIPPING states
    if (currentState == RETRIEVAL || currentState == SHIPPING) {
        if (vl53.dataReady()) {
            int16_t distance = vl53.distance();
            // Serial.print("Lidar Distance: ");
            // Serial.println(distance);

            if (distance != -1 && distance < lidarThreshold) {
                mqttClient.publish(sensorsTopic, "{\"event\": \"lidar_threshold_met\"}");
                Serial.println("Event published: lidar_threshold_met");
            }
            vl53.clearInterrupt();
        }
    } else {
        // Avoid unnecessary Lidar readings
        //Serial.println("Lidar ignored in current state.");
    }
}


// ============================ Motor Control ============================
void setMotors(int pwm1, int pwm2) {
    if (pwm1 > 0) {
        ledcWrite(pwmChannel1Fwd, pwm1);
        ledcWrite(pwmChannel1Rev, 0);
    } else {
        ledcWrite(pwmChannel1Fwd, 0);
        ledcWrite(pwmChannel1Rev, -pwm1);
    }

    if (pwm2 > 0) {
        ledcWrite(pwmChannel2Fwd, pwm2);
        ledcWrite(pwmChannel2Rev, 0);
    } else {
        ledcWrite(pwmChannel2Fwd, 0);
        ledcWrite(pwmChannel2Rev, -pwm2);
    }
}

void stopMotors() {
    setMotors(0, 0);
    Serial.println("Motors stopped.");
}

// ============================ Setup ============================
void setup() {
    Serial.begin(115200);

    // Initialize Wi-Fi and MQTT
    connectToWiFi();
    mqttClient.setServer(mqttServer, mqttPort);
    mqttClient.setCallback(mqttCallback);
    connectToMQTT();

    // Initialize OLED
    if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
        Serial.println(F("OLED initialization failed"));
        while (true);
    }
    // This process clears and displays
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.setTextSize(1);
    display.print("Welcome!");
    display.display();
    Serial.println("OLED Initialized");
    delay(2);
    updateOLED("CURRENT STATE:", "BOOT-UP");

    // Initialize MPU6050
    Wire.begin();
    Wire.setClock(400000);
    Wire.beginTransmission(MPU);
    Wire.write(0x6B);  // Power management register
    Wire.write(0x00);  // Wake up MPU6050
    Wire.endTransmission(true);
    Serial.println("MPU6050 initialized.");

    // Initialize Lidar
    if (!vl53.begin(0x29, &Wire)) {
        Serial.println(F("Lidar initialization failed."));
        while (true);
    }
    vl53.setTimingBudget(50);
    vl53.startRanging();
    Serial.println("Lidar initialized.");

    // Initialize Motor PWM
    ledcSetup(pwmChannel1Fwd, pwmFreq, pwmResolution);
    ledcAttachPin(motor1Pin1, pwmChannel1Fwd);
    ledcSetup(pwmChannel1Rev, pwmFreq, pwmResolution);
    ledcAttachPin(motor1Pin2, pwmChannel1Rev);
    ledcSetup(pwmChannel2Fwd, pwmFreq, pwmResolution);
    ledcAttachPin(motor2Pin1, pwmChannel2Fwd);
    ledcSetup(pwmChannel2Rev, pwmFreq, pwmResolution);
    ledcAttachPin(motor2Pin2, pwmChannel2Rev);

    mqttClient.publish(sensorsTopic, "{\"event\": \"robot_initialized\"}");
    delay(1);
    Serial.println("Setup complete. Event published: robot_initialized");
}

// ============================ Loop ============================
void loop() {
    if (!mqttClient.connected()) {
        connectToMQTT();
    }
    mqttClient.loop();

    switch (currentState) {
        case AWAIT_ACTIVATION:
            tapDetected();
            break;

        case DETECT_PICKUP_LOCATION:
            break;

        case RETRIEVAL:
            handleLidar();
            break;

        case PROCUREMENT:
            cupDetected();
            break;

        case SHIPPING:
            handleLidar();
            break;

        case DELIVERY:
            cupDetected();
            break;

        default:
            break;
    }
}

