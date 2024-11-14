#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_SSD1306.h>
#include <math.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire);

const int photoPin = 36; // ADC0 on ESP32
#define MPU 0x68  // MPU6050 I2C address

unsigned long tapDisplayTime = 0;  // Track tap display duration

void setup() {
    Serial.begin(115200);
    Wire.begin();
    Wire.setClock(400000);  // Set I2C speed

    // Initialize OLED
    if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
        Serial.println(F("OLED initialization failed"));
        while (true);
    }
    display.clearDisplay();
    display.display();

    // Initialize MPU6050
    Wire.beginTransmission(MPU);
    Wire.write(0x6B);  // Power management register
    Wire.write(0x00);  // Wake up MPU6050
    Wire.endTransmission(true);
    delay(1000);
    Serial.println("MPU6050 initialized");
}

// Function to detect if a cup is present based on photoresistor reading
String cupDetected() {
    int sensorValue = analogRead(photoPin);
    float voltage = sensorValue * (3.3 / 4095.0);

    if (voltage < 0.5) {
        return "Cup Detected";
    } else {
        return "Place Cup";
    }
}

// Function to detect a tap based on acceleration magnitude
bool tapDetected() {
    int16_t ax, ay, az;

    // Read accelerometer data
    Wire.beginTransmission(MPU);
    Wire.write(0x3B);  // Accelerometer data register
    Wire.endTransmission(false);
    Wire.requestFrom(MPU, 6);

    ax = (Wire.read() << 8 | Wire.read());
    ay = (Wire.read() << 8 | Wire.read());
    az = (Wire.read() << 8 | Wire.read());

    // Calculate the magnitude of acceleration
    float accelMagnitude = sqrt(ax * ax + ay * ay + az * az) / 16384.0;

    // Output formatted data for the VS Code extension Serial Plotter
    Serial.print(">");
    Serial.print("accelMagnitude:");
    Serial.print(accelMagnitude);
    Serial.println("\r"); // Ensure each line ends with \r\n for plotter compatibility

    if (accelMagnitude > 2.2) {  // Adjust threshold as needed
        tapDisplayTime = millis();  // Reset tap display timer
        return true;
    }
    return false;
}

void loop() {
    // Check cup status
    String cupStatus = cupDetected();

    // Check for tap and update display time if detected
    bool isTapped = tapDetected();

    // Update OLED display
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);

    // Display cup status on the first line
    display.setCursor(0, 0);
    display.setTextSize(1);
    display.print(cupStatus);

    // Display "TAPPED" on the second line if a tap was recently detected
    if (isTapped || (millis() - tapDisplayTime < 3000)) { // Show "TAPPED" for 3 seconds
        display.setCursor(0, 16);
        display.setTextSize(2);
        display.print("TAPPED");
    }

    // Show updated content on OLED
    display.display();

    delay(1);  // Small delay for loop timing
}
