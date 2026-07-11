#include "DHT.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include "time.h"
#include <FirebaseESP32.h>

FirebaseAuth auth;
FirebaseConfig config;


// System Settings
#define WIFI_SSID     "Harshiv 4G"
#define WIFI_PASSWORD "Google900B"

#define FIREBASE_URL  "https://esp32-airquality-monitor-default-rtdb.asia-southeast1.firebasedatabase.app/"
#define DEVICE_ID     "AIR-SENSE-PRO"   


// SENSOR SETTINGS
 
#define DHTTYPE DHT11
#define DHTPIN  4
#define MQ135_PIN 34
#define Green_LED 25
#define RED_LED 26
#define BUZZER 27

#define RL_VALUE 10000.0
#define R0 7600.0
#define A -0.45
#define B 1.90


//   TIME SETTINGS
const char* ntpServer = "pool.ntp.org";
const long  gmtOffset_sec = 19800;   // IST
const int   daylightOffset_sec = 0;


DHT dht(DHTPIN, DHTTYPE);
unsigned long lastUpdate = 0;
const unsigned long uploadInterval = 15000; // 15 sec Interval

// NON-BLOCKING BLINK
unsigned long lastBlink = 0;
bool ledState = false;


//  GET TIMESTAMP
String getTimeStamp() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) return "TIME_ERROR";

  char buffer[40];
  strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", &timeinfo);
  return String(buffer);
}


// Data Upload  TO Firebase  
void sendToFirebase(float temp, float hum, float ppm) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi lost ❌ Skipping upload.");
    return;
  }

  HTTPClient http;
  String deviceLower = DEVICE_ID;
  deviceLower.toLowerCase();


  String timestamp = getTimeStamp();
  String path = String(FIREBASE_URL) + "devices/" + DEVICE_ID + "/readings/" + timestamp + ".json";


  String json = "{";
  json += "\"temperature\":" + String(temp, 1) + ",";
  json += "\"humidity\":" + String(hum, 1) + ",";
  json += "\"gas_ppm\":" + String(ppm, 1) + ",";
  json += "\"status\":\"" + String((ppm <= 50) ? "GOOD" : "BAD") + "\"";
  json += "}";

  http.begin(path);
  http.addHeader("Content-Type", "application/json");

  int code = http.PUT(json);

  Serial.println("\n📡 Uploading to Firebase...");
  Serial.println("URL: " + path);
  Serial.println("Payload: " + json);

  if (code > 0) {
    Serial.println("Firebase Response: " + http.getString());
  } else {
    Serial.print("Firebase Error ❌ ");
    Serial.println(http.errorToString(code));
  }

  http.end();
}


//  AIR QUALITY STATUS (NON-BLOCKING)
void handleAlerts(float ppm) {
  unsigned long currentMillis = millis();

  // NON BLOCKING BLINK FREQUENCY
  int interval = 600; // default slow blink

  if (ppm <= 50) {
    digitalWrite(Green_LED, HIGH);
    digitalWrite(RED_LED, LOW);
    digitalWrite(BUZZER, LOW);
    return;
  }
  else if (ppm <= 100) interval = 400;
  else if (ppm <= 150) interval = 300;
  else if (ppm <= 200) interval = 200;
  else if (ppm <= 300) interval = 150;
  else interval = 90;

  // NON-BLOCKING LED + BUZZER BLINK
  if (currentMillis - lastBlink >= interval) {
    lastBlink = currentMillis;
    ledState = !ledState;

    digitalWrite(RED_LED, ledState);
    digitalWrite(BUZZER, ledState);
    digitalWrite(Green_LED, LOW);
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(Green_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);
  pinMode(BUZZER, OUTPUT);

  dht.begin();

  // -------------------- WIFI --------------------
  Serial.println("Connecting to WiFi...");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected ✅");

  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
  Serial.println("⏰ NTP Time Synced!");

  // 🔐 Firebase configuration
  config.api_key = "AIzaSyDExWY6TkAItsbj7bG5KlGy4Ehvn21Qojw";
  config.database_url = "https://esp32-airquality-monitor-default-rtdb.asia-southeast1.firebasedatabase.app/";

  if (Firebase.signUp(&config, &auth, "", "")) {
    Serial.println("Auth success");
  } else {
    Serial.printf("Auth failed: %s\n", config.signer.signupError.message.c_str());
  }

  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);
}

void loop() {

  float hum = dht.readHumidity();
  float temp = dht.readTemperature();

  if (isnan(hum) || isnan(temp)) {
    Serial.println("DHT11 ERROR ❌");
    return;
  }

  int adcValue = analogRead(MQ135_PIN);
  float sensor_volt = adcValue * (3.3 / 4095.0);
  float Rs = ((3.3 - sensor_volt) * RL_VALUE) / (sensor_volt + 0.0001);
  float ratio = Rs / R0;
  float ppm = pow(10, (B + (A * log10(ratio + 0.0001))));

  handleAlerts(ppm);   //non blocking alerts 
 
  Serial.println("\n---------------------------------------");
  Serial.println("Time       : " + getTimeStamp());
  Serial.print("Temp       : "); Serial.print(temp); Serial.println(" °C");
  Serial.print("Humidity   : "); Serial.print(hum);  Serial.println(" %");
  Serial.print("Gas (PPM)  : "); Serial.println(ppm);
  Serial.println("---------------------------------------");

  // Firebasse Upload 
  if (millis() - lastUpdate >= uploadInterval) {
    lastUpdate = millis();
    sendToFirebase(temp, hum, ppm);
  }
}
