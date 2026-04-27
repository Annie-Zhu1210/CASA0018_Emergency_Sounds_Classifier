#include <PDM.h>
#include <EmergencyVehicleSirens_inferencing.h>

#define MOTOR_PIN           9
#define CONFIDENCE_THRESHOLD 0.70

// Vibration patterns
#define MODERATE_DURATION   400
#define MODERATE_REPEATS    2
#define INTENSE_DURATION    1800
#define INTENSE_REPEATS     1

/* Audio buffer */
typedef struct {
  int16_t *buffer;
  uint8_t buf_ready;
  uint32_t buf_count;
  uint32_t n_samples;
} inference_t;

static inference_t inference;
static signed short sampleBuffer[2048];
static bool debug_nn = false;

void vibrate_moderate() {
  for (int i = 0; i < MODERATE_REPEATS; i++) {
    digitalWrite(MOTOR_PIN, HIGH);
    delay(MODERATE_DURATION);
    digitalWrite(MOTOR_PIN, LOW);
    delay(200);
  }
}

void vibrate_intense() {
  for (int i = 0; i < INTENSE_REPEATS; i++) {
    digitalWrite(MOTOR_PIN, HIGH);
    delay(INTENSE_DURATION);
    digitalWrite(MOTOR_PIN, LOW);
    delay(200);
  }
}

static void pdm_data_ready_inference_callback(void) {
  int bytesAvailable = PDM.available();
  int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);

  if (inference.buf_ready == 0) {
    for (int i = 0; i < bytesRead >> 1; i++) {
      inference.buffer[inference.buf_count++] = sampleBuffer[i];
      if (inference.buf_count >= inference.n_samples) {
        inference.buf_count = 0;
        inference.buf_ready = 1;
        break;
      }
    }
  }
}

static bool microphone_inference_start(uint32_t n_samples) {
  inference.buffer = (int16_t *)malloc(n_samples * sizeof(int16_t));
  if (inference.buffer == NULL) return false;

  inference.buf_count = 0;
  inference.n_samples = n_samples;
  inference.buf_ready = 0;

  PDM.onReceive(pdm_data_ready_inference_callback);
  PDM.setBufferSize(4096);

  if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
    ei_printf("ERR: Failed to start PDM!\r\n");
    microphone_inference_end();
    return false;
  }

  PDM.setGain(127);
  return true;
}

static bool microphone_inference_record(void) {
  inference.buf_ready = 0;
  inference.buf_count = 0;

  uint32_t start = millis();
  while (inference.buf_ready == 0) {
    if (millis() - start > 5000) {
      ei_printf("ERR: Audio capture timed out\r\n");
      return false;
    }
    delay(10);
  }
  return true;
}

static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
  numpy::int16_to_float(&inference.buffer[offset], out_ptr, length);
  return 0;
}

static void microphone_inference_end(void) {
  PDM.end();
  free(inference.buffer);
}

void setup() {
  Serial.begin(115200);
  pinMode(MOTOR_PIN, OUTPUT);
  digitalWrite(MOTOR_PIN, LOW);

  // Quick motor test on boot
  Serial.println("Motor test...");
  vibrate_moderate();
  delay(500);

  Serial.println("Edge Impulse Inference - Emergency Sound Classifier");
  Serial.println("Classes: background | carhorn | siren_hilo | siren_wail&yelp");

  if (!microphone_inference_start(EI_CLASSIFIER_RAW_SAMPLE_COUNT)) {
    ei_printf("ERR: Failed to setup audio sampling\r\n");
    while (1);
  }

  Serial.println("Listening...");
}

void loop() {
  if (!microphone_inference_record()) {
    ei_printf("ERR: Failed to record audio...\r\n");
    return;
  }

  signal_t signal;
  signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
  signal.get_data = &microphone_audio_signal_get_data;

  ei_impulse_result_t result = { 0 };
  EI_IMPULSE_ERROR r = run_classifier(&signal, &result, debug_nn);

  if (r != EI_IMPULSE_OK) {
    ei_printf("ERR: Failed to run classifier (%d)\r\n", r);
    return;
  }

  // Print all predictions
  ei_printf("Predictions (DSP: %d ms., Classification: %d ms.):\r\n",
            result.timing.dsp, result.timing.classification);
  for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
    ei_printf("    %s: %.5f\r\n",
              result.classification[ix].label,
              result.classification[ix].value);
  }

  // Find highest confidence class
  float max_val = 0.0;
  const char* max_label = "";

  for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
    if (result.classification[ix].value > max_val) {
      max_val = result.classification[ix].value;
      max_label = result.classification[ix].label;
    }
  }

  // Check for siren (either subtype) and merge in code 
  bool is_siren = (
    (strcmp(max_label, "siren_hilo") == 0 ||
     strcmp(max_label, "siren_wail&yelp") == 0) &&
    max_val >= CONFIDENCE_THRESHOLD
  );

  bool is_carhorn = (
    strcmp(max_label, "carhorn") == 0 &&
    max_val >= CONFIDENCE_THRESHOLD
  );

  if (is_siren) {
    ei_printf(">> EMERGENCY SIREN detected (%.2f) - intensive vibration\r\n", max_val);
    vibrate_intense();
  } else if (is_carhorn) {
    ei_printf(">> CAR HORN detected (%.2f) - moderate vibration\r\n", max_val);
    vibrate_moderate();
  } else {
    ei_printf(">> Background / below threshold (%.2f)\r\n", max_val);
  }
}