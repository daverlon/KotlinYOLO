#include <jni.h>
#include <cstdint>
#include <algorithm>

constexpr int MODEL_INPUT_SIZE = 640;

extern "C"
JNIEXPORT void JNICALL
Java_com_example_kotlinyolo_YoloPreprocess_yuvToRgbLetterboxFloat(
        JNIEnv *env,
jobject thiz,
        jbyteArray yPlane,
jbyteArray uPlane,
        jbyteArray vPlane,
jint width,
        jint height,
jfloatArray outTensor) {

jbyte* y = env->GetByteArrayElements(yPlane, nullptr);
jbyte* u = env->GetByteArrayElements(uPlane, nullptr);
jbyte* v = env->GetByteArrayElements(vPlane, nullptr);
jfloat* out = env->GetFloatArrayElements(outTensor, nullptr);

int hw = width * height;

// Compute scaling for letterbox
float scale = static_cast<float>(MODEL_INPUT_SIZE) / std::max(width, height);
int newW = static_cast<int>(width * scale);
int newH = static_cast<int>(height * scale);
int padX = (MODEL_INPUT_SIZE - newW) / 2;
int padY = (MODEL_INPUT_SIZE - newH) / 2;

// Iterate over output 640x640 image
for (int y_out = 0; y_out < MODEL_INPUT_SIZE; ++y_out) {
for (int x_out = 0; x_out < MODEL_INPUT_SIZE; ++x_out) {

// Compute corresponding source pixel (inverse mapping)
int srcX = static_cast<int>((x_out - padX) / scale);
int srcY = static_cast<int>((y_out - padY) / scale);

// Apply 180° rotation: (x, y) -> (width-1-x, height-1-y)
srcX = width - 1 - srcX;
srcY = height - 1 - srcY;

float R = 0.0f, G = 0.0f, B = 0.0f;

if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
int yIndex = srcY * width + srcX;
int uvIndex = (srcY / 2) * (width / 2) + (srcX / 2);

float Y = static_cast<unsigned char>(y[yIndex]);
float U = static_cast<unsigned char>(u[uvIndex]) - 128.0f;
float V = static_cast<unsigned char>(v[uvIndex]) - 128.0f;

// Convert to RGB
R = Y + 1.402f * V;
G = Y - 0.344136f * U - 0.714136f * V;
B = Y + 1.772f * U;

// Normalize 0-1
R = std::clamp(R / 255.0f, 0.0f, 1.0f);
G = std::clamp(G / 255.0f, 0.0f, 1.0f);
B = std::clamp(B / 255.0f, 0.0f, 1.0f);
}

int outIndex = y_out * MODEL_INPUT_SIZE + x_out;
out[0 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE + outIndex] = R;
out[1 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE + outIndex] = G;
out[2 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE + outIndex] = B;
}
}

env->ReleaseByteArrayElements(yPlane, y, 0);
env->ReleaseByteArrayElements(uPlane, u, 0);
env->ReleaseByteArrayElements(vPlane, v, 0);
env->ReleaseFloatArrayElements(outTensor, out, 0);
}