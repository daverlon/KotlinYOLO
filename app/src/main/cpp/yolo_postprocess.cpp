#include <jni.h>
#include <vector>
#include <algorithm>

struct Detection {
    float x;
    float y;
    float w;
    float h;
    float conf;
    int classId;
};

// Compute IOU between two boxes
float iou(const Detection &a, const Detection &b) {
    float ax1 = a.x - a.w / 2;
    float ay1 = a.y - a.h / 2;
    float ax2 = a.x + a.w / 2;
    float ay2 = a.y + a.h / 2;

    float bx1 = b.x - b.w / 2;
    float by1 = b.y - b.h / 2;
    float bx2 = b.x + b.w / 2;
    float by2 = b.y + b.h / 2;

    float interLeft = std::max(ax1, bx1);
    float interTop = std::max(ay1, by1);
    float interRight = std::min(ax2, bx2);
    float interBottom = std::min(ay2, by2);

    float interW = std::max(0.0f, interRight - interLeft);
    float interH = std::max(0.0f, interBottom - interTop);
    float interArea = interW * interH;

    float unionArea = a.w * a.h + b.w * b.h - interArea;
    return unionArea > 0 ? interArea / unionArea : 0.0f;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_kotlinyolo_YoloPostprocess_nms(
        JNIEnv *env,
        jobject thiz,
        jfloatArray boxes,  // [x, y, w, h, conf, classId] * N
        jint numBoxes,
        jfloat iouThreshold,
        jfloat confThreshold
) {
    jfloat* data = env->GetFloatArrayElements(boxes, nullptr);

    std::vector<Detection> dets;
    for (int i = 0; i < numBoxes; ++i) {
        float conf = data[i * 6 + 4];
        if (conf < confThreshold) continue;

        Detection d;
        d.x = data[i * 6 + 0];
        d.y = data[i * 6 + 1];
        d.w = data[i * 6 + 2];
        d.h = data[i * 6 + 3];
        d.conf = conf;
        d.classId = (int)data[i * 6 + 5];
        dets.push_back(d);
    }

    std::sort(dets.begin(), dets.end(), [](const Detection &a, const Detection &b) {
        return a.conf > b.conf;
    });

    std::vector<Detection> finalDets;
    std::vector<bool> active(dets.size(), true);

    for (size_t i = 0; i < dets.size(); ++i) {
        if (!active[i]) continue;
        finalDets.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (!active[j]) continue;
            if (dets[i].classId == dets[j].classId && iou(dets[i], dets[j]) > iouThreshold) {
                active[j] = false;
            }
        }
    }

    // Return flat array [x, y, w, h, conf, classId] for each box
    jfloatArray outArray = env->NewFloatArray(finalDets.size() * 6);
    std::vector<jfloat> flat(finalDets.size() * 6);
    for (size_t i = 0; i < finalDets.size(); ++i) {
        flat[i * 6 + 0] = finalDets[i].x;
        flat[i * 6 + 1] = finalDets[i].y;
        flat[i * 6 + 2] = finalDets[i].w;
        flat[i * 6 + 3] = finalDets[i].h;
        flat[i * 6 + 4] = finalDets[i].conf;
        flat[i * 6 + 5] = static_cast<float>(finalDets[i].classId);
    }

    env->SetFloatArrayRegion(outArray, 0, flat.size(), flat.data());
    env->ReleaseFloatArrayElements(boxes, data, 0);
    return outArray;
}
