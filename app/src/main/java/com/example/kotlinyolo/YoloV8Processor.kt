package com.example.kotlinyolo

import android.content.Context
import ai.onnxruntime.*
import kotlin.math.max
import kotlin.math.min

class YoloV8Processor(context: Context, private val ortEnv: OrtEnvironment, private val session: OrtSession) {
//class YoloV8Processor(context: Context) {

    // Define the number of classes for your YOLOv8n model.
    // For a standard COCO-trained YOLOv8n, this is 80.
    // Adjust if you're using a custom-trained model.
    private val NUM_CLASSES = 80

    data class Detection(
        val x: Float,  // center X (normalized 0-1 relative to 640x640 input)
        val y: Float,  // center Y (normalized 0-1 relative to 640x640 input)
        val w: Float,  // width (normalized 0-1 relative to 640x640 input)
        val h: Float,  // height (normalized 0-1 relative to 640x640 input)
        val conf: Float,  // confidence score (max class probability)
        val classId: Int  // class index
    )

    /**
     * Run YOLOv8 inference on the input tensor and decode detections.
     *
     * @param inputTensor ONNX tensor with shape [1,3,640,640], normalized RGB image.
     * @return List of detections after confidence thresholding and NMS.
     */
    fun runInference(inputTensor: OnnxTensor): List<Detection> {
        val outputs = session.run(mapOf("images" to inputTensor))

        val rawOutput = outputs[0].value as Array<Array<FloatArray>>
        val featureData = rawOutput[0]

        val numPredictions = featureData[0].size // e.g., 8400

        val dets = mutableListOf<Detection>()

        for (i in 0 until numPredictions) {
            val x = featureData[0][i]
            val y = featureData[1][i]
            val w = featureData[2][i]
            val h = featureData[3][i]

            var maxConf = 0.0f
            var classId = -1

            for (j in 0 until NUM_CLASSES) {
                val classConf = featureData[4 + j][i]
                if (classConf > maxConf) {
                    maxConf = classConf
                    classId = j
                }
            }

            val confidenceThreshold = 0.25f // Common threshold
            if (maxConf > confidenceThreshold) {
//                Log.d("YoloDebug", "Detected: x=$x, y=$y, w=$w, h=$h, conf=$maxConf, class=$classId")
                dets.add(Detection(x, y, w, h, maxConf, classId))
            }
        }

        // Run Non-Max Suppression (NMS) to filter overlapping boxes
        val iouThreshold = 0.5f
        return nonMaxSuppression(dets, iouThreshold)
    }

    private fun nonMaxSuppression(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        val sorted = detections.sortedByDescending { it.conf }
        val selected = mutableListOf<Detection>()

        val active = BooleanArray(sorted.size) { true }

        for (i in sorted.indices) {
            if (!active[i]) continue

            val currentDetection = sorted[i]
            selected.add(currentDetection)

            for (j in i + 1 until sorted.size) {
                if (!active[j]) continue

                val otherDetection = sorted[j]
                if (iou(currentDetection, otherDetection) > iouThreshold) {
                    active[j] = false
                }
            }
        }
        return selected
    }

    private fun iou(a: Detection, b: Detection): Float {
        val ax1 = a.x - a.w / 2
        val ay1 = a.y - a.h / 2
        val ax2 = a.x + a.w / 2
        val ay2 = a.y + a.h / 2

        val bx1 = b.x - b.w / 2
        val by1 = b.y - b.h / 2
        val bx2 = b.x + b.w / 2
        val by2 = b.y + b.h / 2

        val interLeft = max(ax1, bx1)
        val interTop = max(ay1, by1)
        val interRight = min(ax2, bx2)
        val interBottom = min(ay2, by2)

        val interWidth = max(0f, interRight - interLeft)
        val interHeight = max(0f, interBottom - interTop)
        val interArea = interWidth * interHeight

        val aArea = a.w * a.h
        val bArea = b.w * b.h

        val unionArea = aArea + bArea - interArea

        return if (unionArea > 0) interArea / unionArea else 0f
    }
}