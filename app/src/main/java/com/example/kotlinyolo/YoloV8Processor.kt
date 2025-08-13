package com.example.kotlinyolo

import android.content.Context
import ai.onnxruntime.*
import kotlin.math.max
import kotlin.math.min

class YoloV8Processor(
    context: Context,
    private val ortEnv: OrtEnvironment,
    private val session: OrtSession
) {

    private val NUM_CLASSES = 80

    data class Detection(
        val x: Float,  // center X normalized
        val y: Float,  // center Y normalized
        val w: Float,  // width normalized
        val h: Float,  // height normalized
        val conf: Float,  // confidence score
        val classId: Int  // class index
    )

    fun runInference(inputTensor: OnnxTensor): List<Detection> {
        val outputs = session.run(mapOf("images" to inputTensor))
        val rawOutput = outputs[0].value as Array<Array<FloatArray>>
        val featureData = rawOutput[0]
        val numPredictions = featureData[0].size

        val detections = mutableListOf<Detection>()
        val confidenceThreshold = 0.25f

        for (i in 0 until numPredictions) {
            val x = featureData[0][i]
            val y = featureData[1][i]
            val w = featureData[2][i]
            val h = featureData[3][i]

            var maxConf = 0f
            var classId = -1
            for (j in 0 until NUM_CLASSES) {
                val conf = featureData[4 + j][i]
                if (conf > maxConf) {
                    maxConf = conf
                    classId = j
                }
            }

            if (maxConf > confidenceThreshold) {
                detections.add(Detection(x, y, w, h, maxConf, classId))
            }
        }

        return nonMaxSuppression(detections, 0.5f)
    }

    private fun nonMaxSuppression(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        val sorted = detections.sortedByDescending { it.conf }
        val selected = mutableListOf<Detection>()
        val active = BooleanArray(sorted.size) { true }

        for (i in sorted.indices) {
            if (!active[i]) continue
            val current = sorted[i]
            selected.add(current)

            for (j in i + 1 until sorted.size) {
                if (!active[j]) continue
                if (iou(current, sorted[j]) > iouThreshold) {
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

        val unionArea = a.w * a.h + b.w * b.h - interArea
        return if (unionArea > 0) interArea / unionArea else 0f
    }
}