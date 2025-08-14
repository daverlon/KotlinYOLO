package com.example.kotlinyolo

import android.content.Context
import ai.onnxruntime.*

object YoloPostprocess {

    init {
        System.loadLibrary("yolo_compute_offload")
    }

    external fun nms(
        boxes: FloatArray,
        numBoxes: Int,
        iouThreshold: Float = 0.5f,
        confThreshold: Float = 0.25f
    ): FloatArray
}

class YoloV8Processor(
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

        // Extract detections
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

        // Convert to flat array for C++ NMS: [x, y, w, h, conf, classId] * N
        val rawBoxes = FloatArray(detections.size * 6)
        for ((i, det) in detections.withIndex()) {
            rawBoxes[i * 6 + 0] = det.x
            rawBoxes[i * 6 + 1] = det.y
            rawBoxes[i * 6 + 2] = det.w
            rawBoxes[i * 6 + 3] = det.h
            rawBoxes[i * 6 + 4] = det.conf
            rawBoxes[i * 6 + 5] = det.classId.toFloat()
        }

        // Call C++ NMS
        val nmsResult = YoloPostprocess.nms(
            rawBoxes,
            detections.size,
            iouThreshold = 0.5f,
            confThreshold = confidenceThreshold
        )

        val finalBoxes = mutableListOf<Detection>()
        for (i in nmsResult.indices step 6) {
            finalBoxes.add(
                Detection(
                    x = nmsResult[i],
                    y = nmsResult[i + 1],
                    w = nmsResult[i + 2],
                    h = nmsResult[i + 3],
                    conf = nmsResult[i + 4],
                    classId = nmsResult[i + 5].toInt()
                )
            )
        }

        return finalBoxes
    }
}