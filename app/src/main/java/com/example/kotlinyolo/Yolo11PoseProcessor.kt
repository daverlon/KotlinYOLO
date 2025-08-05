package com.example.kotlinyolo

import ai.onnxruntime.*

class YoloPose11Processor(
    private val ortEnv: OrtEnvironment,
    private val session: OrtSession
) {
    data class Keypoint(val x: Float, val y: Float, val conf: Float)
    data class PoseDetection(val keypoints: List<Keypoint>, val score: Float, val bbox: FloatArray)

    companion object {
        private const val NUM_KEYPOINTS = 17
        private const val CONFIDENCE_THRESHOLD = 0.25f
        private const val NMS_IOU_THRESHOLD = 0.4f
    }

    fun runInference(inputTensor: OnnxTensor): List<PoseDetection> {
        val output = session.run(mapOf(session.inputNames.iterator().next() to inputTensor))
        val outputBuffer = (output[0].value as Array<Array<FloatArray>>)[0]  // [56][8400]

        val numDetections = outputBuffer[0].size
        val poses = mutableListOf<PoseDetection>()

        for (i in 0 until numDetections) {
            val score = outputBuffer[4][i]
            if (score < CONFIDENCE_THRESHOLD) continue

            val cx = outputBuffer[0][i]
            val cy = outputBuffer[1][i]
            val w = outputBuffer[2][i]
            val h = outputBuffer[3][i]
            val bbox = floatArrayOf(cx - w / 2f, cy - h / 2f, w, h)

            val keypoints = (0 until NUM_KEYPOINTS).map { kpIdx ->
                val base = 5 + kpIdx * 3
                Keypoint(
                    outputBuffer[base][i],
                    outputBuffer[base + 1][i],
                    outputBuffer[base + 2][i]
                )
            }

            poses.add(PoseDetection(keypoints, score, bbox))
        }

        output.close()


        return nonMaximumSuppression(poses, NMS_IOU_THRESHOLD).take(3)
//        return nonMaximumSuppression(poses, NMS_IOU_THRESHOLD)
    }

    private fun nonMaximumSuppression(
        poses: List<PoseDetection>,
        iouThreshold: Float
    ): List<PoseDetection> {
        val sorted = poses.sortedByDescending { it.score }.toMutableList()
        val result = mutableListOf<PoseDetection>()

        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            result.add(best)

            val iterator = sorted.iterator()
            while (iterator.hasNext()) {
                val other = iterator.next()
                if (iou(best.bbox, other.bbox) > iouThreshold) {
                    iterator.remove()
                }
            }
        }

        return result
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val (xA, yA, wA, hA) = a
        val (xB, yB, wB, hB) = b

        val left = maxOf(xA, xB)
        val top = maxOf(yA, yB)
        val right = minOf(xA + wA, xB + wB)
        val bottom = minOf(yA + hA, yB + hB)

        val intersection = maxOf(0f, right - left) * maxOf(0f, bottom - top)
        val union = wA * hA + wB * hB - intersection

        return if (union <= 0f) 0f else intersection / union
    }
}
