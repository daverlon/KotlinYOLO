package com.example.kotlinyolo

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

class PoseOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    var poses: List<YoloPose11Processor.PoseDetection> = emptyList()
        set(value) {
            field = value
            invalidate()
        }

    private val keypointPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val linePaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f
        color = Color.GREEN
        isAntiAlias = true
    }

    private val keypointRadius = 4f

    // Standard COCO keypoint connections (17 keypoints)
    private val skeletonConnections = listOf(
        Pair(0, 1),   // Nose to Left Eye
        Pair(0, 2),    // Nose to Right Eye
        Pair(1, 3),    // Left Eye to Left Ear
        Pair(2, 4),    // Right Eye to Right Ear
        Pair(5, 6),    // Left Shoulder to Right Shoulder
        Pair(5, 7),    // Left Shoulder to Left Elbow
        Pair(7, 9),    // Left Elbow to Left Wrist
        Pair(6, 8),    // Right Shoulder to Right Elbow
        Pair(8, 10),   // Right Elbow to Right Wrist
        Pair(5, 11),   // Left Shoulder to Left Hip
        Pair(6, 12),   // Right Shoulder to Right Hip
        Pair(11, 12),  // Left Hip to Right Hip
        Pair(11, 13),  // Left Hip to Left Knee
        Pair(13, 15),  // Left Knee to Left Ankle
        Pair(12, 14),  // Right Hip to Right Knee
        Pair(14, 16)   // Right Knee to Right Ankle
    )

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        for (pose in poses) {
            // Draw skeleton connections first
            for ((startIdx, endIdx) in skeletonConnections) {
                if (startIdx < pose.keypoints.size && endIdx < pose.keypoints.size) {
                    val start = pose.keypoints[startIdx]
                    val end = pose.keypoints[endIdx]

                    // Only draw if both keypoints are confident enough
                    if (start.conf > 0.3f && end.conf > 0.3f) {
                        canvas.drawLine(start.x, start.y, end.x, end.y, linePaint)
                    }
                }
            }

            // Draw keypoints on top of connections
            for (kp in pose.keypoints) {
                if (kp.conf > 0.1f) {
                    keypointPaint.color = confidenceToColor(kp.conf)
                    canvas.drawCircle(kp.x, kp.y, keypointRadius, keypointPaint)
                }
            }
        }
    }

    private fun confidenceToColor(conf: Float): Int {
        val hue = (120 * conf).coerceIn(0f, 120f) // Green (120°) to Red (0°)
        return Color.HSVToColor(floatArrayOf(hue, 1f, 1f))
    }
}