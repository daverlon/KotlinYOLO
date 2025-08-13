package com.example.kotlinyolo

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

class BoxOverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    var boxes: List<RenderedBox> = emptyList()

    private val boxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }

    private val textPaint = Paint().apply {
        color = Color.BLACK
        textSize = 12f
        isAntiAlias = true
    }

    private val bgPaint = Paint().apply {
        style = Paint.Style.FILL
        alpha = 200  // Slight transparency
    }

    var labels: List<String> = listOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    )

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        for (box in boxes) {
            val left = box.x
            val top = box.y
            val right = left + box.w
            val bottom = top + box.h

            // Get color for this class
            val boxColor = getColorForClass(box.classId)

            // Set paint colors accordingly
            boxPaint.color = boxColor
            bgPaint.color = boxColor

            canvas.drawRect(left, top, right, bottom, boxPaint)

            val label = labels.getOrNull(box.classId) ?: "Unknown"
            val labelText = "$label ${"%.1f".format(box.conf * 100)}%"

            val textBounds = Rect()
            textPaint.getTextBounds(labelText, 0, labelText.length, textBounds)
            val textPadding = 2f

            val bgLeft = left
            val bgTop = top - textBounds.height() - 2 * textPadding
            val bgRight = left + textBounds.width() + 2 * textPadding
            val bgBottom = top

            // Draw label background rectangle (semi-transparent)
            canvas.drawRect(bgLeft, bgTop, bgRight, bgBottom, bgPaint)

            // Draw label text (black)
            val textX = bgLeft + textPadding
            val textY = bgBottom - textPadding
            canvas.drawText(labelText, textX, textY, textPaint)
        }
    }

    private fun getColorForClass(classId: Int): Int {
        // Generate pseudo-random pastel color based on classId
        val hsv = floatArrayOf((classId * 37f) % 360, 0.6f, 0.9f)
        return Color.HSVToColor(hsv)
    }

    data class RenderedBox(
        val x: Float,
        val y: Float,
        val w: Float,
        val h: Float,
        val conf: Float,
        val classId: Int
    )
}