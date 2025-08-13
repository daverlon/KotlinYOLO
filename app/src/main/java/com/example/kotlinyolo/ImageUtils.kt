package com.example.kotlinyolo

import androidx.camera.core.ImageProxy

fun yuv420888ToNv21(image: ImageProxy): ByteArray {
    val width = image.width
    val height = image.height

    val yPlane = image.planes[0]
    val uPlane = image.planes[1]
    val vPlane = image.planes[2]

    val ySize = yPlane.buffer.remaining()
    val uSize = uPlane.buffer.remaining()
    val vSize = vPlane.buffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)

    yPlane.buffer.get(nv21, 0, ySize)

    val chromaHeight = height / 2
    val chromaWidth = width / 2
    val rowStrideU = uPlane.rowStride
    val pixelStrideU = uPlane.pixelStride
    val rowStrideV = vPlane.rowStride
    val pixelStrideV = vPlane.pixelStride

    var pos = ySize
    for (row in 0 until chromaHeight) {
        for (col in 0 until chromaWidth) {
            val vIndex = row * rowStrideV + col * pixelStrideV
            val uIndex = row * rowStrideU + col * pixelStrideU
            nv21[pos++] = vPlane.buffer.get(vIndex)
            nv21[pos++] = uPlane.buffer.get(uIndex)
        }
    }
    return nv21
}