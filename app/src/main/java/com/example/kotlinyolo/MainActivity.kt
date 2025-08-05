package com.example.kotlinyolo

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import ai.onnxruntime.*
import android.view.View
import java.io.ByteArrayOutputStream
import java.nio.FloatBuffer
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var boxOverlay: BoxOverlayView

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val REQUEST_CODE_CAMERA = 100
    private val MODEL_INPUT_SIZE = 640

    private lateinit var ortEnv: OrtEnvironment
    private lateinit var yolo: YoloV8Processor
    private lateinit var yoloSession: OrtSession

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        window.decorView.systemUiVisibility = (
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                        or View.SYSTEM_UI_FLAG_FULLSCREEN
                        or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                        or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                        or View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                        or View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                )

        previewView = findViewById(R.id.previewView)
        boxOverlay = findViewById(R.id.boxOverlay)

        ortEnv = OrtEnvironment.getEnvironment()

        val yoloBytes = assets.open("yolov8n.onnx").readBytes()
        yoloSession = ortEnv.createSession(yoloBytes)
        yolo = YoloV8Processor(this, ortEnv, yoloSession)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this,
                arrayOf(Manifest.permission.CAMERA), REQUEST_CODE_CAMERA)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_CAMERA &&
            grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalyzer.setAnalyzer(cameraExecutor) { imageProxy ->
                processImage(imageProxy)
            }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            } catch (e: Exception) {
                Log.e("CameraX", "Use case binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImage(imageProxy: ImageProxy) {
        try {
            val bitmap = yuvToRgbRotatedAndPadded(imageProxy) ?: run {
                imageProxy.close()
                return
            }

            val inputTensor = bitmapToOnnxTensor(ortEnv, bitmap)
            val detections = yolo.runInference(inputTensor)
            inputTensor.close()

            previewView.post {
                val paddingX = 80f
                val scaleX = 480f / (640f - 2 * paddingX)
                val scaleY = 480f / 640f

                val mappedBoxes = detections.map { det ->
                    val centerX = (det.x - paddingX) * scaleX
                    val centerY = det.y * scaleY
                    val width = det.w * scaleX
                    val height = det.h * scaleY

                    BoxOverlayView.RenderedBox(
                        x = centerX - width / 2f,
                        y = centerY - height / 2f,
                        w = width,
                        h = height,
                        conf = det.conf,
                        classId = det.classId
                    )
                }

                boxOverlay.boxes = mappedBoxes
                boxOverlay.invalidate()
            }
        } catch (e: Exception) {
            Log.e("Inference", "Error during inference", e)
        } finally {
            imageProxy.close()
        }
    }

    private fun yuvToRgbRotatedAndPadded(image: ImageProxy): Bitmap? {
        val nv21 = yuv420888ToNv21(image)  // Assuming this function exists and is correct

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)

        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
        val yuvBytes = out.toByteArray()

        var bitmap = BitmapFactory.decodeByteArray(yuvBytes, 0, yuvBytes.size) ?: return null

        val matrix = Matrix().apply { postRotate(90f) }
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

        val paddedBitmap = Bitmap.createBitmap(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(paddedBitmap)
        canvas.drawColor(Color.BLACK)

        val left = (MODEL_INPUT_SIZE - bitmap.width) / 2f
        val top = (MODEL_INPUT_SIZE - bitmap.height) / 2f
        canvas.drawBitmap(bitmap, left, top, null)

        return paddedBitmap
    }

    fun bitmapToOnnxTensor(env: OrtEnvironment, bitmap: Bitmap): OnnxTensor {
        val width = bitmap.width
        val height = bitmap.height

        val inputShape = longArrayOf(1, 3, height.toLong(), width.toLong())
        val floatArray = FloatArray(3 * height * width)
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = pixels[y * width + x]

                floatArray[0 * height * width + y * width + x] = ((pixel shr 16) and 0xFF) / 255f // R
                floatArray[1 * height * width + y * width + x] = ((pixel shr 8) and 0xFF) / 255f  // G
                floatArray[2 * height * width + y * width + x] = (pixel and 0xFF) / 255f           // B
            }
        }

        return OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArray), inputShape)
    }
}
