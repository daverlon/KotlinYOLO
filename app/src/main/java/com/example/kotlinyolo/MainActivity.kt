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
    private lateinit var poseOverlay: PoseOverlayView

    private val cameraExecutor = Executors.newSingleThreadExecutor()

    private val REQUEST_CODE_CAMERA = 100
    private val MODEL_INPUT_SIZE = 640

//    private lateinit var debugImageView: ImageView

    // --- init models --- //

    private lateinit var ortEnv: OrtEnvironment

    private lateinit var yolo: YoloV8Processor
    private lateinit var yoloSession: OrtSession

    private lateinit var yoloPose: YoloPose11Processor
    private lateinit var yoloPoseSession: OrtSession

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
        poseOverlay = findViewById(R.id.poseOverlay)

        // Initialize ONNX Runtime environment and session
        ortEnv = OrtEnvironment.getEnvironment()

        val yoloBytes = assets.open("yolov8n.onnx").readBytes()
        yoloSession = ortEnv.createSession(yoloBytes)
        yolo = YoloV8Processor(this, ortEnv, yoloSession)

        val yoloPoseBytes = assets.open("yolo11n-pose.onnx").readBytes()
        yoloPoseSession = ortEnv.createSession(yoloPoseBytes)
        yoloPose = YoloPose11Processor(ortEnv, yoloPoseSession)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                REQUEST_CODE_CAMERA
            )
        }

//        debugImageView = findViewById(R.id.debugImageView)

    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_CAMERA && grantResults.isNotEmpty()
            && grantResults[0] == PackageManager.PERMISSION_GRANTED
        ) {
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
//                .setTargetRotation(Surface.ROTATION_90)
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
            // 1) Convert ImageProxy YUV to Bitmap
            val bitmap = yuvToRgbRotatedAndPadded(imageProxy)
            if (bitmap == null) {
                imageProxy.close()
                return
            }

//            runOnUiThread {
//                debugImageView.setImageBitmap(bitmap)
//            }

            val inputTensor = bitmapToOnnxTensor(ortEnv, bitmap)

            val detections = yolo.runInference(inputTensor)

            val poses: List<YoloPose11Processor.PoseDetection> =  yoloPose.runInference(inputTensor)


            inputTensor.close()

            previewView.post {

                val paddingX = 80f
                val scaleX = 480f / (640f - 2 * paddingX)  // horizontal scale
                val scaleY = 480f / 640f                    // vertical scale

                val mappedBoxes = detections.map { detection ->
                    val centerX = (detection.x - paddingX) * scaleX
                    val centerY = detection.y * scaleY
                    val width = detection.w * scaleX
                    val height = detection.h * scaleY

                    BoxOverlayView.RenderedBox(
                        x = centerX - width / 2f,   // top-left x
                        y = centerY - height / 2f,  // top-left y
                        w = width,
                        h = height,
                        conf = detection.conf,
                        classId = detection.classId
                    )
                }
                boxOverlay.boxes = mappedBoxes
                boxOverlay.invalidate()

// Filter and map poses with proper scaling
//                val paddingX = 80f
//                val scaleX = 480f / (640f - 2 * paddingX)
//                val scaleY = 480f / 640f

                val confidenceThreshold = 0.5f
                val keypointConfidenceThreshold = 0.6f

                poseOverlay.poses = poses
                    .filter { it.score >= confidenceThreshold }
                    .map { pose ->
                        val (x0, y0, w, h) = pose.bbox

                        val centerX = (x0 + w / 2f - paddingX) * scaleX
                        val centerY = (y0 + h / 2f) * scaleY
                        val mappedW = w * scaleX
                        val mappedH = h * scaleY

                        YoloPose11Processor.PoseDetection(
                            keypoints = pose.keypoints
                                .filter { it.conf >= keypointConfidenceThreshold }
                                .map { kp ->
                                    YoloPose11Processor.Keypoint(
                                        x = (kp.x - paddingX) * scaleX,
                                        y = kp.y * scaleY,
                                        conf = kp.conf
                                    )
                                },
                            score = pose.score,
                            bbox = floatArrayOf(
                                centerX - mappedW / 2f,
                                centerY - mappedH / 2f,
                                mappedW,
                                mappedH
                            )
                        )
                    }

                poseOverlay.invalidate()
            }

        } catch (e: Exception) {
            Log.e("Inference", "Error during inference", e)
        } finally {
            imageProxy.close()
        }
    }


    private fun yuvToRgbRotatedAndPadded(image: ImageProxy): Bitmap? {
        val nv21 = yuv420888ToNv21(image)  // Use the correct conversion function

        val yuvImage = YuvImage(
            nv21,
            ImageFormat.NV21,
            image.width,
            image.height,
            null
        )

        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
        val yuvBytes = out.toByteArray()

        var bitmap = BitmapFactory.decodeByteArray(yuvBytes, 0, yuvBytes.size)
        if (bitmap == null) return null

        // Rotate 90 degrees anti-clockwise
        val matrix = Matrix()
        matrix.postRotate(90f)
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

        // Pad to 640x640 evenly
        val paddedBitmap = Bitmap.createBitmap(
            MODEL_INPUT_SIZE,
            MODEL_INPUT_SIZE,
            Bitmap.Config.ARGB_8888
        )
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

                val r = (pixel shr 16 and 0xFF).toFloat() / 255f
                val g = (pixel shr 8 and 0xFF).toFloat() / 255f
                val b = (pixel and 0xFF).toFloat() / 255f

                floatArray[0 * height * width + y * width + x] = r // Red channel
                floatArray[1 * height * width + y * width + x] = g // Green channel
                floatArray[2 * height * width + y * width + x] = b // Blue channel
            }
        }

        return OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArray), inputShape)
    }
}