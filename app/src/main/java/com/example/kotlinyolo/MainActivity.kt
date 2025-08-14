package com.example.kotlinyolo

import android.Manifest
import android.content.pm.PackageManager
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
import java.nio.FloatBuffer
import java.util.concurrent.Executors
import kotlin.math.abs

object YoloPreprocess {

    init {
        System.loadLibrary("yolo_compute_offload")
    }

    external fun yuvToRgbLetterboxFloat(
        yPlane: ByteArray,
        uPlane: ByteArray,
        vPlane: ByteArray,
        width: Int,
        height: Int,
        outTensor: FloatArray
    )
}

class MainActivity : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var boxOverlay: BoxOverlayView

//    private lateinit var nnPreview: ImageView

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val REQUEST_CODE_CAMERA = 100

    private lateinit var ortEnv: OrtEnvironment
    private lateinit var yolo: YoloV8Processor
    private lateinit var yoloSession: OrtSession

    private var screenW: Float = 0f
    private var screenH: Float = 0f
    private var camW: Float = 0f
    private var camH: Float = 0f

    override fun onPause() {
        super.onPause()
        finish()
    }

    override fun onStop() {
        super.onStop()
        finish()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

//        val screenSize = Point().apply { windowManager.defaultDisplay.getRealSize(this) }
//        Log.d("ScreenSize", screenSize.toString())

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
//        nnPreview = findViewById(R.id.nnPreview)

        ortEnv = OrtEnvironment.getEnvironment()

        val yoloBytes = assets.open("yolo11n.onnx").readBytes()
        yoloSession = ortEnv.createSession(yoloBytes)
        yolo = YoloV8Processor(yoloSession)

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

            if (screenW == 0f || screenH == 0f) {
                screenW = previewView.width.toFloat()
                screenH = previewView.height.toFloat()
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalyzer.setAnalyzer(cameraExecutor) { imageProxy ->
                if (camW == 0f || camH == 0f) {
                    camW = imageProxy.width.toFloat()
                    camH = imageProxy.height.toFloat()
                }
//                Log.d("CameraX", "Native camera resolution: ${nativeWidth}x${nativeHeight}")

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

    fun preprocessImage(image: ImageProxy): FloatArray {
        val yPlane = ByteArray(image.planes[0].buffer.remaining())
        val uPlane = ByteArray(image.planes[1].buffer.remaining())
        val vPlane = ByteArray(image.planes[2].buffer.remaining())

        image.planes[0].buffer.get(yPlane)
        image.planes[1].buffer.get(uPlane)
        image.planes[2].buffer.get(vPlane)

        val floatArray = FloatArray(3 * 640 * 640)

        YoloPreprocess.yuvToRgbLetterboxFloat(
            yPlane, uPlane, vPlane,
            image.width, image.height,
            floatArray
        )

        return floatArray
    }

    private fun processImage(imageProxy: ImageProxy) {
//        Log.d("ImageProxy", "Image Proxy Size: ${imageProxy.width}x${imageProxy.height}")
        try {
            val floatArray = preprocessImage(imageProxy)
            val inputTensor = OnnxTensor.createTensor(
                ortEnv,
                FloatBuffer.wrap(floatArray),
                longArrayOf(1, 3, 640, 640)
            )

            val detections = yolo.runInference(inputTensor)
            inputTensor.close()

            previewView.post {

                val paddingX = (abs(screenW - camW))
                val paddingY = (abs(screenH - camH))

                val scaleX = screenW / camW
                val scaleY = screenH / camH

                val mappedBoxes = detections.map { det ->
                    val centerX = (det.x - paddingX * scaleX) * scaleX
                    val centerY = (det.y - paddingY * scaleY) * scaleY
                    val width = det.w
                    val height = det.h

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
}