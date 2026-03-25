package com.kikoeruca.debug

import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.view.View
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.switchmaterial.SwitchMaterial

class MainActivity : AppCompatActivity(), AudioCapture.Listener {

    companion object {
        private const val TAG = "Kikoeruca"
    }

    private lateinit var engine: APDEngine
    private lateinit var capture: AudioCapture
    private lateinit var controller: InferenceController
    private val handler = Handler(Looper.getMainLooper())
    private lateinit var prefs: SharedPreferences

    // Screens
    private lateinit var startScreen: View
    private lateinit var measuringScreen: View
    private lateinit var menuScreen: View

    // Start screen
    private lateinit var btnMeasure: FrameLayout

    // Measuring screen
    private lateinit var ringGauge: RingGaugeView
    private lateinit var zoneText: TextView
    private lateinit var scoreText: TextView
    private lateinit var statusText: TextView
    private lateinit var btnStop: LinearLayout

    // Menu
    private lateinit var switchVibration: SwitchMaterial
    private lateinit var switchBackground: SwitchMaterial
    private lateinit var seekMeterIntensity: SeekBar

    // Zone definitions (updated names per UI design)
    private data class Zone(val name: String, val color: Int)
    private val zones = listOf(
        Zone("聞き取り不可", Color.parseColor("#E82020")),
        Zone("困難",        Color.parseColor("#FF8C00")),
        Zone("やや困難",    Color.parseColor("#F5D020")),
        Zone("問題なし",    Color.parseColor("#4CDD4C")),
    )
    private val thresholds = floatArrayOf(0.3f, 0.5f, 0.8f)

    private var isMeasuring = false
    private var lastVibrationTime = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        prefs = getSharedPreferences("kikoeruca", Context.MODE_PRIVATE)

        // Init screens
        startScreen = findViewById(R.id.startScreen)
        measuringScreen = findViewById(R.id.measuringScreen)
        menuScreen = findViewById(R.id.menuScreen)

        // Start screen
        btnMeasure = findViewById(R.id.btnMeasure)
        findViewById<ImageView>(R.id.hamburgerStart).setOnClickListener { showMenu() }

        // Measuring screen
        ringGauge = findViewById(R.id.ringGauge)
        zoneText = findViewById(R.id.zoneText)
        scoreText = findViewById(R.id.scoreText)
        statusText = findViewById(R.id.statusText)
        btnStop = findViewById(R.id.btnStop)
        findViewById<ImageView>(R.id.hamburgerMeasure).setOnClickListener { showMenu() }

        // Menu
        switchVibration = findViewById(R.id.switchVibration)
        switchBackground = findViewById(R.id.switchBackground)
        seekMeterIntensity = findViewById(R.id.seekMeterIntensity)
        findViewById<ImageView>(R.id.hamburgerMenu).setOnClickListener { hideMenu() }

        // Load saved settings
        switchVibration.isChecked = prefs.getBoolean("vibration", false)
        switchBackground.isChecked = prefs.getBoolean("background", false)
        seekMeterIntensity.progress = prefs.getInt("meter_intensity", 50)

        switchVibration.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean("vibration", checked).apply()
        }
        switchBackground.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean("background", checked).apply()
            if (checked && isMeasuring) {
                startBackgroundService()
            } else if (!checked) {
                stopBackgroundService()
            }
        }
        seekMeterIntensity.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                prefs.edit().putInt("meter_intensity", progress).apply()
                updateMeterIntensity(progress)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // Init engine
        engine = APDEngine()
        capture = AudioCapture(this)
        controller = InferenceController()

        engine.loadModel(assets, "model.apd")

        btnMeasure.setOnClickListener { startMeasurement() }
        btnStop.setOnClickListener { stopMeasurement() }

        updateMeterIntensity(seekMeterIntensity.progress)
    }

    // ── Screen navigation ──

    private fun showScreen(screen: View) {
        startScreen.visibility = View.GONE
        measuringScreen.visibility = View.GONE
        menuScreen.visibility = View.GONE
        screen.visibility = View.VISIBLE
    }

    private fun showMenu() {
        showScreen(menuScreen)
    }

    private fun hideMenu() {
        if (isMeasuring) {
            showScreen(measuringScreen)
        } else {
            showScreen(startScreen)
        }
    }

    // ── Measurement control ──

    private fun startMeasurement() {
        if (!capture.hasPermission()) {
            capture.requestPermission()
            return
        }
        if (!engine.isLoaded()) return

        capture.setListener(this)
        controller.reset()

        if (capture.start()) {
            isMeasuring = true
            showScreen(measuringScreen)
            statusText.text = "計測中..."
            zoneText.text = ""
            scoreText.text = "--"
            ringGauge.score = 1.0f
            ringGauge.gaugeColor = Color.parseColor("#4CDD4C")

            if (switchBackground.isChecked) {
                startBackgroundService()
            }
        }
    }

    private fun stopMeasurement() {
        capture.stop()
        isMeasuring = false
        showScreen(startScreen)
        stopBackgroundService()
    }

    // ── Audio callback (background thread) ──

    override fun onAudioFrame(audio: FloatArray) {
        val rms = engine.computeRms(audio)
        val decision = controller.decide(rms)

        var result: APDEngine.Result? = null

        when (decision.action) {
            InferenceController.Action.INFER_SPIKE,
            InferenceController.Action.INFER_SCHEDULED -> {
                result = engine.infer(audio)
                controller.onInferenceResult(result.score)
                Log.d(TAG, "inference: %.1fms | score=%.2f | rms=%.4f | action=%s | interval=%d".format(
                    result.inferenceMs, result.score, rms, decision.action.name, decision.interval))
            }
            InferenceController.Action.SKIP_NO_SPEECH -> {
                Log.v(TAG, "skip (no speech): rms=%.4f | action=%s".format(rms, decision.action.name))
            }
            else -> {
                Log.v(TAG, "skip: rms=%.4f | action=%s".format(rms, decision.action.name))
            }
        }

        val r = result
        val d = decision
        handler.post { updateUI(rms, d, r) }
    }

    // ── UI update (main thread) ──

    private fun updateUI(
        rms: Float,
        decision: InferenceController.Decision,
        result: APDEngine.Result?,
    ) {
        if (result != null) {
            val score = result.score
            val zone = getZone(score)

            scoreText.text = "%.2f".format(score)
            zoneText.text = zones[zone].name
            ringGauge.score = score
            ringGauge.gaugeColor = zones[zone].color
            statusText.text = "計測中..."

            // Vibration feedback for difficult zones
            if (switchVibration.isChecked && zone <= 1) {
                vibrateForZone(zone)
            }
        }

        // SKIP_NO_SPEECH: stationary noise detected, show "問題なし"
        if (decision.action == InferenceController.Action.SKIP_NO_SPEECH) {
            scoreText.text = "--"
            zoneText.text = zones[3].name
            ringGauge.score = 1.0f
            ringGauge.gaugeColor = zones[3].color
            statusText.text = "計測中..."
        }
        // SKIP_QUIET: keep showing the previous result as-is
    }

    // ── Vibration ──

    private fun vibrateForZone(zone: Int) {
        val now = System.currentTimeMillis()
        // Throttle: don't vibrate more than once per 2 seconds
        if (now - lastVibrationTime < 2000) return
        lastVibrationTime = now

        val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vm = getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
            vm.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }

        // zone 0 = 聞き取り不可 → stronger vibration
        // zone 1 = 困難 → lighter vibration
        val duration = if (zone == 0) 500L else 200L
        val amplitude = if (zone == 0) 255 else 128

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createOneShot(duration, amplitude))
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(duration)
        }
    }

    // ── Helpers ──

    private fun getZone(score: Float): Int {
        if (score >= thresholds[2]) return 3
        if (score >= thresholds[1]) return 2
        if (score >= thresholds[0]) return 1
        return 0
    }

    private fun updateMeterIntensity(progress: Int) {
        // Map slider 0-100 to quietThreshold.
        // Higher slider = more sensitive = lower threshold
        val minThreshold = 0.001f
        val maxThreshold = 0.02f
        controller.quietThreshold = maxThreshold - (progress / 100f) * (maxThreshold - minThreshold)
    }

    // ── Background service ──

    private fun startBackgroundService() {
        val intent = Intent(this, MeasurementService::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent)
        } else {
            startService(intent)
        }
    }

    private fun stopBackgroundService() {
        stopService(Intent(this, MeasurementService::class.java))
    }

    // ── Lifecycle ──

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        when {
            menuScreen.visibility == View.VISIBLE -> hideMenu()
            isMeasuring -> stopMeasurement()
            else -> @Suppress("DEPRECATION") super.onBackPressed()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == AudioCapture.REQUEST_CODE &&
            grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startMeasurement()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        capture.stop()
        engine.release()
        stopBackgroundService()
    }
}
