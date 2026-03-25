package com.kikoeruca.debug

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import android.app.Activity

/**
 * Captures 16kHz mono audio in a background thread.
 * Delivers float PCM samples via callback.
 */
class AudioCapture(private val activity: Activity) {

    companion object {
        const val SAMPLE_RATE = 16000
        const val WINDOW_SAMPLES = 16000  // 1 second
        const val REQUEST_CODE = 1001
    }

    interface Listener {
        fun onAudioFrame(audio: FloatArray)
    }

    private var recorder: AudioRecord? = null
    private var thread: Thread? = null
    @Volatile private var running = false
    private var listener: Listener? = null

    fun setListener(l: Listener) { listener = l }

    fun hasPermission(): Boolean =
        ContextCompat.checkSelfPermission(activity, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED

    fun requestPermission() {
        ActivityCompat.requestPermissions(
            activity,
            arrayOf(Manifest.permission.RECORD_AUDIO),
            REQUEST_CODE,
        )
    }

    fun start(): Boolean {
        if (!hasPermission()) return false

        val bufSize = maxOf(
            AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_FLOAT,
            ),
            WINDOW_SAMPLES * 4 * 2,  // at least 2 windows
        )

        try {
            recorder = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_FLOAT,
                bufSize,
            )
        } catch (e: SecurityException) {
            return false
        }

        if (recorder?.state != AudioRecord.STATE_INITIALIZED) {
            recorder?.release()
            recorder = null
            return false
        }

        running = true
        recorder?.startRecording()

        thread = Thread({
            val buffer = FloatArray(WINDOW_SAMPLES)
            while (running) {
                val read = recorder?.read(buffer, 0, WINDOW_SAMPLES, AudioRecord.READ_BLOCKING) ?: 0
                if (read == WINDOW_SAMPLES) {
                    listener?.onAudioFrame(buffer.clone())
                }
            }
        }, "AudioCapture").apply {
            priority = Thread.MAX_PRIORITY
            start()
        }

        return true
    }

    fun stop() {
        running = false
        thread?.join(2000)
        thread = null
        recorder?.stop()
        recorder?.release()
        recorder = null
    }
}
