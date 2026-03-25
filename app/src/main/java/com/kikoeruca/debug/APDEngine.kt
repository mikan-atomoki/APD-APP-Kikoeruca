package com.kikoeruca.debug

import android.content.res.AssetManager

/**
 * JNI wrapper for the APD inference engine.
 *
 * Usage:
 *   val engine = APDEngine()
 *   engine.loadModel(assets, "model.apd")
 *   val result = engine.infer(audioFloatArray)
 *   engine.release()
 */
class APDEngine {
    private var handle: Long = 0

    companion object {
        init {
            System.loadLibrary("apd_engine")
        }
    }

    data class Result(
        val preSigmoid: Float,
        val score: Float,       // 0.0 - 1.0
        val inferenceMs: Float,
    )

    fun loadModel(assetManager: AssetManager, path: String): Boolean {
        handle = nativeLoadModel(assetManager, path)
        return handle != 0L
    }

    /**
     * Run inference on 16kHz mono float audio.
     * Audio should be 16000 samples (1 second).
     */
    fun infer(audio: FloatArray): Result {
        check(handle != 0L) { "Model not loaded" }
        val arr = nativeInfer(handle, audio) ?: return Result(0f, 0.5f, 0f)
        return Result(arr[0], arr[1], arr[2])
    }

    /** Compute RMS of audio buffer (cheap, no model needed). */
    fun computeRms(audio: FloatArray): Float = nativeComputeRms(audio)

    fun release() {
        if (handle != 0L) {
            nativeRelease(handle)
            handle = 0
        }
    }

    fun isLoaded(): Boolean = handle != 0L

    private external fun nativeLoadModel(assetManager: AssetManager, path: String): Long
    private external fun nativeInfer(handle: Long, audio: FloatArray): FloatArray?
    private external fun nativeComputeRms(audio: FloatArray): Float
    private external fun nativeRelease(handle: Long)
}
