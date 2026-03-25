package com.kikoeruca.debug

/**
 * Controls when to invoke the AI model based on RMS levels and speech detection.
 *
 * Strategy:
 *   - Always compute RMS (near-zero cost via native)
 *   - If RMS < quietThreshold → environment is quiet, no inference needed
 *   - Simple VAD via RMS variance: speech causes fluctuating RMS, stationary
 *     noise (AC, fans) has stable RMS. Skip inference for stationary noise.
 *   - If RMS spikes (rapid change) → immediate inference
 *   - If RMS is above threshold and stable → inference at adaptive interval
 *   - Interval grows from 1s → 5s → 10s as score stabilizes
 */
class InferenceController {

    // RMS thresholds (configurable)
    var quietThreshold: Float = 0.005f      // below this = silence, skip inference
    var spikeRatio: Float = 3.0f            // RMS jumps by this factor = spike

    // VAD: RMS variance-based speech detection
    // Speech has high RMS variance (syllables, pauses); stationary noise is flat.
    // Track coefficient of variation (CV = std/mean) over a sliding window.
    private val rmsHistory = FloatArray(VAD_WINDOW_SIZE)
    private var rmsHistoryIdx = 0
    private var rmsHistoryCount = 0

    // Adaptive interval (frames, where 1 frame = 1 second)
    private var framesUntilNextInference = 0
    private var currentInterval = 1          // start at every frame
    private var stableCount = 0
    private var lastRms = 0f
    private var lastScore = -1f

    enum class Action {
        SKIP_QUIET,       // too quiet, don't infer
        SKIP_NO_SPEECH,   // above threshold but no speech detected (stationary noise)
        INFER_SPIKE,      // RMS spiked, infer immediately
        INFER_SCHEDULED,  // regular scheduled inference
        WAIT,             // waiting for next scheduled inference
    }

    data class Decision(
        val action: Action,
        val rms: Float,
        val interval: Int,
    )

    /**
     * Call once per audio frame (1 second). Returns whether to run inference.
     */
    fun decide(rms: Float): Decision {
        // Update RMS history for VAD
        rmsHistory[rmsHistoryIdx] = rms
        rmsHistoryIdx = (rmsHistoryIdx + 1) % VAD_WINDOW_SIZE
        if (rmsHistoryCount < VAD_WINDOW_SIZE) rmsHistoryCount++

        val decision: Action

        if (rms < quietThreshold) {
            // Quiet environment — reset interval, no inference
            currentInterval = 1
            stableCount = 0
            framesUntilNextInference = 0
            decision = Action.SKIP_QUIET
        } else if (lastRms > 0 && rms / lastRms > spikeRatio) {
            // RMS spike detected — infer immediately, reset interval
            currentInterval = 1
            stableCount = 0
            framesUntilNextInference = 0
            decision = Action.INFER_SPIKE
        } else if (!isSpeechLikely()) {
            // Above quiet threshold but RMS is too stable → stationary noise, no speech
            // Don't reset interval/stableCount — preserve state for when speech returns
            decision = Action.SKIP_NO_SPEECH
        } else if (framesUntilNextInference <= 0) {
            // Scheduled inference
            decision = Action.INFER_SCHEDULED
            framesUntilNextInference = currentInterval
        } else {
            framesUntilNextInference--
            decision = Action.WAIT
        }

        lastRms = rms
        return Decision(decision, rms, currentInterval)
    }

    /**
     * Simple VAD: check if RMS has enough variation to indicate speech.
     * Speech typically has CV (coefficient of variation) > 0.15 due to
     * syllable/pause patterns. Stationary noise has CV < 0.05.
     */
    private fun isSpeechLikely(): Boolean {
        if (rmsHistoryCount < MIN_VAD_FRAMES) return true // not enough data yet, assume speech

        val n = rmsHistoryCount
        var sum = 0f
        var sumSq = 0f
        for (i in 0 until n) {
            val v = rmsHistory[i]
            sum += v
            sumSq += v * v
        }
        val mean = sum / n
        if (mean < 1e-7f) return false

        val variance = sumSq / n - mean * mean
        val cv = kotlin.math.sqrt(kotlin.math.max(0f, variance)) / mean

        return cv > VAD_CV_THRESHOLD
    }

    /**
     * Call after inference with the new score. Adapts the interval.
     */
    fun onInferenceResult(score: Float) {
        if (lastScore >= 0) {
            val scoreDelta = kotlin.math.abs(score - lastScore)
            if (scoreDelta < 0.05f) {
                // Score is stable — extend interval
                stableCount++
                currentInterval = when {
                    stableCount > 10 -> 10
                    stableCount > 5 -> 5
                    stableCount > 2 -> 2
                    else -> 1
                }
            } else {
                // Score changed significantly — reset to fast polling
                stableCount = 0
                currentInterval = 1
            }
        }
        lastScore = score
    }

    fun reset() {
        framesUntilNextInference = 0
        currentInterval = 1
        stableCount = 0
        lastRms = 0f
        lastScore = -1f
        rmsHistoryIdx = 0
        rmsHistoryCount = 0
        rmsHistory.fill(0f)
    }

    companion object {
        private const val VAD_WINDOW_SIZE = 5    // frames (seconds) of RMS history
        private const val MIN_VAD_FRAMES = 3     // need at least 3 frames before VAD kicks in
        private const val VAD_CV_THRESHOLD = 0.15f // CV below this = stationary noise
    }
}
