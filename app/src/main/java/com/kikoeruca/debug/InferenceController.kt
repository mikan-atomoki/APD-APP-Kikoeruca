package com.kikoeruca.debug

/**
 * Controls when to invoke the AI model based on RMS levels.
 *
 * Strategy:
 *   - Always compute RMS (near-zero cost via native)
 *   - If RMS < quietThreshold → environment is quiet, no inference needed
 *   - If RMS spikes (rapid change) → immediate inference
 *   - If RMS is above threshold and stable → inference at adaptive interval
 *   - Interval grows from 1s → 5s → 10s as score stabilizes
 */
class InferenceController {

    // RMS thresholds (configurable)
    var quietThreshold: Float = 0.005f      // below this = silence, skip inference
    var spikeRatio: Float = 3.0f            // RMS jumps by this factor = spike

    // Adaptive interval (frames, where 1 frame = 1 second)
    private var framesUntilNextInference = 0
    private var currentInterval = 1          // start at every frame
    private var stableCount = 0
    private var lastRms = 0f
    private var lastScore = -1f

    enum class Action {
        SKIP_QUIET,       // too quiet, don't infer
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
    }
}
