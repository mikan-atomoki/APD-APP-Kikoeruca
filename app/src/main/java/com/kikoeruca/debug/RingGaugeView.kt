package com.kikoeruca.debug

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

/**
 * Arc gauge with a gap at the bottom.
 * Draws a track (light gray) and a fill arc whose length is proportional to the score.
 * The arc spans 308° with a 52° gap at the bottom.
 */
class RingGaugeView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val density = resources.displayMetrics.density
    private val strokePx = 14f * density

    private val trackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = strokePx
        strokeCap = Paint.Cap.ROUND
        color = 0xFFE8E8E8.toInt()
    }

    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = strokePx
        strokeCap = Paint.Cap.ROUND
        color = 0xFF4CDD4C.toInt()
    }

    private val rect = RectF()

    // Arc geometry: 308° sweep with 52° gap at bottom
    // Start angle: 116° from 3 o'clock position (= bottom-left of gap)
    // Sweep: 308° clockwise
    private val arcStartAngle = 116f
    private val arcTotalSweep = 308f

    /** Score from 0.0 to 1.0 — determines how much of the arc is filled */
    var score: Float = 1.0f
        set(value) {
            field = value.coerceIn(0f, 1f)
            invalidate()
        }

    /** Color of the fill arc */
    var gaugeColor: Int = 0xFF4CDD4C.toInt()
        set(value) {
            field = value
            fillPaint.color = value
            invalidate()
        }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val inset = strokePx / 2f + 4f * density
        rect.set(inset, inset, width - inset, height - inset)

        // Draw track (full arc, light gray)
        canvas.drawArc(rect, arcStartAngle, arcTotalSweep, false, trackPaint)

        // Draw fill arc (proportional to score)
        val fillSweep = arcTotalSweep * score
        if (fillSweep > 0f) {
            canvas.drawArc(rect, arcStartAngle, fillSweep, false, fillPaint)
        }
    }
}
