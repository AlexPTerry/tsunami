package app.aaps.plugins.smoothing

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

/**
 * Converts the UKF's established five-minute dynamics to an arbitrary elapsed time.
 *
 * The conversion is deliberately calibrated so that F(5) and Q(5) are exactly the
 * legacy matrices. It changes cadence handling without inventing new physiological
 * tuning before representative one-minute sensor data is available.
 */
internal object UkfCadenceModel {

    const val NOMINAL_INTERVAL_MINUTES = 5.0
    const val RATE_DAMPING_5_MINUTES = 0.98
    const val MINIMUM_INTERVAL_MINUTES = 0.25

    private val decayRate = ln(RATE_DAMPING_5_MINUTES) / NOMINAL_INTERVAL_MINUTES
    private val glucoseRateScale = NOMINAL_INTERVAL_MINUTES / (1.0 - RATE_DAMPING_5_MINUTES)

    data class Transition(
        val glucoseFromRate: Double,
        val rateDamping: Double
    )

    fun transition(dtMinutes: Double): Transition {
        val dt = max(dtMinutes, MINIMUM_INTERVAL_MINUTES)
        val damping = exp(decayRate * dt)
        return Transition(
            glucoseFromRate = glucoseRateScale * (1.0 - damping),
            rateDamping = damping
        )
    }

    /**
     * Discretize a continuous diffusion matrix selected to reproduce [nominalQ]
     * exactly at the five-minute calibration interval.
     */
    fun processNoise(nominalQ: DoubleArray, dtMinutes: Double): DoubleArray {
        require(nominalQ.size == 4)
        val dt = max(dtMinutes, MINIMUM_INTERVAL_MINUTES)
        val nominalIntegrals = integrals(NOMINAL_INTERVAL_MINUTES)

        val diffusionRate = nominalQ[3] / nominalIntegrals.exp2
        val diffusionCross = (
            nominalQ[1] -
                diffusionRate * glucoseRateScale * (nominalIntegrals.exp1 - nominalIntegrals.exp2)
            ) / nominalIntegrals.exp1
        val diffusionGlucose = (
            nominalQ[0] -
                2.0 * diffusionCross * glucoseRateScale * (NOMINAL_INTERVAL_MINUTES - nominalIntegrals.exp1) -
                diffusionRate * glucoseRateScale * glucoseRateScale *
                (NOMINAL_INTERVAL_MINUTES - 2.0 * nominalIntegrals.exp1 + nominalIntegrals.exp2)
            ) / NOMINAL_INTERVAL_MINUTES

        val elapsedIntegrals = integrals(dt)
        val qRate = diffusionRate * elapsedIntegrals.exp2
        val qCross =
            diffusionCross * elapsedIntegrals.exp1 +
                diffusionRate * glucoseRateScale * (elapsedIntegrals.exp1 - elapsedIntegrals.exp2)
        val qGlucose =
            diffusionGlucose * dt +
                2.0 * diffusionCross * glucoseRateScale * (dt - elapsedIntegrals.exp1) +
                diffusionRate * glucoseRateScale * glucoseRateScale *
                (dt - 2.0 * elapsedIntegrals.exp1 + elapsedIntegrals.exp2)

        return doubleArrayOf(qGlucose, qCross, qCross, qRate)
    }

    /**
     * Until one-minute residuals can be characterized, limit measurement information
     * to at most the legacy five-minute information rate.
     */
    fun measurementNoise(baseFiveMinuteR: Double, dtMinutes: Double): Double {
        val dt = max(dtMinutes, MINIMUM_INTERVAL_MINUTES)
        val cadenceMultiplier = (NOMINAL_INTERVAL_MINUTES / dt).coerceIn(1.0, NOMINAL_INTERVAL_MINUTES)
        return baseFiveMinuteR * cadenceMultiplier
    }

    private data class Integrals(
        val exp1: Double,
        val exp2: Double
    )

    private fun integrals(dtMinutes: Double): Integrals =
        Integrals(
            exp1 = (exp(decayRate * dtMinutes) - 1.0) / decayRate,
            exp2 = (exp(2.0 * decayRate * dtMinutes) - 1.0) / (2.0 * decayRate)
        )
}
