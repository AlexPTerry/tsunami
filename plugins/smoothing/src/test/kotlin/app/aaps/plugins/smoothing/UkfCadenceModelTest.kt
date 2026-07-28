package app.aaps.plugins.smoothing

import com.google.common.truth.Truth.assertThat
import org.junit.jupiter.api.Test

class UkfCadenceModelTest {

    private val nominalQ = doubleArrayOf(1.0, 0.0, 0.0, 0.4)

    @Test
    fun fiveMinuteValuesMatchLegacyCalibration() {
        val transition = UkfCadenceModel.transition(5.0)
        val noise = UkfCadenceModel.processNoise(nominalQ, 5.0)

        assertThat(transition.glucoseFromRate).isWithin(1e-10).of(5.0)
        assertThat(transition.rateDamping).isWithin(1e-10).of(0.98)
        noise.indices.forEach { i ->
            assertThat(noise[i]).isWithin(1e-9).of(nominalQ[i])
        }
        assertThat(UkfCadenceModel.measurementNoise(25.0, 5.0)).isWithin(1e-10).of(25.0)
    }

    @Test
    fun fiveOneMinutePredictionsEqualOneFiveMinutePrediction() {
        val oneMinuteF = transitionMatrix(1.0)
        val oneMinuteQ = UkfCadenceModel.processNoise(nominalQ, 1.0)
        var accumulatedF = identity()
        var accumulatedQ = DoubleArray(4)

        repeat(5) {
            accumulatedQ = add(multiply(multiply(oneMinuteF, accumulatedQ), transpose(oneMinuteF)), oneMinuteQ)
            accumulatedF = multiply(oneMinuteF, accumulatedF)
        }

        val fiveMinuteF = transitionMatrix(5.0)
        val fiveMinuteQ = UkfCadenceModel.processNoise(nominalQ, 5.0)
        accumulatedF.indices.forEach { i ->
            assertThat(accumulatedF[i]).isWithin(1e-9).of(fiveMinuteF[i])
            assertThat(accumulatedQ[i]).isWithin(1e-9).of(fiveMinuteQ[i])
        }
    }

    @Test
    fun arbitrarySubdivisionsComposeToSamePredictionNoise() {
        val intervals = listOf(2.0, 1.0, 2.0)
        var accumulatedQ = DoubleArray(4)
        for (dt in intervals) {
            val f = transitionMatrix(dt)
            accumulatedQ = add(
                multiply(multiply(f, accumulatedQ), transpose(f)),
                UkfCadenceModel.processNoise(nominalQ, dt)
            )
        }

        val fiveMinuteQ = UkfCadenceModel.processNoise(nominalQ, 5.0)
        accumulatedQ.indices.forEach { i ->
            assertThat(accumulatedQ[i]).isWithin(1e-9).of(fiveMinuteQ[i])
        }
    }

    @Test
    fun oneMinuteMeasurementsKeepFiveMinuteInformationBudget() {
        assertThat(UkfCadenceModel.measurementNoise(25.0, 1.0)).isWithin(1e-10).of(125.0)
    }

    private fun transitionMatrix(dt: Double): DoubleArray {
        val transition = UkfCadenceModel.transition(dt)
        return doubleArrayOf(1.0, transition.glucoseFromRate, 0.0, transition.rateDamping)
    }

    private fun identity() = doubleArrayOf(1.0, 0.0, 0.0, 1.0)

    private fun transpose(matrix: DoubleArray) =
        doubleArrayOf(matrix[0], matrix[2], matrix[1], matrix[3])

    private fun add(a: DoubleArray, b: DoubleArray) =
        DoubleArray(4) { i -> a[i] + b[i] }

    private fun multiply(a: DoubleArray, b: DoubleArray) =
        doubleArrayOf(
            a[0] * b[0] + a[1] * b[2],
            a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2],
            a[2] * b[1] + a[3] * b[3]
        )
}
