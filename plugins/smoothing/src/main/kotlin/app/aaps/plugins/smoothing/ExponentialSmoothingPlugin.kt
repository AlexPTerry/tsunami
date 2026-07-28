package app.aaps.plugins.smoothing

import app.aaps.core.data.iob.InMemoryGlucoseValue
import app.aaps.core.data.model.TrendArrow
import app.aaps.core.data.plugin.PluginType
import app.aaps.core.interfaces.logging.AAPSLogger
import app.aaps.core.interfaces.plugin.PluginBase
import app.aaps.core.interfaces.plugin.PluginDescription
import app.aaps.core.interfaces.resources.ResourceHelper
import app.aaps.core.interfaces.smoothing.Smoothing
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.round

@Singleton
class ExponentialSmoothingPlugin @Inject constructor(
    aapsLogger: AAPSLogger,
    rh: ResourceHelper
) : PluginBase(
    PluginDescription()
        .mainType(PluginType.SMOOTHING)
        .pluginIcon(app.aaps.core.ui.R.drawable.ic_timeline_24)
        .pluginName(R.string.exponential_smoothing_name)
        .shortName(R.string.smoothing_shortname)
        .description(R.string.description_exponential_smoothing),
    aapsLogger, rh
), Smoothing {

    override fun smooth(data: MutableList<InMemoryGlucoseValue>): MutableList<InMemoryGlucoseValue> {
        if (data.isEmpty()) return data

        val firstOrderWeight = 0.4
        val firstOrderAlpha5Minutes = 0.5
        val secondOrderAlpha5Minutes = 0.4
        val trendAlpha5Minutes = 1.0
        var windowSize = data.size

        // Work only on the newest continuous sensor segment.
        for (i in 0 until data.lastIndex) {
            val gapMinutes = (data[i].timestamp - data[i + 1].timestamp) / (1000.0 * 60.0)
            if (gapMinutes >= 12.0) {
                windowSize = i + 1
                break
            }
            if (data[i].value <= 38.0) {
                windowSize = i
                break
            }
        }

        if (windowSize < 4) {
            copyRaw(data)
            return data
        }

        val firstOrder = DoubleArray(windowSize)
        val secondOrder = DoubleArray(windowSize)
        val oldestIndex = windowSize - 1
        firstOrder[oldestIndex] = data[oldestIndex].value
        secondOrder[oldestIndex] = data[oldestIndex].value

        var firstLevel = data[oldestIndex].value
        var secondLevel = data[oldestIndex].value
        val initialDt = intervalMinutes(data[oldestIndex - 1], data[oldestIndex])
        var trendPerMinute = (data[oldestIndex - 1].value - data[oldestIndex].value) / initialDt

        // Input is newest-first; filter forward from the oldest point.
        for (i in oldestIndex - 1 downTo 0) {
            val dt = intervalMinutes(data[i], data[i + 1])

            val firstAlpha = alphaForInterval(firstOrderAlpha5Minutes, dt)
            firstLevel += firstAlpha * (data[i].value - firstLevel)
            firstOrder[i] = firstLevel

            val predictedLevel = secondLevel + trendPerMinute * dt
            val secondAlpha = alphaForInterval(secondOrderAlpha5Minutes, dt)
            val updatedLevel = secondAlpha * data[i].value + (1.0 - secondAlpha) * predictedLevel
            val trendAlpha = alphaForInterval(trendAlpha5Minutes, dt)
            trendPerMinute =
                trendAlpha * ((updatedLevel - secondLevel) / dt) +
                    (1.0 - trendAlpha) * trendPerMinute
            secondLevel = updatedLevel
            secondOrder[i] = secondLevel
        }

        for (i in data.indices) {
            data[i].smoothed =
                if (i < windowSize) {
                    max(round(firstOrderWeight * firstOrder[i] + (1.0 - firstOrderWeight) * secondOrder[i]), 39.0)
                } else {
                    max(data[i].value, 39.0)
                }
            data[i].trendArrow = TrendArrow.NONE
        }

        return data
    }

    private fun intervalMinutes(newer: InMemoryGlucoseValue, older: InMemoryGlucoseValue): Double =
        ((newer.timestamp - older.timestamp) / (1000.0 * 60.0)).coerceAtLeast(0.25)

    private fun alphaForInterval(fiveMinuteAlpha: Double, dtMinutes: Double): Double =
        if (fiveMinuteAlpha >= 1.0) 1.0
        else 1.0 - (1.0 - fiveMinuteAlpha).pow(dtMinutes / 5.0)

    private fun copyRaw(data: MutableList<InMemoryGlucoseValue>) {
        data.forEach {
            it.smoothed = max(it.value, 39.0)
            it.trendArrow = TrendArrow.NONE
        }
    }
}
