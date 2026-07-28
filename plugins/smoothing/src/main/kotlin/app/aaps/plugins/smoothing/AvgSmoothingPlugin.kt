package app.aaps.plugins.smoothing

import app.aaps.core.data.iob.InMemoryGlucoseValue
import app.aaps.core.data.model.TrendArrow
import app.aaps.core.data.plugin.PluginType
import app.aaps.core.data.time.T
import app.aaps.core.interfaces.logging.AAPSLogger
import app.aaps.core.interfaces.logging.LTag
import app.aaps.core.interfaces.plugin.PluginBase
import app.aaps.core.interfaces.plugin.PluginDescription
import app.aaps.core.interfaces.resources.ResourceHelper
import app.aaps.core.interfaces.smoothing.Smoothing
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class AvgSmoothingPlugin @Inject constructor(
    aapsLogger: AAPSLogger,
    rh: ResourceHelper
) : PluginBase(
    PluginDescription()
        .mainType(PluginType.SMOOTHING)
        .pluginIcon(app.aaps.core.ui.R.drawable.ic_timeline_24)
        .pluginName(R.string.avg_smoothing_name)
        .shortName(R.string.smoothing_shortname)
        .description(R.string.description_avg_smoothing),
    aapsLogger, rh
), Smoothing {

    override fun smooth(data: MutableList<InMemoryGlucoseValue>): MutableList<InMemoryGlucoseValue> {
        if (data.lastIndex < 4) {
            aapsLogger.debug(LTag.GLUCOSE, "Not enough value's to smooth!")
            return data
        }

        for (i in data.indices) {
            val newerValue = interpolateValue(data, data[i].timestamp + T.mins(5).msecs())
            val olderValue = interpolateValue(data, data[i].timestamp - T.mins(5).msecs())
            if (isValid(data[i].value) && newerValue != null && olderValue != null && isValid(newerValue) && isValid(olderValue)) {
                // Sample the same +/- five-minute window regardless of source cadence.
                // On regular five-minute data this is identical to the legacy three-point average.
                data[i].smoothed = (newerValue + data[i].value + olderValue) / 3.0
                data[i].trendArrow = TrendArrow.NONE
            } else {
                val currentTime = data[i].timestamp
                val value = data[i].value
                aapsLogger.debug(LTag.GLUCOSE, "Value: $value at $currentTime not smoothed")
            }
        }
        return data
    }

    private fun interpolateValue(data: List<InMemoryGlucoseValue>, timestamp: Long): Double? {
        val olderIndex = data.indexOfFirst { it.timestamp <= timestamp }
        if (olderIndex < 0) return null
        val older = data[olderIndex]
        if (older.timestamp == timestamp) return older.value
        if (olderIndex == 0) return null

        val newer = data[olderIndex - 1]
        val interval = newer.timestamp - older.timestamp
        if (interval <= 0L || interval > T.mins(12).msecs()) return null
        val fraction = (timestamp - older.timestamp).toDouble() / interval
        return older.value + fraction * (newer.value - older.value)
    }

    private fun isValid(n: Double): Boolean {
        // For Dexcom: Below 39 is LOW, above 401 Dexcom just says HI
        return n > 39 && n < 401
    }
}
