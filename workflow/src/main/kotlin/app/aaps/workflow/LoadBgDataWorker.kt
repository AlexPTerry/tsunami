package app.aaps.workflow

import android.content.Context
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import app.aaps.core.data.time.T
import app.aaps.core.interfaces.aps.AutosensDataStore
import app.aaps.core.interfaces.db.PersistenceLayer
import app.aaps.core.interfaces.iob.IobCobCalculator
import app.aaps.core.interfaces.logging.AAPSLogger
import app.aaps.core.interfaces.logging.LTag
import app.aaps.core.interfaces.plugin.ActivePlugin
import app.aaps.core.interfaces.rx.bus.RxBus
import app.aaps.core.interfaces.rx.events.EventBucketedDataCreated
import app.aaps.core.interfaces.utils.DateUtil
import app.aaps.core.objects.extensions.fromGv
import app.aaps.core.objects.workflow.LoggingWorker
import app.aaps.core.utils.receivers.DataWorkerStorage
import kotlinx.coroutines.Dispatchers
import javax.inject.Inject

class LoadBgDataWorker(
    context: Context,
    params: WorkerParameters
) : LoggingWorker(context, params, Dispatchers.Default) {

    @Inject lateinit var dataWorkerStorage: DataWorkerStorage
    @Inject lateinit var dateUtil: DateUtil
    @Inject lateinit var rxBus: RxBus
    @Inject lateinit var persistenceLayer: PersistenceLayer
    @Inject lateinit var activePlugin: ActivePlugin

    class LoadBgData(
        val iobCobCalculator: IobCobCalculator,
        val end: Long
    )

    private fun AutosensDataStore.loadBgData(to: Long, persistenceLayer: PersistenceLayer, aapsLogger: AAPSLogger, dateUtil: DateUtil) {
        synchronized(dataLock) {
            val start = to - T.hours((24 + 10 /* max dia */).toLong()).msecs()
            // there can be some readings with time in close future (caused by wrong time setting on sensor)
            // so add 2 minutes
            bgReadings = persistenceLayer
                .getBgReadingsDataFromTimeToTime(start, to + T.mins(2).msecs(), false)
            aapsLogger.debug(LTag.AUTOSENS) { "BG data loaded. Size: ${bgReadings.size} Start date: ${dateUtil.dateAndTimeString(start)} End date: ${dateUtil.dateAndTimeString(to)}" }
        }
    }

    private fun AutosensDataStore.smoothData(activePlugin: ActivePlugin) {
        synchronized(dataLock) {
            val denseData = bgReadings
                .sortedByDescending { it.timestamp }
                .map { app.aaps.core.data.iob.InMemoryGlucoseValue.fromGv(it) }
                .toMutableList()

            smoothedData = activePlugin.activeSmoothing.smooth(denseData)

            val denseSmoothedData = smoothedData ?: return
            if (denseSmoothedData.none { it.smoothed != null }) return

            bucketedData?.forEach { bucket ->
                interpolateSmoothedValue(denseSmoothedData, bucket.timestamp)?.let { bucket.smoothed = it }
            }
        }
    }

    private fun interpolateSmoothedValue(
        data: List<app.aaps.core.data.iob.InMemoryGlucoseValue>,
        timestamp: Long
    ): Double? {
        val olderIndex = data.indexOfFirst { it.timestamp <= timestamp }
        if (olderIndex < 0) return null

        val older = data[olderIndex]
        if (older.timestamp == timestamp) return older.smoothed
        if (olderIndex == 0) return null

        val newer = data[olderIndex - 1]
        val olderValue = older.smoothed ?: return null
        val newerValue = newer.smoothed ?: return null
        val interval = newer.timestamp - older.timestamp
        if (interval <= 0L) return null

        val fraction = (timestamp - older.timestamp).toDouble() / interval
        return olderValue + fraction * (newerValue - olderValue)
    }

    override suspend fun doWorkAndLog(): Result {

        val data = dataWorkerStorage.pickupObject(inputData.getLong(DataWorkerStorage.STORE_KEY, -1)) as LoadBgData?
            ?: return Result.failure(workDataOf("Error" to "missing input data"))

        data.iobCobCalculator.ads.loadBgData(data.end, persistenceLayer, aapsLogger, dateUtil)
        data.iobCobCalculator.ads.createBucketedData(aapsLogger, dateUtil)
        data.iobCobCalculator.ads.smoothData(activePlugin)
        rxBus.send(EventBucketedDataCreated())
        data.iobCobCalculator.clearCache()
        return Result.success()
    }
}
