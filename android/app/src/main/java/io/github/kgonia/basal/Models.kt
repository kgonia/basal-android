package io.github.kgonia.basal

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/** The models the app offers: LiteRT ports of basal-2.0 on Hugging Face (convert/publish.py writes them). */
data class CatalogModel(val id: String, val repo: String, val label: String, val source: String) {
    fun url(file: String) = "https://huggingface.co/$repo/resolve/main/$file"
}

val CATALOG = listOf(
    CatalogModel("basal-2.0-mini", "SearchingMan/basal-2.0-mini-LiteRT", "mini · 1.5B", "Remek/basal-2.0-mini"),
    CatalogModel("basal-2.0-4.5B", "SearchingMan/basal-2.0-4.5B-LiteRT", "4.5B", "Remek/basal-2.0-4.5B"),
)

/**
 * Model files live in the app's external files dir, models/<id>/ (adb can push them there too). litert.json is
 * written last, so its presence means the download is complete.
 */
class ModelStore(private val context: Context) {
    private val dm = context.getSystemService(DownloadManager::class.java)

    fun dir(m: CatalogModel) = File(context.getExternalFilesDir(null), "models/${m.id}")

    fun isReady(m: CatalogModel) = File(dir(m), "litert.json").isFile

    /** The model's litert.json on Hugging Face (a network call): its .tflite file name, size and buckets. */
    fun fetchManifest(m: CatalogModel): JSONObject {
        val c = URL(m.url("litert.json")).openConnection() as HttpURLConnection
        c.connectTimeout = 15000
        c.readTimeout = 15000
        try {
            if (c.responseCode != 200) throw IllegalStateException("HTTP ${c.responseCode} for ${m.url("litert.json")}")
            return JSONObject(c.inputStream.bufferedReader().readText())
        } finally {
            c.disconnect()
        }
    }

    /** Starts the downloads; returns their ids. The manifest is saved once every file has arrived. */
    fun startDownload(m: CatalogModel, manifest: JSONObject): List<Long> {
        val d = dir(m).apply { mkdirs() }
        File(d, "litert.json.pending").writeText(manifest.toString())
        val files = listOf(manifest.getString("file"), "tokenizer.json", "CALIBRATION.json")
        return files.map { name ->
            File(d, name).delete()
            dm.enqueue(DownloadManager.Request(Uri.parse(m.url(name)))
                .setTitle("${m.id}: $name")
                .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE)
                .setDestinationInExternalFilesDir(context, null, "models/${m.id}/$name"))
        }
    }

    data class Progress(val done: Long, val total: Long, val failed: Boolean, val finished: Boolean)

    fun progress(ids: List<Long>): Progress {
        var done = 0L
        var total = 0L
        var failed = false
        var finished = 0
        dm.query(DownloadManager.Query().setFilterById(*ids.toLongArray())).use { c ->
            while (c.moveToNext()) {
                done += c.getLong(c.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR))
                total += maxOf(0, c.getLong(c.getColumnIndexOrThrow(DownloadManager.COLUMN_TOTAL_SIZE_BYTES)))
                when (c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))) {
                    DownloadManager.STATUS_FAILED -> failed = true
                    DownloadManager.STATUS_SUCCESSFUL -> finished++
                }
            }
        }
        return Progress(done, total, failed, finished == ids.size)
    }

    /** After every download succeeded: checks the model file's size and makes the model ready. */
    fun finish(m: CatalogModel) {
        val d = dir(m)
        val pending = File(d, "litert.json.pending")
        val manifest = JSONObject(pending.readText())
        val size = File(d, manifest.getString("file")).length()
        check(size == manifest.getLong("size")) { "model file has $size bytes, expected ${manifest.getLong("size")}" }
        pending.renameTo(File(d, "litert.json"))
    }

    fun cancel(ids: List<Long>) {
        if (ids.isNotEmpty()) dm.remove(*ids.toLongArray())
    }

    fun delete(m: CatalogModel) {
        dir(m).deleteRecursively()
    }
}
