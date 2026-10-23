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
    // not published yet: CatalogModel("basal-2.0-4.5B", "SearchingMan/basal-2.0-4.5B-LiteRT", "4.5B", "Remek/basal-2.0-4.5B"),
)

/**
 * Model files live in the app's external files dir, models/<id>/ (adb can push them there too): litert.json, the
 * tokenizer and calibration, and the files of the variants downloaded. A variant is ready when each of its files has
 * the size litert.json gives.
 */
class ModelStore(private val context: Context) {
    private val dm = context.getSystemService(DownloadManager::class.java)
    private val prefs = context.getSharedPreferences("models", Context.MODE_PRIVATE)

    fun dir(m: CatalogModel) = File(context.getExternalFilesDir(null), "models/${m.id}")

    fun manifest(m: CatalogModel): JSONObject? =
        File(dir(m), "litert.json").takeIf { it.isFile }?.let { runCatching { JSONObject(it.readText()) }.getOrNull() }

    private fun complete(m: CatalogModel, meta: JSONObject, variant: String) =
        meta.getJSONObject("variants").has(variant) &&
            (Engine.files(meta, variant) + COMMON.map { it to -1L }).all { (name, size) ->
                File(dir(m), name).let { it.isFile && (size < 0 || it.length() == size) }
            }

    /** The variant to run: this phone's NPU one if downloaded and not failed its self-test, else the CPU one. */
    fun readyVariant(m: CatalogModel): String? {
        val meta = manifest(m) ?: return null
        val npu = Engine.npuVariant(meta)
        return when {
            npu != null && !npuFailed(m) && complete(m, meta, npu) -> npu
            complete(m, meta, CPU) -> CPU
            else -> null
        }
    }

    /** An NPU variant that loaded but failed its self-test is not used again for this model. */
    fun npuFailed(m: CatalogModel) = prefs.getBoolean("npu_failed.${m.id}", false)

    fun markNpuFailed(m: CatalogModel) = prefs.edit().putBoolean("npu_failed.${m.id}", true).apply()

    /** The NPU self-test runs once per downloaded variant: its key names the files and their sizes. */
    private fun testKey(m: CatalogModel, variant: String) =
        "npu_ok.${m.id}." + manifest(m)?.let { Engine.files(it, variant).joinToString(",") { (n, s) -> "$n:$s" } }

    fun npuPassed(m: CatalogModel, variant: String) = prefs.getBoolean(testKey(m, variant), false)

    fun markNpuPassed(m: CatalogModel, variant: String) = prefs.edit().putBoolean(testKey(m, variant), true).apply()

    /** The model's litert.json on Hugging Face (a network call): its variants, files and sizes. */
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

    /**
     * Bytes on disk once [variant] is running: its files, and for the CPU XNNPACK's packed copy of the weights in the
     * cache dir (about the size of the model again).
     */
    fun diskNeeded(meta: JSONObject, variant: String): Long {
        val files = Engine.files(meta, variant).sumOf { it.second }
        return if (variant == CPU) 2 * files else files
    }

    /** Starts the downloads of [variant]; returns their ids. Files already complete are kept. */
    fun startDownload(m: CatalogModel, meta: JSONObject, variant: String): List<Long> {
        val d = dir(m).apply { mkdirs() }
        File(d, "litert.json").writeText(meta.toString())
        val free = d.usableSpace
        val need = diskNeeded(meta, variant)
        check(free > need) { context.getString(R.string.no_space, need / 1e9, free / 1e9) }
        val files = Engine.files(meta, variant) + COMMON.map { it to -1L }
        return files.filterNot { (name, size) -> File(d, name).let { it.isFile && it.length() == size } }.map { (name, _) ->
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

    fun cancel(ids: List<Long>) {
        if (ids.isNotEmpty()) dm.remove(*ids.toLongArray())
    }

    fun delete(m: CatalogModel) {
        dir(m).deleteRecursively()
        context.cacheDir.listFiles { f -> f.name.endsWith(".xnnpack") && f.name.startsWith(m.id) }?.forEach { it.delete() }
        prefs.edit().remove("npu_failed.${m.id}").apply()
    }

    companion object {
        const val CPU = "cpu"
        /** The files every variant needs, from the source model unchanged. */
        val COMMON = listOf("tokenizer.json", "CALIBRATION.json")
    }
}
