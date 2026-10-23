package io.github.kgonia.basal

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Runs a pushed model over the port-check items on this phone and writes the code logits in gate.py's row format.
 * Needs no screen, so it works on a locked phone:
 *
 *   adb push items_ids.jsonl /sdcard/Android/data/io.github.kgonia.basal/files/check/
 *   adb shell am instrument -w -e model basal-2.0-mini -e variant npu-SM8550 -e n 300 \
 *       io.github.kgonia.basal.test/androidx.test.runner.AndroidJUnitRunner
 *
 * Output: files/check/<model>_<variant>.jsonl (gate.py compare reads it) and the load time in logcat (tag basal-check).
 *
 * -e selftest 1 first runs litert.json's self-test (what the app runs on a chip nobody tested) and logs its result;
 * -e variant npu-SM8550 on another chip tries that chip's file there.
 */
@RunWith(AndroidJUnit4::class)
class DeviceCheck {
    @Test
    fun run() {
        val args = InstrumentationRegistry.getArguments()
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val modelId = args.getString("model", "basal-2.0-mini")
        val variant = args.getString("variant", "cpu")
        val n = args.getString("n", "0").toInt()
        val base = ctx.getExternalFilesDir(null)!!
        val items = File(base, "check/items_ids.jsonl").readLines().filter { it.isNotBlank() }.let { if (n > 0) it.take(n) else it }
        val out = File(base, "check/${modelId}_$variant.jsonl")

        val t0 = System.nanoTime()
        val threads = args.getString("threads")?.toInt()
        val engine = Engine.open(File(base, "models/$modelId"), variant, ctx.cacheDir, ctx.applicationInfo.nativeLibraryDir, threads)
        val loadMs = (System.nanoTime() - t0) / 1_000_000
        Log.i(TAG, "$modelId $variant: loaded in $loadMs ms")
        engine.use { e ->
            if (args.getString("selftest") == "1") {
                val st = System.nanoTime()
                val failed = e.selfTest(JSONObject(File(base, "models/$modelId/litert.json").readText()))
                Log.i(TAG, "selftest ${failed ?: "passed"} in ${(System.nanoTime() - st) / 1_000_000} ms")
            }
            out.bufferedWriter().use { w ->
                var total = 0L
                items.forEachIndexed { i, line ->
                    val it = JSONObject(line)
                    val idsJson = it.getJSONArray("ids")
                    val ids = IntArray(idsJson.length()) { idsJson.getInt(it) }
                    if (ids.size > e.maxTokens) return@forEachIndexed
                    val k = it.getJSONArray("keys").length()
                    val t = System.nanoTime()
                    // longer than the variant's buckets once its prefix is taken off
                    val logits = try { e.codeLogits(ids) } catch (_: IllegalArgumentException) { return@forEachIndexed }
                    val ms = (System.nanoTime() - t) / 1_000_000
                    total += ms
                    it.remove("ids")
                    it.put("tokens", ids.size).put("ms", ms)
                    // written by hand: JSONObject.toString() returns null for NaN, which a broken GPU run gives
                    val row = it.toString()
                    // an app readout: the NPU's own logits too, for comparison
                    val npu = e.lastNpuLogits?.let { l -> ", \"npu_logits\": [" + l.take(k).joinToString(", ") + "]" } ?: ""
                    w.write(row.dropLast(1) + ", \"logits\": [" + logits.take(k).joinToString(", ") + "]" + npu + "}")
                    w.newLine()
                    if ((i + 1) % 50 == 0) Log.i(TAG, "${i + 1} items, ${total / (i + 1)} ms/item")
                }
                Log.i(TAG, "done: ${items.size} items, ${total / maxOf(1, items.size)} ms/item, load $loadMs ms")
            }
        }
    }

    companion object {
        const val TAG = "basal-check"
    }
}
