package io.github.kgonia.basal

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Runs a pushed model over the port-check items on this phone and writes the code logits in gate.py's row format.
 * Needs no screen, so it works on a locked phone:
 *
 *   adb push items_ids.jsonl /sdcard/Android/data/io.github.kgonia.basal/files/check/
 *   adb shell am instrument -w -e model basal-2.0-mini -e accel GPU -e n 300 \
 *       io.github.kgonia.basal.test/androidx.test.runner.AndroidJUnitRunner
 *
 * Output: files/check/<model>_<accel>.jsonl (gate.py compare reads it) and the load time in logcat (tag basal-check).
 */
@RunWith(AndroidJUnit4::class)
class DeviceCheck {
    @Test
    fun run() {
        val args = InstrumentationRegistry.getArguments()
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val modelId = args.getString("model", "basal-2.0-mini")
        val accel = Accelerator.valueOf(args.getString("accel", "GPU"))
        val n = args.getString("n", "0").toInt()
        val precision = args.getString("precision")?.let { CompiledModel.GpuOptions.Precision.valueOf(it) }
        val base = ctx.getExternalFilesDir(null)!!
        val items = File(base, "check/items_ids.jsonl").readLines().filter { it.isNotBlank() }.let { if (n > 0) it.take(n) else it }
        val out = File(base, "check/${modelId}_$accel${precision?.let { "_$it" } ?: ""}.jsonl")

        val t0 = System.nanoTime()
        val threads = args.getString("threads")?.toInt()
        val engine = Engine.open(File(base, "models/$modelId"), accel, ctx.cacheDir, precision, threads)
        val loadMs = (System.nanoTime() - t0) / 1_000_000
        Log.i(TAG, "$modelId on $accel: loaded in $loadMs ms")
        engine.use { e ->
            out.bufferedWriter().use { w ->
                var total = 0L
                items.forEachIndexed { i, line ->
                    val it = JSONObject(line)
                    val idsJson = it.getJSONArray("ids")
                    val ids = IntArray(idsJson.length()) { idsJson.getInt(it) }
                    if (ids.size > e.maxTokens) return@forEachIndexed
                    val k = it.getJSONArray("keys").length()
                    val t = System.nanoTime()
                    val logits = e.codeLogits(ids)
                    val ms = (System.nanoTime() - t) / 1_000_000
                    total += ms
                    it.remove("ids")
                    it.put("tokens", ids.size).put("ms", ms)
                    // written by hand: JSONObject.toString() returns null for NaN, which a broken GPU run gives
                    val row = it.toString()
                    w.write(row.dropLast(1) + ", \"logits\": [" + logits.take(k).joinToString(", ") + "]}")
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
