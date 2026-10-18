package io.github.kgonia.basal

import android.os.Build
import android.system.Os
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * A downloaded model directory: litert.json (written by convert/publish.py), the .tflite files, tokenizer.json and
 * CALIBRATION.json. One forward pass per prompt; the logits of the option codes at the last token are the decision.
 *
 * A model is one file or several parts run in a row (convert/export.py --part): the first takes `tokens`, each
 * next one the previous part's `hidden`, and the last one also `last` and returns `logits`.
 */
class Engine private constructor(
    private val parts: List<CompiledModel>,
    val accelerator: Accelerator,
    private val tokenizer: Tokenizer,
    private val calibration: Calibration,
    private val buckets: List<Int>,
    private val padId: Int,
) : AutoCloseable {
    data class Answer(
        val question: Format2.Question,
        val keys: List<String>,
        val options: List<String>,
        /** Calibrated probabilities in option order; for MULTI, P(label applies) per label. */
        val probabilities: List<Double>,
        val tokens: Int,
        val millis: Long,
    ) {
        val best get() = probabilities.indices.maxBy { probabilities[it] }
        val confidence get() = probabilities[best]
        /** SCORE: the expected level, the first level being 0. */
        val score get() = probabilities.withIndex().sumOf { (i, p) -> i * p }
        /** MULTI with the default threshold objective (FORMAT2 6.4): the labels with p >= 0.5. */
        val selected get() = keys.filterIndexed { i, _ -> probabilities[i] >= 0.5 }
    }

    private class Buffers(val inputs: Map<String, TensorBuffer>, val outputs: Map<String, TensorBuffer>)

    // per bucket length, per part
    private val buffers = HashMap<Int, List<Buffers>>()

    val maxTokens get() = buckets.last()

    private fun buffersFor(len: Int): List<Buffers> = buffers.getOrPut(len) {
        val sig = "p$len"
        parts.mapIndexed { i, m ->
            val ins = buildList {
                add(if (i == 0) "tokens" else "hidden")
                if (i == parts.lastIndex) add("last")
            }
            Buffers(ins.associateWith { m.createInputBuffer(it, sig) },
                    mapOf(if (i == parts.lastIndex) "logits" to m.createOutputBuffer("logits", sig)
                          else "hidden" to m.createOutputBuffer("hidden", sig)))
        }
    }

    /** The 255 code logits at the last token of [ids]. */
    @Synchronized
    fun codeLogits(ids: IntArray): FloatArray {
        val len = buckets.firstOrNull { it >= ids.size }
            ?: throw IllegalArgumentException("The prompt has ${ids.size} tokens; this model takes at most $maxTokens.")
        val sig = "p$len"
        val bufs = buffersFor(len)
        bufs[0].inputs.getValue("tokens").writeInt(IntArray(len) { if (it < ids.size) ids[it] else padId })
        bufs.last().inputs.getValue("last").writeInt(intArrayOf(ids.size - 1))
        for (i in parts.indices) {
            if (i > 0) bufs[i].inputs.getValue("hidden").writeFloat(bufs[i - 1].outputs.getValue("hidden").readFloat())
            parts[i].run(bufs[i].inputs, bufs[i].outputs, sig)
        }
        return bufs.last().outputs.getValue("logits").readFloat()
    }

    @Synchronized
    fun answer(state: String, q: Format2.Question): Answer {
        val t0 = System.nanoTime()
        val prompts = Format2.prompts(state, q)
        var tokens = 0
        val probs = prompts.map { p ->
            val ids = tokenizer.encode(p.text)
            tokens += ids.size
            val k = p.options.size
            Calibration.probabilities(codeLogits(ids).copyOf(k), calibration.temperature(q.type, k))
        }
        val millis = (System.nanoTime() - t0) / 1_000_000
        return if (q.type == Format2.Type.MULTI) {
            Answer(q, q.keys, prompts.map { it.label!! }, probs.map { it[0] }, tokens, millis)
        } else {
            Answer(q, prompts[0].keys, prompts[0].options, probs[0].toList(), tokens, millis)
        }
    }

    override fun close() {
        buffers.values.flatten().forEach { b -> (b.inputs.values + b.outputs.values).forEach { it.close() } }
        buffers.clear()
        parts.forEach { it.close() }
    }

    companion object {
        /** The SoC the NPU files are compiled for, e.g. "SM8550"; null before Android 12. */
        val soc: String? get() = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else null

        /** The files of the model for [accelerator]: litert.json's "file" (CPU), or "npu"'s entry for this SoC. */
        fun files(meta: JSONObject, accelerator: Accelerator): List<String>? {
            val v = if (accelerator == Accelerator.NPU) meta.optJSONObject("npu")?.opt(soc ?: "") else meta.opt("file")
            return when (v) {
                is String -> listOf(v)
                is JSONArray -> (0 until v.length()).map { v.getString(it) }
                else -> null
            }
        }

        /**
         * [cacheDir] keeps XNNPACK's packed weights between runs (CPU). [nativeLibDir] holds LiteRT's Qualcomm
         * dispatch library and the QNN runtime (NPU).
         */
        fun open(dir: File, accelerator: Accelerator, cacheDir: File, nativeLibDir: String? = null, threads: Int? = null): Engine {
            val meta = JSONObject(File(dir, "litert.json").readText())
            val tokenizer = Tokenizer.load(File(dir, "tokenizer.json").readText())
            // FORMAT2 3.6: every code must be one token of its own
            val codes = Format2.CODES.map { tokenizer.tokenToId(it) }
            require(codes.none { it == null } && codes.toSet().size == codes.size) { "tokenizer fails the code check" }
            val calibration = Calibration(File(dir, "CALIBRATION.json").readText())
            val b = meta.getJSONArray("buckets")
            val buckets = (0 until b.length()).map { b.getInt(it) }.sorted()
            val names = files(meta, accelerator) ?: throw IllegalArgumentException("no ${accelerator.name} files for ${soc ?: "this phone"}")
            val env = if (accelerator == Accelerator.NPU) {
                // the Hexagon DSP loads libQnnHtpV*Skel.so from this path
                Os.setenv("ADSP_LIBRARY_PATH", nativeLibDir!!, true)
                Environment.create(mapOf(Environment.Option.DispatchLibraryDir to nativeLibDir))
            } else null
            val parts = names.map { name ->
                val f = File(dir, name)
                if (env != null) {
                    CompiledModel.create(f.path, CompiledModel.Options(Accelerator.NPU, Accelerator.CPU), env)
                } else {
                    // Without a weight cache XNNPACK packs the weights once per signature, a copy of the model each,
                    // and the low-memory killer stops the app. The file cache is shared and kept for next time.
                    val options = CompiledModel.Options(Accelerator.CPU)
                    options.cpuOptions = CompiledModel.CpuOptions(threads, null, File(cacheDir, "${f.name}.${f.length()}.xnnpack").path)
                    CompiledModel.create(f.path, options)
                }
            }
            return Engine(parts, accelerator, tokenizer, calibration, buckets, meta.optInt("pad_id", 2))
        }
    }
}
