package io.github.kgonia.basal

import android.os.Build
import android.system.Os
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import org.json.JSONObject
import java.io.File
import java.util.Locale
import kotlin.math.abs

/**
 * A downloaded model directory: litert.json (written by convert/publish.py), the .tflite files, tokenizer.json and
 * CALIBRATION.json. One forward pass per prompt; the logits of the option codes at the last token are the decision.
 *
 * litert.json lists variants: "cpu" for every phone, "npu-<SoC>" compiled ahead of time for one Qualcomm SoC. A
 * variant is one file or several parts run in a row (convert/export.py --blocks): the first takes `tokens`, each
 * next one the previous part's `hidden`, and the last one also `last` and returns `logits`. An NPU variant may hold
 * the keys and values of the prompt start every prompt of a language shares ("prefix": token ids per language):
 * its signatures are <lang>_p<L> and take only the tokens after that prefix.
 */
class Engine private constructor(
    private val parts: List<CompiledModel>,
    val accelerator: Accelerator,
    private val tokenizer: Tokenizer,
    private val calibration: Calibration,
    private val buckets: List<Int>,
    private val prefixes: Map<String, IntArray>,
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

    // per signature, per part
    private val buffers = HashMap<String, List<Buffers>>()

    /** The longest prompt the model takes, in tokens. */
    val maxTokens get() = buckets.last() + (prefixes.values.minOfOrNull { it.size } ?: 0)

    private fun buffersFor(sig: String): List<Buffers> = buffers.getOrPut(sig) {
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
        // the cached prompt start of the prompt's language, if the variant has one
        val (lang, prefix) = if (prefixes.isEmpty()) null to IntArray(0) else prefixes.entries
            .firstOrNull { (_, p) -> ids.size > p.size && p.indices.all { ids[it] == p[it] } }?.toPair()
            ?: throw IllegalArgumentException("The prompt does not start like a basal prompt of a known language.")
        val n = ids.size - prefix.size
        val len = buckets.firstOrNull { it >= n }
            ?: throw IllegalArgumentException("The prompt has ${ids.size} tokens; this model takes at most $maxTokens.")
        val sig = if (lang == null) "p$len" else "${lang}_p$len"
        val bufs = buffersFor(sig)
        bufs[0].inputs.getValue("tokens").writeInt(IntArray(len) { if (it < n) ids[prefix.size + it] else padId })
        bufs.last().inputs.getValue("last").writeInt(intArrayOf(n - 1))
        for (i in parts.indices) {
            if (i > 0) bufs[i].inputs.getValue("hidden").writeFloat(bufs[i - 1].outputs.getValue("hidden").readFloat())
            parts[i].run(bufs[i].inputs, bufs[i].outputs, sig)
        }
        return bufs.last().outputs.getValue("logits").readFloat()
    }

    /**
     * Runs litert.json's "selftest" prompts, one per language and bucket so that every signature runs, and compares
     * the logits with the reference ones: an NPU file compiled for a chip nobody tested may load and still compute
     * nonsense. Null when all pass, else what went wrong.
     */
    fun selfTest(meta: JSONObject): String? {
        val cases = meta.optJSONArray("selftest") ?: return null
        for (c in 0 until cases.length()) {
            val t = cases.getJSONObject(c)
            val ids = t.getJSONArray("ids").let { a -> IntArray(a.length()) { a.getInt(it) } }
            if (ids.size > maxTokens) continue
            val want = t.getJSONArray("logits").let { a -> DoubleArray(a.length()) { a.getDouble(it) } }
            val got = codeLogits(ids)
            val diff = want.indices.maxOf { abs(got[it] - want[it]) }
            val same = want.indices.maxBy { want[it] } == want.indices.maxBy { got[it].toDouble() }
            if (!same || diff > t.optDouble("tolerance", 1.0)) {
                return String.format(Locale.ROOT, "%s, %d tokens: logits differ by up to %.2f from the reference",
                                     t.optString("lang"), ids.size, diff)
            }
        }
        return null
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

        /** The NPU variant's name for this phone, or null when litert.json has none. */
        fun npuVariant(meta: JSONObject): String? =
            soc?.let { "npu-$it" }?.takeIf { meta.getJSONObject("variants").has(it) }

        /** A variant's files: name and size in bytes. */
        fun files(meta: JSONObject, variant: String): List<Pair<String, Long>> {
            val a = meta.getJSONObject("variants").getJSONObject(variant).getJSONArray("files")
            return (0 until a.length()).map { a.getJSONObject(it).let { f -> f.getString("name") to f.getLong("size") } }
        }

        /**
         * [cacheDir] keeps XNNPACK's packed weights between runs (CPU). [nativeLibDir] holds LiteRT's Qualcomm
         * dispatch library and the QNN runtime (NPU).
         */
        fun open(dir: File, variant: String, cacheDir: File, nativeLibDir: String? = null, threads: Int? = null): Engine {
            val meta = JSONObject(File(dir, "litert.json").readText())
            val tokenizer = Tokenizer.load(File(dir, "tokenizer.json").readText())
            // FORMAT2 3.6: every code must be one token of its own
            val codes = Format2.CODES.map { tokenizer.tokenToId(it) }
            require(codes.none { it == null } && codes.toSet().size == codes.size) { "tokenizer fails the code check" }
            val calibration = Calibration(File(dir, "CALIBRATION.json").readText())
            val v = meta.getJSONObject("variants").getJSONObject(variant)
            val buckets = v.getJSONArray("buckets").let { b -> (0 until b.length()).map { b.getInt(it) }.sorted() }
            val prefixes = v.optJSONObject("prefix")?.let { p ->
                p.keys().asSequence().associateWith { lang -> p.getJSONArray(lang).let { a -> IntArray(a.length()) { a.getInt(it) } } }
            } ?: emptyMap()
            val npu = variant.startsWith("npu-")
            val env = if (npu) {
                // the Hexagon DSP loads libQnnHtpV*Skel.so from this path
                Os.setenv("ADSP_LIBRARY_PATH", nativeLibDir!!, true)
                Environment.create(mapOf(Environment.Option.DispatchLibraryDir to nativeLibDir))
            } else null
            val parts = files(meta, variant).map { (name, _) ->
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
            return Engine(parts, if (npu) Accelerator.NPU else Accelerator.CPU, tokenizer, calibration, buckets, prefixes,
                          meta.optInt("pad_id", 2))
        }
    }
}
