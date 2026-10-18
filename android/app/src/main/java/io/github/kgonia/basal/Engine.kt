package io.github.kgonia.basal

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import org.json.JSONObject
import java.io.File

/**
 * A downloaded model directory: litert.json (written by convert/publish.py), the .tflite file, tokenizer.json and
 * CALIBRATION.json. One forward pass per prompt; the logits of the option codes at the last token are the decision.
 */
class Engine private constructor(
    private val model: CompiledModel,
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

    private val buffers = HashMap<Int, Pair<Map<String, TensorBuffer>, Map<String, TensorBuffer>>>()

    val maxTokens get() = buckets.last()

    private fun codeLogits(ids: IntArray): FloatArray {
        val len = buckets.firstOrNull { it >= ids.size }
            ?: throw IllegalArgumentException("The prompt has ${ids.size} tokens; this model file takes at most $maxTokens.")
        val sig = "p$len"
        val (inputs, outputs) = buffers.getOrPut(len) {
            mapOf("tokens" to model.createInputBuffer("tokens", sig), "last" to model.createInputBuffer("last", sig)) to
                mapOf("logits" to model.createOutputBuffer("logits", sig))
        }
        val padded = IntArray(len) { if (it < ids.size) ids[it] else padId }
        inputs.getValue("tokens").writeInt(padded)
        inputs.getValue("last").writeInt(intArrayOf(ids.size - 1))
        model.run(inputs, outputs, sig)
        return outputs.getValue("logits").readFloat()
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
        buffers.values.forEach { (i, o) -> (i.values + o.values).forEach { it.close() } }
        buffers.clear()
        model.close()
    }

    companion object {
        fun open(dir: File, accelerator: Accelerator): Engine {
            val meta = JSONObject(File(dir, "litert.json").readText())
            val tokenizer = Tokenizer.load(File(dir, "tokenizer.json").readText())
            // FORMAT2 3.6: every code must be one token of its own
            val codes = Format2.CODES.map { tokenizer.tokenToId(it) }
            require(codes.none { it == null } && codes.toSet().size == codes.size) { "tokenizer fails the code check" }
            val calibration = Calibration(File(dir, "CALIBRATION.json").readText())
            val b = meta.getJSONArray("buckets")
            val buckets = (0 until b.length()).map { b.getInt(it) }.sorted()
            val model = CompiledModel.create(File(dir, meta.getString("file")).path, CompiledModel.Options(accelerator))
            return Engine(model, accelerator, tokenizer, calibration, buckets, meta.optInt("pad_id", 2))
        }
    }
}
