package io.github.kgonia.basal

import org.json.JSONObject
import java.util.PriorityQueue

/**
 * The basal tokenizer (Bielik's APT4 tokenizer.json): added tokens split first, then a Metaspace pre-tokenizer
 * (prepend scheme "first", no split) and BPE with byte fallback. Only what basal's tokenizer.json uses is supported;
 * [Tokenizer.load] refuses anything else rather than tokenizing differently from the reference.
 */
class Tokenizer private constructor(
    private val vocab: Map<String, Int>,
    private val mergeRank: Map<Long, Int>,
    private val mergeResult: Map<Long, Int>,
    private val added: List<Pair<String, Int>>,
) {
    fun tokenToId(token: String): Int? = vocab[token] ?: added.firstOrNull { it.first == token }?.second

    /** Token ids of [text], without special tokens added (`<s>` in the text becomes its id). */
    fun encode(text: String): IntArray {
        val out = ArrayList<Int>(text.length / 3 + 8)
        var start = 0
        var i = 0
        while (i < text.length) {
            val hit = added.firstOrNull { text.startsWith(it.first, i) }
            if (hit == null) {
                i++
                continue
            }
            if (i > start) encodeSegment(text.substring(start, i), start == 0, out)
            out.add(hit.second)
            i += hit.first.length
            start = i
        }
        if (start < text.length) encodeSegment(text.substring(start), start == 0, out)
        return out.toIntArray()
    }

    private fun encodeSegment(segment: String, atStart: Boolean, out: MutableList<Int>) {
        var s = segment.replace(' ', META)
        if (atStart && !s.startsWith(META)) s = META + s
        // symbols: one per code point, or its UTF-8 bytes when the vocabulary lacks it (byte fallback)
        val ids = ArrayList<Int>(s.length)
        var i = 0
        while (i < s.length) {
            val cp = s.codePointAt(i)
            val ch = String(Character.toChars(cp))
            val id = vocab[ch]
            if (id != null) ids.add(id) else for (b in ch.toByteArray(Charsets.UTF_8)) {
                ids.add(vocab[String.format("<0x%02X>", b.toInt() and 0xff)] ?: error("no byte token"))
            }
            i += Character.charCount(cp)
        }
        out.addAll(bpe(ids))
    }

    /** Merges the lowest-ranked adjacent pair first, the leftmost one on ties, as tokenizers' BPE does. */
    private fun bpe(ids: List<Int>): List<Int> {
        val n = ids.size
        if (n < 2) return ids
        val sym = ids.toIntArray()
        val prev = IntArray(n) { it - 1 }
        val next = IntArray(n) { if (it + 1 < n) it + 1 else -1 }
        val alive = BooleanArray(n) { true }
        // entries: rank, left position, the two ids at queue time (to drop stale entries)
        val queue = PriorityQueue<LongArray>(compareBy<LongArray>({ it[0] }, { it[1] }))
        fun push(left: Int) {
            val right = next[left]
            if (right < 0) return
            val r = mergeRank[key(sym[left], sym[right])] ?: return
            queue.add(longArrayOf(r.toLong(), left.toLong(), sym[left].toLong(), sym[right].toLong()))
        }
        for (p in 0 until n - 1) push(p)
        while (queue.isNotEmpty()) {
            val e = queue.poll()!!
            val left = e[1].toInt()
            val right = next.getOrElse(left) { -1 }
            if (!alive[left] || right < 0 || sym[left].toLong() != e[2] || sym[right].toLong() != e[3]) continue
            sym[left] = mergeResult[key(sym[left], sym[right])]!!
            alive[right] = false
            next[left] = next[right]
            if (next[right] >= 0) prev[next[right]] = left
            if (prev[left] >= 0) push(prev[left])
            push(left)
        }
        val out = ArrayList<Int>()
        var p = 0
        while (p >= 0) {
            out.add(sym[p])
            p = next[p]
        }
        return out
    }

    companion object {
        private const val META = '▁'
        private fun key(a: Int, b: Int) = (a.toLong() shl 32) or (b.toLong() and 0xffffffffL)

        fun load(json: String): Tokenizer {
            val root = JSONObject(json)
            val model = root.getJSONObject("model")
            require(model.getString("type") == "BPE" && model.optBoolean("byte_fallback")) { "not a byte-fallback BPE" }
            require(root.isNull("normalizer")) { "normalizers are not supported" }
            val pre = root.getJSONObject("pre_tokenizer")
            require(pre.getString("type") == "Metaspace" && pre.optString("prepend_scheme") == "first" &&
                !pre.optBoolean("split", true) && pre.optString("replacement") == META.toString()) {
                "unsupported pre-tokenizer $pre"
            }
            val v = model.getJSONObject("vocab")
            val vocab = HashMap<String, Int>(v.length() * 2)
            for (k in v.keys()) vocab[k] = v.getInt(k)
            val merges = model.getJSONArray("merges")
            val rank = HashMap<Long, Int>(merges.length() * 2)
            val result = HashMap<Long, Int>(merges.length() * 2)
            for (r in 0 until merges.length()) {
                val m = merges.get(r)
                val (a, b) = if (m is String) m.split(' ', limit = 2).let { it[0] to it[1] }
                else (m as org.json.JSONArray).let { it.getString(0) to it.getString(1) }
                val ka = vocab[a] ?: continue
                val kb = vocab[b] ?: continue
                val merged = vocab[a + b] ?: continue
                val k = key(ka, kb)
                if (k !in rank) {
                    rank[k] = r
                    result[k] = merged
                }
            }
            val addedArr = root.getJSONArray("added_tokens")
            val added = (0 until addedArr.length()).map { addedArr.getJSONObject(it) }
                .map { it.getString("content") to it.getInt("id") }
                .sortedByDescending { it.first.length }
            return Tokenizer(vocab, rank, result, added)
        }
    }
}
