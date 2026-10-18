package io.github.kgonia.basal

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Test
import java.io.File

/** Prompts and token ids against the engine's format2_vectors.jsonl and tokenizer_cases.jsonl. */
class Format2Test {
    companion object {
        private var tok: Tokenizer? = null

        @BeforeClass
        @JvmStatic
        fun loadTokenizer() {
            val f = File(System.getProperty("basal.tokenizer") ?: "")
            if (f.isFile) tok = Tokenizer.load(f.readText())
        }

        private fun lines(name: String) =
            Format2Test::class.java.getResourceAsStream("/$name")!!.bufferedReader().readLines().filter { it.isNotBlank() }

        /** A JSON value as text, written like Python's json.dumps(value, ensure_ascii=False) (FORMAT2 3.2). */
        fun text(e: JsonElement): String = if (e.isJsonPrimitive && e.asJsonPrimitive.isString) e.asString else dumps(e)

        private fun dumps(e: JsonElement): String = when {
            e.isJsonNull -> "null"
            e.isJsonObject -> e.asJsonObject.entrySet().joinToString(", ", "{", "}") { (k, v) -> quote(k) + ": " + dumps(v) }
            e.isJsonArray -> e.asJsonArray.joinToString(", ", "[", "]") { dumps(it) }
            e.asJsonPrimitive.isString -> quote(e.asString)
            else -> e.toString()
        }

        private fun quote(s: String) = buildString {
            append('"')
            for (c in s) when (c) {
                '"' -> append("\\\""); '\\' -> append("\\\\"); '\n' -> append("\\n"); '\r' -> append("\\r"); '\t' -> append("\\t")
                else -> if (c < ' ') append(String.format("\\u%04x", c.code)) else append(c)
            }
            append('"')
        }

        private fun strings(a: JsonArray) = a.map { text(it) }

        /** The request's question in the app's model; null for the types the app does not offer. */
        fun question(q: JsonObject): Format2.Question? {
            val type = q.get("type")?.asString ?: "choice"
            val instructions = q.get("instructions")?.let { text(it) } ?: ""
            val crit = q.get("criteria") ?: q.get("levels")
            val hide = q.get("option_keys")?.asString == "hide"
            fun keyed(t: Format2.Type, o: JsonObject): Format2.Question {
                val keys = o.keySet().toList()
                return Format2.Question(t, instructions, keys, keys.map { k -> o.get(k).takeUnless { it.isJsonNull }?.let { text(it) } }, hide)
            }
            return when (type) {
                "noul" -> {
                    val c = crit?.takeIf { it.isJsonObject }?.asJsonObject
                    fun opt(k: String) = c?.get(k)?.takeUnless { it.isJsonNull }?.let { text(it) }?.ifEmpty { null }
                    Format2.Question(Format2.Type.NOUL, instructions, listOf("true", "false"), listOf(opt("true"), opt("false")))
                }
                "choice" -> if (crit!!.isJsonArray) Format2.Question(Format2.Type.CHOICE, instructions, strings(crit.asJsonArray))
                            else keyed(Format2.Type.CHOICE, crit.asJsonObject)
                "score" -> if (crit!!.isJsonArray) Format2.Question.scoreLevels(instructions, strings(crit.asJsonArray))
                           else keyed(Format2.Type.SCORE, crit.asJsonObject)
                "multi" -> if (crit!!.isJsonArray) Format2.Question(Format2.Type.MULTI, instructions, strings(crit.asJsonArray))
                           else keyed(Format2.Type.MULTI, crit.asJsonObject)
                else -> null
            }
        }

        private fun ints(a: JsonArray) = IntArray(a.size()) { a[it].asInt }
    }

    @Test
    fun promptsMatchVectors() {
        var n = 0
        for (line in lines("format2_vectors.jsonl")) {
            val v = JsonParser.parseString(line).asJsonObject
            val id = v.get("id").asString
            if (v.get("orders").asInt != 1) continue
            val req = v.getAsJsonObject("request")
            val qs = req.getAsJsonObject("questions")
            val questions = qs.keySet().map { question(qs.getAsJsonObject(it)) }
            if (questions.any { it == null }) continue
            val got = questions.flatMap { Format2.prompts(text(req.get("state")), it!!) }
            val expected = v.getAsJsonArray("prompts")
            assertEquals(id, expected.size(), got.size)
            for (i in got.indices) {
                val e = expected[i].asJsonObject
                assertEquals(id, e.get("prompt").asString, got[i].text)
                assertEquals(id, strings(e.getAsJsonArray("options")), got[i].options)
                tok?.let { t ->
                    assertArrayEquals(id, ints(e.getAsJsonArray("input_ids")), t.encode(got[i].text))
                    val codes = ints(e.getAsJsonArray("code_ids"))
                    assertArrayEquals(id, codes, IntArray(codes.size) { t.tokenToId(Format2.CODES[it])!! })
                }
                n++
            }
        }
        assertEquals(42, n)
    }

    @Test
    fun tokenizerMatchesReference() {
        val t = tok
        assumeTrue("set -Dbasal.tokenizer to a basal tokenizer.json", t != null)
        for (line in lines("tokenizer_cases.jsonl")) {
            val c = JsonParser.parseString(line).asJsonObject
            assertArrayEquals(c.get("text").asString, ints(c.getAsJsonArray("ids")), t!!.encode(c.get("text").asString))
        }
    }

    @Test
    fun calibration() {
        val cal = Calibration("""{"temperature_per_prim": {"choice": 1.26, "noul": 1.8, "score": 1.9},
            "temperature_per_bucket": {"choice": {"2-10": 1.63, "11-26": 0.96, "27-255": 0.89}}}""")
        assertEquals(1.63, cal.temperature(Format2.Type.CHOICE, 4), 0.0)
        assertEquals(0.89, cal.temperature(Format2.Type.CHOICE, 59), 0.0)
        assertEquals(1.8, cal.temperature(Format2.Type.MULTI, 2), 0.0)
        // logits 2 and 0: softmax 0.8808; T = 2 halves the log-odds, giving 0.7311
        assertEquals(0.7311, Calibration.probabilities(floatArrayOf(2f, 0f), 2.0)[0], 1e-4)
    }
}
