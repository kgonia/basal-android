package io.github.kgonia.basal

/**
 * Prefilled requests, from the basal-2.0 model card and in its style, one per answer type in the order of the type row
 * (and one in English). The chip shows the type and [topic]. Options are "key: description" lines.
 */
data class Example(val topic: String, val state: String, val type: Format2.Type, val instructions: String, val options: String)

private const val LOGIN = "Klient: od wczoraj nie mogę zalogować się do bankowości internetowej, system pokazuje błąd hasła."
private const val LAPTOP = "Klient: Zamówiłem laptopa trzy tygodnie temu, przyszedł z pękniętym ekranem. Kurier zostawił paczkę " +
    "pod drzwiami. Chcę zwrotu pieniędzy, nie naprawy."

val EXAMPLES = listOf(
    Example("dział", LOGIN, Format2.Type.CHOICE, "Do którego działu skierować zgłoszenie?",
        "cards: Reklamacje kart\nonline: Wsparcie bankowości elektronicznej\nloans: Kredyty"),
    Example("dostęp", LOGIN, Format2.Type.NOUL, "Czy klient nie może korzystać z usługi?", ""),
    Example("pilność", LAPTOP, Format2.Type.SCORE, "Jak pilne jest zgłoszenie?", "niska\nśrednia\nwysoka\nkrytyczna"),
    Example("tematy", LAPTOP, Format2.Type.MULTI, "Których tematów dotyczy wiadomość?",
        "damage: uszkodzony towar\nrefund: zwrot pieniędzy\nrepair: naprawa\ninvoice: faktura"),
    Example("English", "Customer: I was charged twice for my March subscription. Please fix it today, I need the money back.",
        Format2.Type.CHOICE, "What does the customer want?",
        "refund: A refund of the duplicate charge\ncancel: To cancel the subscription\ninfo: Information about the plan"),
)

/** Parses the options field: one option per line, "key: description" or the option text alone. */
fun parseOptions(text: String): Pair<List<String>, List<String?>> {
    val lines = text.lines().map { it.trim() }.filter { it.isNotEmpty() }
    val keyed = Regex("^([A-Za-z0-9_.-]{1,40}):\\s+(.+)$")
    val pairs = lines.map { l -> keyed.find(l)?.let { it.groupValues[1] to it.groupValues[2] } ?: (l to null) }
    return pairs.map { it.first } to pairs.map { it.second }
}

fun buildQuestion(type: Format2.Type, instructions: String, options: String, hideKeys: Boolean): Format2.Question {
    val (keys, descs) = parseOptions(options)
    return when (type) {
        Format2.Type.NOUL -> Format2.Question(type, instructions, listOf("true", "false"))
        Format2.Type.SCORE -> Format2.Question.scoreLevels(instructions, options.lines().map { it.trim() }.filter { it.isNotEmpty() })
        else -> {
            require(keys.toSet().size == keys.size) { "duplicate option keys" }
            Format2.Question(type, instructions, keys, descs, hideKeys)
        }
    }
}
