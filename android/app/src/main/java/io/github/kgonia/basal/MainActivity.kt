package io.github.kgonia.basal

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.Checkbox
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.google.ai.edge.litert.Accelerator
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

/** The loaded engine outlives the activity (rotation), like a model held by a process. */
private object Session {
    var engine: Engine? = null
    var engineModel: String? = null
    var downloads = HashMap<String, List<Long>>()
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MaterialTheme(colorScheme = if (isSystemInDarkTheme()) darkColorScheme() else lightColorScheme()) {
                Surface(Modifier.fillMaxSize()) { App() }
            }
        }
    }
}

@Composable
private fun App() {
    val context = LocalContext.current
    val store = remember { ModelStore(context) }
    val scope = rememberCoroutineScope()
    var modelId by rememberSaveable { mutableStateOf(CATALOG[0].id) }
    val model = CATALOG.first { it.id == modelId }
    var useGpu by rememberSaveable { mutableStateOf(true) }
    var ready by remember { mutableStateOf(CATALOG.associate { it.id to store.isReady(it) }) }
    var progress by remember { mutableStateOf<ModelStore.Progress?>(null) }
    var status by remember { mutableStateOf("") }
    var busy by remember { mutableStateOf(false) }
    var askLicence by remember { mutableStateOf(false) }
    var showAbout by remember { mutableStateOf(false) }

    val ex = EXAMPLES[0]
    var state by rememberSaveable { mutableStateOf(ex.state) }
    var type by rememberSaveable { mutableStateOf(ex.type) }
    var instructions by rememberSaveable { mutableStateOf(ex.instructions) }
    var options by rememberSaveable { mutableStateOf(ex.options) }
    var hideKeys by rememberSaveable { mutableStateOf(true) }
    var answer by remember { mutableStateOf<Engine.Answer?>(null) }

    // follows a running download of the selected model, also after the activity was recreated
    LaunchedEffect(modelId) {
        while (true) {
            val ids = Session.downloads[modelId]
            if (ids != null) {
                val p = withContext(Dispatchers.IO) { store.progress(ids) }
                progress = p
                if (p.failed) {
                    Session.downloads.remove(modelId)
                    status = context.getString(R.string.download_failed)
                } else if (p.finished) {
                    Session.downloads.remove(modelId)
                    status = try {
                        withContext(Dispatchers.IO) { store.finish(model) }
                        ready = ready + (modelId to true)
                        ""
                    } catch (e: Exception) {
                        e.message ?: e.toString()
                    }
                    progress = null
                }
            }
            delay(700)
        }
    }

    fun load(onLoaded: () -> Unit = {}) {
        if (Session.engineModel == "$modelId/$useGpu") return onLoaded()
        busy = true
        status = context.getString(R.string.loading)
        scope.launch {
            val accel = if (useGpu) Accelerator.GPU else Accelerator.CPU
            val result = withContext(Dispatchers.Default) {
                Session.engine?.close()
                Session.engine = null
                Session.engineModel = null
                val t0 = System.nanoTime()
                runCatching { Engine.open(store.dir(model), accel) }.map { it to (System.nanoTime() - t0) / 1_000_000 }
            }
            result.onSuccess { (e, ms) ->
                Session.engine = e
                Session.engineModel = "$modelId/$useGpu"
                status = context.getString(R.string.loaded, if (useGpu) "GPU" else "CPU", ms / 1000.0)
                busy = false
                onLoaded()
            }.onFailure {
                status = context.getString(R.string.load_failed, it.message ?: it.toString())
                busy = false
            }
        }
    }

    fun ask() {
        val q = try {
            buildQuestion(type, instructions.trim(), options, hideKeys)
        } catch (e: Exception) {
            status = e.message ?: e.toString()
            return
        }
        load {
            busy = true
            scope.launch {
                val r = withContext(Dispatchers.Default) { runCatching { Session.engine!!.answer(state.trim(), q) } }
                r.onSuccess { answer = it; status = "" }.onFailure { status = it.message ?: it.toString() }
                busy = false
            }
        }
    }

    Column(
        Modifier.safeDrawingPadding().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(stringResource(R.string.app_name), style = MaterialTheme.typography.headlineSmall, modifier = Modifier.weight(1f))
            TextButton(onClick = { showAbout = true }) { Text(stringResource(R.string.about)) }
        }
        Text(stringResource(R.string.tagline), style = MaterialTheme.typography.bodyMedium)

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            for (m in CATALOG) FilterChip(selected = m.id == modelId, onClick = { modelId = m.id; answer = null },
                label = { Text("${m.id} (${m.params})") })
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FilterChip(selected = useGpu, onClick = { useGpu = true }, label = { Text("GPU") })
            FilterChip(selected = !useGpu, onClick = { useGpu = false }, label = { Text("CPU") })
        }

        val dl = Session.downloads[modelId]
        when {
            dl != null -> {
                val p = progress
                val frac = if (p != null && p.total > 0) p.done.toFloat() / p.total else 0f
                Text(stringResource(R.string.downloading, (p?.done ?: 0) / 1e6, (p?.total ?: 0) / 1e6))
                LinearProgressIndicator(progress = { frac }, modifier = Modifier.fillMaxWidth())
                OutlinedButton(onClick = { store.cancel(dl); Session.downloads.remove(modelId); progress = null }) {
                    Text(stringResource(R.string.cancel))
                }
            }
            ready[modelId] != true -> Button(onClick = { askLicence = true }, enabled = !busy) {
                Text(stringResource(R.string.download, model.id))
            }
            else -> Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                Text(stringResource(R.string.model_ready), modifier = Modifier.weight(1f))
                OutlinedButton(onClick = {
                    if (Session.engineModel?.startsWith("$modelId/") == true) {
                        Session.engine?.close(); Session.engine = null; Session.engineModel = null
                    }
                    store.delete(model); ready = ready + (modelId to false); answer = null
                }, enabled = !busy) { Text(stringResource(R.string.delete)) }
            }
        }

        Text(stringResource(R.string.examples), fontWeight = FontWeight.SemiBold)
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            EXAMPLES.take(3).forEach { e -> ExampleChip(e) { state = e.state; type = e.type; instructions = e.instructions; options = e.options; answer = null } }
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            EXAMPLES.drop(3).forEach { e -> ExampleChip(e) { state = e.state; type = e.type; instructions = e.instructions; options = e.options; answer = null } }
        }

        OutlinedTextField(state, { state = it }, Modifier.fillMaxWidth(), label = { Text(stringResource(R.string.state)) }, minLines = 3)
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            for ((t, label) in listOf(Format2.Type.CHOICE to R.string.type_choice, Format2.Type.NOUL to R.string.type_noul,
                                      Format2.Type.SCORE to R.string.type_score, Format2.Type.MULTI to R.string.type_multi)) {
                FilterChip(selected = type == t, onClick = { type = t; answer = null }, label = { Text(stringResource(label)) })
            }
        }
        OutlinedTextField(instructions, { instructions = it }, Modifier.fillMaxWidth(), label = { Text(stringResource(R.string.question)) })
        if (type != Format2.Type.NOUL) {
            val hint = if (type == Format2.Type.SCORE) R.string.levels_hint else R.string.options_hint
            OutlinedTextField(options, { options = it }, Modifier.fillMaxWidth(), label = { Text(stringResource(hint)) }, minLines = 3)
        }
        if (type == Format2.Type.CHOICE) Row(verticalAlignment = Alignment.CenterVertically) {
            Checkbox(hideKeys, { hideKeys = it })
            Text(stringResource(R.string.hide_keys), style = MaterialTheme.typography.bodySmall)
        }
        Button(onClick = { ask() }, enabled = !busy && ready[modelId] == true, modifier = Modifier.fillMaxWidth()) {
            Text(stringResource(R.string.ask))
        }
        if (busy) LinearProgressIndicator(Modifier.fillMaxWidth())
        if (status.isNotEmpty()) SelectionContainer { Text(status, style = MaterialTheme.typography.bodySmall) }
        answer?.let { AnswerCard(it) }
        Spacer(Modifier.height(24.dp))
    }

    if (askLicence) AlertDialog(
        onDismissRequest = { askLicence = false },
        title = { Text(stringResource(R.string.licence_title, model.id)) },
        text = { Text(stringResource(R.string.licence_text, model.source, model.repo)) },
        confirmButton = {
            TextButton(onClick = {
                askLicence = false
                busy = true
                status = context.getString(R.string.checking)
                scope.launch {
                    val r = withContext(Dispatchers.IO) { runCatching { store.startDownload(model, store.fetchManifest(model)) } }
                    r.onSuccess { Session.downloads[modelId] = it; status = "" }.onFailure { status = it.message ?: it.toString() }
                    busy = false
                }
            }) { Text(stringResource(R.string.accept_download)) }
        },
        dismissButton = { TextButton(onClick = { askLicence = false }) { Text(stringResource(R.string.cancel)) } },
    )
    if (showAbout) AlertDialog(
        onDismissRequest = { showAbout = false },
        title = { Text(stringResource(R.string.about)) },
        text = { Box(Modifier.verticalScroll(rememberScrollState())) { Text(stringResource(R.string.about_text, BuildConfig.VERSION_NAME)) } },
        confirmButton = { TextButton(onClick = { showAbout = false }) { Text("OK") } },
    )
}

@Composable
private fun ExampleChip(e: Example, onClick: () -> Unit) {
    OutlinedButton(onClick = onClick) { Text(e.title, style = MaterialTheme.typography.labelMedium) }
}

@Composable
private fun AnswerCard(a: Engine.Answer) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            val headline = when (a.question.type) {
                Format2.Type.MULTI -> stringResource(R.string.answer_multi, a.selected.joinToString().ifEmpty { "—" })
                Format2.Type.SCORE -> stringResource(R.string.answer_score, a.score, a.options[a.best])
                Format2.Type.NOUL -> stringResource(R.string.answer_noul, pct(a.probabilities[0]))
                Format2.Type.CHOICE -> stringResource(R.string.answer_choice, a.keys[a.best], pct(a.confidence))
            }
            Text(headline, style = MaterialTheme.typography.titleMedium)
            a.probabilities.forEachIndexed { i, p ->
                val name = when (a.question.type) {
                    Format2.Type.NOUL -> a.options[i]
                    Format2.Type.MULTI -> a.options[i]
                    else -> a.question.descriptions.getOrNull(i)?.takeIf { a.question.levels }?.let { it } ?: a.keys[i]
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(name, Modifier.width(120.dp), style = MaterialTheme.typography.bodySmall, maxLines = 2)
                    LinearProgressIndicator(progress = { p.toFloat() }, modifier = Modifier.weight(1f).height(10.dp))
                    Text(pct(p), Modifier.width(56.dp).padding(start = 6.dp), style = MaterialTheme.typography.bodySmall)
                }
            }
            Text(stringResource(R.string.timing, a.millis, a.tokens), style = MaterialTheme.typography.bodySmall)
        }
    }
}

private fun pct(p: Double) = String.format(Locale.ROOT, "%.1f%%", p * 100)
