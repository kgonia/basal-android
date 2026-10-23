package io.github.kgonia.basal

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Add
import androidx.compose.material.icons.outlined.CheckCircle
import androidx.compose.material.icons.outlined.Close
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.Download
import androidx.compose.material.icons.outlined.Info
import androidx.compose.material.icons.outlined.Lightbulb
import androidx.compose.material.icons.outlined.PlayArrow
import androidx.compose.material.icons.outlined.WifiOff
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LargeTopAppBar
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.listSaver
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshots.SnapshotStateList
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.input.nestedscroll.nestedScroll
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.util.Locale

/** The loaded engine outlives the activity (rotation), like a model held by a process. */
private object Session {
    var engine: Engine? = null
    var engineModel: String? = null
    var engineVariant: String? = null
    val downloads = HashMap<String, List<Long>>()
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent { BasalTheme { App() } }
    }
}

private val optionsSaver = listSaver<SnapshotStateList<String>, String>({ it.toList() }, { it.toMutableStateList() })

private fun List<String>.toMutableStateList() = mutableStateListOf<String>().also { it.addAll(this) }

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun App() {
    val context = LocalContext.current
    val store = remember { ModelStore(context) }
    val scope = rememberCoroutineScope()
    val snackbar = remember { SnackbarHostState() }
    val scroll = TopAppBarDefaults.exitUntilCollapsedScrollBehavior()

    var modelId by rememberSaveable { mutableStateOf(CATALOG[0].id) }
    val model = CATALOG.first { it.id == modelId }
    // the variant each model runs, null while none is downloaded
    var ready by remember { mutableStateOf(CATALOG.associate { it.id to store.readyVariant(it) }) }
    var progress by remember { mutableStateOf<ModelStore.Progress?>(null) }
    var loadedInfo by remember { mutableStateOf<String?>(null) }
    var busy by remember { mutableStateOf(false) }
    var offer by remember { mutableStateOf<JSONObject?>(null) }
    // this phone's NPU variant of the selected model, if its litert.json (downloaded) has one
    val npuVariant = remember(modelId, ready) { store.manifest(model)?.let { Engine.npuVariant(it) } }
    var showAbout by remember { mutableStateOf(false) }

    val first = EXAMPLES[0]
    var state by rememberSaveable { mutableStateOf(first.state) }
    var type by rememberSaveable { mutableStateOf(first.type) }
    var instructions by rememberSaveable { mutableStateOf(first.instructions) }
    val options = rememberSaveable(saver = optionsSaver) { first.options.lines().toMutableStateList() }
    var hideKeys by rememberSaveable { mutableStateOf(true) }
    var answer by remember { mutableStateOf<Engine.Answer?>(null) }

    fun say(message: String) = scope.launch { snackbar.showSnackbar(message) }

    // follows a running download of the selected model, also after the activity was recreated
    LaunchedEffect(modelId) {
        while (true) {
            val ids = Session.downloads[modelId]
            if (ids != null) {
                val p = withContext(Dispatchers.IO) { store.progress(ids) }
                progress = p
                if (p.failed) {
                    Session.downloads.remove(modelId)
                    progress = null
                    say(context.getString(R.string.download_failed))
                } else if (p.finished) {
                    Session.downloads.remove(modelId)
                    progress = null
                    val v = withContext(Dispatchers.IO) { store.readyVariant(model) }
                    ready = ready + (modelId to v)
                    if (v == null) say(context.getString(R.string.download_failed))
                }
            }
            delay(700)
        }
    }

    fun load(onLoaded: () -> Unit) {
        val variant = ready[modelId] ?: return
        if (Session.engineModel == modelId && Session.engineVariant == variant) return onLoaded()
        busy = true
        scope.launch {
            val result = withContext(Dispatchers.Default) {
                Session.engine?.close()
                Session.engine = null
                Session.engineModel = null
                val t0 = System.nanoTime()
                runCatching {
                    val e = Engine.open(store.dir(model), variant, context.cacheDir, context.applicationInfo.nativeLibraryDir)
                    // an NPU file for a chip nobody tested may run and compute nonsense: compare with the reference
                    val failed = if (variant == ModelStore.CPU || store.npuPassed(model, variant)) null
                                 else runCatching { e.selfTest(store.manifest(model)!!) }.getOrElse { it.message ?: it.toString() }
                    if (failed != null) {
                        e.close()
                        store.markNpuFailed(model)
                        throw NpuFailed(failed)
                    }
                    if (variant != ModelStore.CPU) store.markNpuPassed(model, variant)
                    e
                }.map { it to (System.nanoTime() - t0) / 1_000_000 }
            }
            result.onSuccess { (e, ms) ->
                Session.engine = e
                Session.engineModel = modelId
                Session.engineVariant = variant
                loadedInfo = context.getString(R.string.loaded, ms / 1000.0,
                    context.getString(if (variant == ModelStore.CPU) R.string.on_cpu else R.string.on_npu))
                onLoaded()
            }.onFailure {
                busy = false
                if (it is NpuFailed || variant != ModelStore.CPU) {
                    // the next try runs the CPU variant if it is there; else the model card offers to download it
                    store.markNpuFailed(model)
                    ready = ready + (modelId to store.readyVariant(model))
                    say(context.getString(R.string.npu_failed, it.message ?: it.toString()))
                } else {
                    say(context.getString(R.string.load_failed, it.message ?: it.toString()))
                }
            }
        }
    }

    fun ask() {
        val q = try {
            buildQuestion(type, instructions.trim(), options.joinToString("\n"), hideKeys)
        } catch (e: Exception) {
            say(e.message ?: e.toString())
            return
        }
        load {
            busy = true
            scope.launch {
                withContext(Dispatchers.Default) { runCatching { Session.engine!!.answer(state.trim(), q) } }
                    .onSuccess { answer = it }
                    .onFailure { say(it.message ?: it.toString()) }
                busy = false
            }
        }
    }

    Scaffold(
        modifier = Modifier.nestedScroll(scroll.nestedScrollConnection),
        topBar = {
            LargeTopAppBar(
                title = {
                    Column {
                        Text(stringResource(R.string.app_name), maxLines = 1)
                        if (scroll.state.collapsedFraction < 0.5f) Text(
                            stringResource(R.string.subtitle), style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    }
                },
                actions = {
                    IconButton(onClick = { showAbout = true }) {
                        Icon(Icons.Outlined.Info, stringResource(R.string.about))
                    }
                },
                scrollBehavior = scroll,
            )
        },
        snackbarHost = { SnackbarHost(snackbar) },
        bottomBar = {
            Surface(tonalElevation = 3.dp) {
                Box(Modifier.navigationBarsPadding().imePadding().padding(horizontal = 16.dp, vertical = 12.dp)) {
                    Button(
                        onClick = { ask() },
                        enabled = !busy && ready[modelId] != null,
                        modifier = Modifier.fillMaxWidth().height(56.dp),
                        contentPadding = PaddingValues(horizontal = 24.dp),
                    ) {
                        if (busy) {
                            CircularProgressIndicator(Modifier.size(20.dp), strokeWidth = 2.dp,
                                color = MaterialTheme.colorScheme.onPrimary)
                            Spacer(Modifier.width(12.dp))
                            Text(stringResource(if (Session.engineModel == modelId) R.string.thinking else R.string.loading))
                        } else {
                            Icon(Icons.Outlined.PlayArrow, null)
                            Spacer(Modifier.width(8.dp))
                            Text(stringResource(if (ready[modelId] != null) R.string.ask else R.string.ask_needs_model),
                                style = MaterialTheme.typography.titleMedium)
                        }
                    }
                }
            }
        },
    ) { padding ->
        Column(
            Modifier.fillMaxSize().padding(padding).verticalScroll(rememberScrollState()).padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            ModelCard(
                modelId = modelId,
                onModel = { modelId = it; answer = null },
                ready = ready[modelId] != null,
                variant = ready[modelId],
                npuVariant = npuVariant,
                onVariant = { v ->
                    store.choose(model, v)
                    val meta = store.manifest(model)
                    if (meta != null && !store.complete(model, meta, v)) {
                        busy = true
                        scope.launch {
                            withContext(Dispatchers.IO) { runCatching { store.startDownload(model, meta, v) } }
                                .onSuccess { ids -> if (ids.isNotEmpty()) Session.downloads[modelId] = ids }
                                .onFailure { say(it.message ?: it.toString()) }
                            busy = false
                        }
                    } else {
                        ready = ready + (modelId to store.readyVariant(model))
                        answer = null
                    }
                },
                downloading = Session.downloads[modelId] != null,
                progress = progress,
                loadedInfo = loadedInfo.takeIf { Session.engineModel == modelId },
                onDownload = {
                    busy = true
                    scope.launch {
                        withContext(Dispatchers.IO) { runCatching { store.fetchManifest(model) } }
                            .onSuccess { offer = it }
                            .onFailure { say(it.message ?: it.toString()) }
                        busy = false
                    }
                },
                onCancel = { Session.downloads.remove(modelId)?.let { store.cancel(it) }; progress = null },
                onDelete = {
                    if (Session.engineModel == modelId) {
                        Session.engine?.close(); Session.engine = null; Session.engineModel = null; Session.engineVariant = null
                    }
                    store.delete(model)
                    ready = ready + (modelId to null)
                    answer = null
                },
                enabled = !busy,
            )

            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                SectionLabel(stringResource(R.string.examples))
                LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(EXAMPLES) { e ->
                        AssistChip(
                            onClick = {
                                state = e.state; type = e.type; instructions = e.instructions
                                options.clear(); options.addAll(e.options.lines().filter { it.isNotBlank() })
                                answer = null
                            },
                            label = { Text(e.title) },
                            leadingIcon = { Icon(Icons.Outlined.Lightbulb, null, Modifier.size(18.dp)) },
                        )
                    }
                }
            }

            QuestionCard(
                state = state, onState = { state = it },
                type = type, onType = { type = it; answer = null },
                instructions = instructions, onInstructions = { instructions = it },
                options = options, hideKeys = hideKeys, onHideKeys = { hideKeys = it },
            )

            AnimatedVisibility(visible = answer != null, enter = fadeIn() + expandVertically()) {
                answer?.let { AnswerCard(it) }
            }
            Spacer(Modifier.height(8.dp))
        }
    }

    offer?.let { meta ->
        DownloadDialog(model, meta, store, onDismiss = { offer = null }) { variant ->
            offer = null
            store.choose(model, variant)
            busy = true
            scope.launch {
                withContext(Dispatchers.IO) { runCatching { store.startDownload(model, meta, variant) } }
                    .onSuccess { ids ->
                        if (ids.isEmpty()) ready = ready + (modelId to store.readyVariant(model))
                        else Session.downloads[modelId] = ids
                    }
                    .onFailure { say(it.message ?: it.toString()) }
                busy = false
            }
        }
    }
    if (showAbout) ModalBottomSheet(onDismissRequest = { showAbout = false }, sheetState = rememberModalBottomSheetState()) {
        Column(Modifier.padding(horizontal = 24.dp).padding(bottom = 32.dp).verticalScroll(rememberScrollState()),
               verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(stringResource(R.string.about), style = MaterialTheme.typography.titleLarge)
            SelectionContainer {
                Text(stringResource(R.string.about_text, BuildConfig.VERSION_NAME), style = MaterialTheme.typography.bodyMedium)
            }
        }
    }
}

private class NpuFailed(message: String) : Exception(message)

/** The licence note and the choice of variant: this phone's NPU if there are files for its chip, or the CPU. */
@Composable
private fun DownloadDialog(model: CatalogModel, meta: JSONObject, store: ModelStore, onDismiss: () -> Unit,
                           onDownload: (String) -> Unit) {
    val npu = remember(meta) { Engine.npuVariant(meta)?.takeUnless { store.npuFailed(model) } }
    var variant by remember(meta) { mutableStateOf(npu ?: ModelStore.CPU) }
    AlertDialog(
        onDismissRequest = onDismiss,
        icon = { Icon(Icons.Outlined.Download, null) },
        title = { Text(stringResource(R.string.licence_title, model.id)) },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(stringResource(R.string.licence_text, model.source, model.repo))
                SectionLabel(stringResource(R.string.runs_on))
                listOfNotNull(npu, ModelStore.CPU).forEach { v ->
                    val tested = v == ModelStore.CPU || meta.getJSONObject("variants").getJSONObject(v).optBoolean("tested")
                    Row(Modifier.fillMaxWidth().selectable(selected = variant == v, onClick = { variant = v }),
                        verticalAlignment = Alignment.CenterVertically) {
                        RadioButton(selected = variant == v, onClick = null)
                        Spacer(Modifier.width(10.dp))
                        Column {
                            Text(if (v == ModelStore.CPU) stringResource(R.string.variant_cpu)
                                 else stringResource(R.string.variant_npu, Engine.soc ?: ""),
                                 style = MaterialTheme.typography.bodyMedium)
                            Text(stringResource(R.string.disk_needed, store.diskNeeded(meta, v) / 1e9) +
                                 (if (tested) "" else " · " + stringResource(R.string.experimental)),
                                 style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = { onDownload(variant) }) { Text(stringResource(R.string.accept_download)) } },
        dismissButton = { TextButton(onClick = onDismiss) { Text(stringResource(R.string.cancel)) } },
    )
}

@Composable
private fun SectionLabel(text: String) {
    Text(text, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
}

@Composable
private fun ModelCard(
    modelId: String, onModel: (String) -> Unit, ready: Boolean, variant: String?, npuVariant: String?,
    onVariant: (String) -> Unit, downloading: Boolean, progress: ModelStore.Progress?,
    loadedInfo: String?, onDownload: () -> Unit, onCancel: () -> Unit, onDelete: () -> Unit, enabled: Boolean,
) {
    ElevatedCard(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            SectionLabel(stringResource(R.string.model))
            SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
                CATALOG.forEachIndexed { i, m ->
                    SegmentedButton(
                        selected = m.id == modelId, onClick = { onModel(m.id) }, enabled = enabled,
                        shape = SegmentedButtonDefaults.itemShape(i, CATALOG.size),
                        label = { Text(m.label) },
                    )
                }
            }
            when {
                downloading -> Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    val p = progress
                    val frac = if (p != null && p.total > 0) p.done.toFloat() / p.total else 0f
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(stringResource(R.string.downloading, (p?.done ?: 0) / 1e6, (p?.total ?: 0) / 1e6),
                             style = MaterialTheme.typography.bodyMedium, modifier = Modifier.weight(1f))
                        IconButton(onClick = onCancel) { Icon(Icons.Outlined.Close, stringResource(R.string.cancel)) }
                    }
                    LinearProgressIndicator(progress = { frac }, modifier = Modifier.fillMaxWidth().height(6.dp).clip(CircleShape))
                }
                !ready -> Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text(stringResource(R.string.model_missing), style = MaterialTheme.typography.bodyMedium, modifier = Modifier.weight(1f))
                    FilledTonalButton(onClick = onDownload, enabled = enabled) {
                        Icon(Icons.Outlined.Download, null, Modifier.size(18.dp))
                        Spacer(Modifier.width(8.dp))
                        Text(stringResource(R.string.download_short))
                    }
                }
                else -> Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Outlined.CheckCircle, null, tint = MaterialTheme.colorScheme.primary)
                    Spacer(Modifier.width(10.dp))
                    Column(Modifier.weight(1f)) {
                        Text(stringResource(R.string.model_ready), style = MaterialTheme.typography.bodyMedium)
                        Text(loadedInfo ?: stringResource(R.string.offline), style = MaterialTheme.typography.bodySmall,
                             color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    IconButton(onClick = onDelete, enabled = enabled) { Icon(Icons.Outlined.Delete, stringResource(R.string.delete)) }
                }
            }
            // CPU or this phone's NPU, picked by hand; one not downloaded yet starts downloading
            if (ready && npuVariant != null && !downloading) {
                SectionLabel(stringResource(R.string.runs_on))
                val choices = listOf(npuVariant to stringResource(R.string.on_npu_chip, npuVariant.removePrefix("npu-")),
                                     ModelStore.CPU to stringResource(R.string.on_cpu))
                SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
                    choices.forEachIndexed { i, (v, label) ->
                        SegmentedButton(selected = variant == v, onClick = { if (variant != v) onVariant(v) }, enabled = enabled,
                            shape = SegmentedButtonDefaults.itemShape(i, choices.size), label = { Text(label) })
                    }
                }
            }
        }
    }
}

@Composable
private fun QuestionCard(
    state: String, onState: (String) -> Unit, type: Format2.Type, onType: (Format2.Type) -> Unit,
    instructions: String, onInstructions: (String) -> Unit,
    options: SnapshotStateList<String>, hideKeys: Boolean, onHideKeys: (Boolean) -> Unit,
) {
    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            OutlinedTextField(
                state, onState, Modifier.fillMaxWidth(), label = { Text(stringResource(R.string.state)) },
                minLines = 3, maxLines = 10, shape = MaterialTheme.shapes.medium,
                trailingIcon = if (state.isNotEmpty()) ({ IconButton(onClick = { onState("") }) { Icon(Icons.Outlined.Close, null) } }) else null,
            )
            SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
                val types = listOf(Format2.Type.CHOICE to R.string.type_choice, Format2.Type.NOUL to R.string.type_noul,
                                   Format2.Type.SCORE to R.string.type_score, Format2.Type.MULTI to R.string.type_multi)
                types.forEachIndexed { i, (t, label) ->
                    SegmentedButton(selected = type == t, onClick = { onType(t) },
                        shape = SegmentedButtonDefaults.itemShape(i, types.size), icon = {},
                        label = { Text(stringResource(label), maxLines = 1, style = MaterialTheme.typography.labelMedium) })
                }
            }
            OutlinedTextField(instructions, onInstructions, Modifier.fillMaxWidth(),
                label = { Text(stringResource(R.string.question)) }, shape = MaterialTheme.shapes.medium)
            if (type != Format2.Type.NOUL) {
                HorizontalDivider()
                SectionLabel(stringResource(if (type == Format2.Type.SCORE) R.string.levels else R.string.options))
                options.forEachIndexed { i, text ->
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        val badge = if (type == Format2.Type.MULTI) "${i + 1}" else Format2.CODES.getOrElse(i) { "?" }
                        Box(Modifier.size(32.dp).clip(CircleShape).background(MaterialTheme.colorScheme.secondaryContainer),
                            contentAlignment = Alignment.Center) {
                            Text(badge, style = MaterialTheme.typography.labelMedium, fontFamily = FontFamily.Monospace,
                                 color = MaterialTheme.colorScheme.onSecondaryContainer)
                        }
                        Spacer(Modifier.width(10.dp))
                        OutlinedTextField(text, { options[i] = it }, Modifier.weight(1f), singleLine = true,
                            shape = MaterialTheme.shapes.small,
                            placeholder = { Text(stringResource(if (type == Format2.Type.SCORE) R.string.level_hint else R.string.option_hint)) })
                        IconButton(onClick = { options.removeAt(i) }, enabled = options.size > 2) {
                            Icon(Icons.Outlined.Close, stringResource(R.string.delete))
                        }
                    }
                }
                TextButton(onClick = { options.add("") }, enabled = options.size < Format2.MAX_OPTIONS) {
                    Icon(Icons.Outlined.Add, null, Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text(stringResource(if (type == Format2.Type.SCORE) R.string.add_level else R.string.add_option))
                }
                if (type == Format2.Type.CHOICE) Row(verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(stringResource(R.string.hide_keys), style = MaterialTheme.typography.bodyMedium)
                        Text(stringResource(R.string.hide_keys_hint), style = MaterialTheme.typography.bodySmall,
                             color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    Switch(hideKeys, onHideKeys)
                }
            }
        }
    }
}

@Composable
private fun AnswerCard(a: Engine.Answer) {
    val scheme = MaterialTheme.colorScheme
    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = scheme.primaryContainer)) {
        Column(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(stringResource(R.string.decision), style = MaterialTheme.typography.labelLarge, color = scheme.onPrimaryContainer)
            val headline = when (a.question.type) {
                Format2.Type.MULTI -> a.selected.joinToString().ifEmpty { "—" }
                Format2.Type.SCORE -> String.format(Locale.getDefault(), "%.2f · %s", a.score, a.options[a.best])
                Format2.Type.NOUL -> stringResource(if (a.probabilities[0] >= 0.5) R.string.yes else R.string.no)
                Format2.Type.CHOICE -> a.question.descriptions.getOrNull(a.best) ?: a.keys[a.best]
            }
            Text(headline, style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold,
                 color = scheme.onPrimaryContainer)
            if (a.question.type != Format2.Type.MULTI) Text(
                stringResource(R.string.confidence, pct(a.confidence)), style = MaterialTheme.typography.bodyMedium,
                color = scheme.onPrimaryContainer)
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                a.probabilities.forEachIndexed { i, p ->
                    val name = when (a.question.type) {
                        Format2.Type.NOUL, Format2.Type.MULTI -> a.options[i]
                        else -> if (a.question.levels) a.options[i] else a.question.descriptions.getOrNull(i) ?: a.keys[i]
                    }
                    val top = if (a.question.type == Format2.Type.MULTI) p >= 0.5 else i == a.best
                    val shown by animateFloatAsState(p.toFloat(), label = "bar")
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Row {
                            Text(name, style = MaterialTheme.typography.bodyMedium, color = scheme.onPrimaryContainer,
                                 fontWeight = if (top) FontWeight.SemiBold else FontWeight.Normal,
                                 maxLines = 2, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
                            Text(pct(p), style = MaterialTheme.typography.bodyMedium, color = scheme.onPrimaryContainer,
                                 fontFamily = FontFamily.Monospace)
                        }
                        LinearProgressIndicator(
                            progress = { shown },
                            modifier = Modifier.fillMaxWidth().height(8.dp).clip(RoundedCornerShape(4.dp)),
                            color = if (top) scheme.primary else scheme.outline,
                            trackColor = scheme.surface.copy(alpha = 0.6f),
                            drawStopIndicator = {},
                        )
                    }
                }
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Outlined.WifiOff, null, Modifier.size(16.dp), tint = scheme.onPrimaryContainer)
                Spacer(Modifier.width(6.dp))
                Text(stringResource(R.string.timing, a.millis / 1000.0, a.tokens), style = MaterialTheme.typography.bodySmall,
                     color = scheme.onPrimaryContainer)
            }
        }
    }
}

private fun pct(p: Double) = String.format(Locale.getDefault(), "%.1f%%", p * 100)
