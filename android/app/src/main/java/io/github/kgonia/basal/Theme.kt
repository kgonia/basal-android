package io.github.kgonia.basal

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp

private val Light = lightColorScheme(
    primary = Color(0xFF0A7D5A), onPrimary = Color.White,
    primaryContainer = Color(0xFFB8E6D2), onPrimaryContainer = Color(0xFF00210F),
    secondaryContainer = Color(0xFFD3E8DD), onSecondaryContainer = Color(0xFF0E1F17),
    surface = Color(0xFFF7FAF7), surfaceContainer = Color(0xFFECF1ED),
    surfaceContainerHigh = Color(0xFFE6ECE7), surfaceContainerHighest = Color(0xFFE0E6E1),
)
private val Dark = darkColorScheme(
    primary = Color(0xFF8CD6B3), onPrimary = Color(0xFF00382A),
    primaryContainer = Color(0xFF00513D), onPrimaryContainer = Color(0xFFB8E6D2),
    secondaryContainer = Color(0xFF354B41), onSecondaryContainer = Color(0xFFD3E8DD),
    surface = Color(0xFF0F1512), surfaceContainer = Color(0xFF1B211E),
    surfaceContainerHigh = Color(0xFF252B28), surfaceContainerHighest = Color(0xFF303633),
)

/** Material 3 with the wallpaper colours on Android 12+, the app's green otherwise. */
@Composable
fun BasalTheme(content: @Composable () -> Unit) {
    val dark = isSystemInDarkTheme()
    val context = LocalContext.current
    val colors = when {
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> if (dark) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        dark -> Dark
        else -> Light
    }
    MaterialTheme(
        colorScheme = colors,
        shapes = Shapes(
            small = RoundedCornerShape(12.dp),
            medium = RoundedCornerShape(20.dp),
            large = RoundedCornerShape(28.dp),
        ),
        content = content,
    )
}
