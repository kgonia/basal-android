# LiteRT calls into its Kotlin classes from native code.
-keep class com.google.ai.edge.litert.** { *; }
# WorkManager (from LiteRT's NPU runtime, through Play asset delivery) creates its Room database by reflection: R8's
# full mode drops the constructor, and the app crashes at startup.
-keep class * extends androidx.room.RoomDatabase { <init>(); }
