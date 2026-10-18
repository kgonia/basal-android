# basal-android

An unofficial Android app and LiteRT conversion for the [basal-2.0](https://github.com/rkinas/basal) typed-decision
models by Remigiusz Kinas: ask a question about a text, give the allowed answers, and get a calibrated probability for
each one, on the phone and offline.

- **Models on Hugging Face:** [SearchingMan/basal-2.0-mini-LiteRT](https://huggingface.co/SearchingMan/basal-2.0-mini-LiteRT)
  (1.5B) and [SearchingMan/basal-2.0-4.5B-LiteRT](https://huggingface.co/SearchingMan/basal-2.0-4.5B-LiteRT) (4.5B).
- **App:** the APK is attached to each [release](https://github.com/kgonia/basal-android/releases). It downloads the
  model you pick (Wi-Fi recommended) and runs it with Google LiteRT on the GPU or the CPU.

Not affiliated with or endorsed by the basal or Bielik authors. The models are theirs (Apache-2.0); this repository
only runs them.

<!-- RESULTS -->

## How it works

basal answers a question in one forward pass, without generating text: the prompt (prompt format 2, see the engine's
[FORMAT2.md](https://github.com/rkinas/basal/blob/v2.0.0/docs/FORMAT2.md)) ends with `{"answer": "`, and the softmax over
the next-token logits of the option codes (`A`, `B`, …) is the decision, then a temperature from the model's
`CALIBRATION.json`.

So the LiteRT file has no KV cache and no sampling. It is the Llama prefill with the head cut down to the 255 code
tokens: `tokens` (int32 `[1, L]`, right-padded) and `last` (the position to read) in, `logits` (float32 `[1, 255]`) out.
One file holds a signature per prompt length `p128` … `p2048`, sharing the weights; attention is causal, so the
padding after `last` cannot change the result.

| folder | what |
|---|---|
| `android/` | the app: `Tokenizer.kt` (the BPE tokenizer), `Format2.kt` (prompts and calibration), `Engine.kt` (LiteRT), `MainActivity.kt` (UI) |


## Build the app

JDK 21 and the Android SDK (compileSdk 36):

```bash
cd android
./gradlew testDebugUnitTest -Dbasal.tokenizer=/path/to/tokenizer.json   # tokenizer and prompts vs the reference
./gradlew assembleRelease
```

For testing without the download, push a model folder (the Hugging Face repo's files) to
`/sdcard/Android/data/io.github.kgonia.basal/files/models/basal-2.0-mini/`.

## Licences

This repository: Apache-2.0 ([LICENSE](LICENSE)). `android/app/src/test/resources/format2_vectors.jsonl` is from
[rkinas/basal](https://github.com/rkinas/basal) v2.0.0 (Apache-2.0). The models: basal-2.0 by Remigiusz Kinas,
Apache-2.0, fine-tuned from Bielik v3 (SpeakLeash, ACK Cyfronet AGH), Apache-2.0. The app uses LiteRT (Google),
Jetpack Compose and AndroidX, all Apache-2.0.
