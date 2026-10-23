#!/bin/bash
# Copies the Qualcomm QNN runtime that the NPU variants need into app/src/main/jniLibs, and Qualcomm's notice file
# into the app's assets, from the ai-edge-litert-sdk-qualcomm Python package (QAIRT 2.50, the one LiteRT 2.3.0's AOT
# compiler uses). Linux or WSL:
#   android/fetch-qnn-libs.sh [python with ai-edge-litert-sdk-qualcomm==2.3.0 installed]
# Without a python argument it installs the package into a temporary venv (uv), which downloads the SDK.
#
# The libraries are Qualcomm's, under the AI Stack License (LICENSE.pdf in the package): they may be distributed only
# in object code as part of an app, with their notice file. This repository does not hold them.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
py=${1:-}
if [ -z "$py" ]; then
    venv=$(mktemp -d)
    uv venv -q "$venv"
    uv pip install -q --python "$venv/bin/python" ai-edge-litert-sdk-qualcomm==2.3.0
    py=$venv/bin/python
fi
sdk=$("$py" -c "import ai_edge_litert_sdk_qualcomm as s; print(s.get_sdk_path())")
jni=$here/app/src/main/jniLibs/arm64-v8a
mkdir -p "$jni" "$here/app/src/main/assets"
cp "$sdk"/lib/aarch64-android/{libQnnHtp.so,libQnnSystem.so,libQnnHtpV73Stub.so,libQnnHtpV75Stub.so,libQnnHtpV79Stub.so} "$jni/"
# the Hexagon DSP side: V73 is the Snapdragon 8 Gen 2, V75 the 8 Gen 3, V79 the 8 Elite (the SM8550 files may run there)
for v in 73 75 79; do cp "$sdk/lib/hexagon-v$v/unsigned/libQnnHtpV${v}Skel.so" "$jni/"; done
cp "$sdk/QNN_NOTICE.txt" "$here/app/src/main/assets/QNN_NOTICE.txt"
ls -la "$jni"
