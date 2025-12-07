#!/bin/bash

WEIGHTS_DIR="weights"
mkdir ./${WEIGHTS_DIR}

# DNSMOS: основная модель SIG/BAK/OVRL
DNSMOS_SIG_BAK_OVR="https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
wget -nc "${DNSMOS_SIG_BAK_OVR}" -O "${WEIGHTS_DIR}/sig_bak_ovr.onnx"

# DNSMOS: вспомогательная модель P.808
DNSMOS_P808="https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/model_v8.onnx"
wget -nc "${DNSMOS_P808}" -O "${WEIGHTS_DIR}/model_v8.onnx"

echo "All weights was successfully downloaded to ${WEIGHTS_DIR}"