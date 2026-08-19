#!/usr/bin/env bash
# [项目注释] 文件职责：启动训练用户模拟器服务。
set -euo pipefail

# UserBench simulation is now an external DeepSeek-compatible API, not a local
# vLLM service. This compatibility entry point only validates the GRPO contract.
: "${GRPO_USER_SIM_BASE_URL:?set GRPO_USER_SIM_BASE_URL}"
: "${GRPO_USER_SIM_API_KEY:?set GRPO_USER_SIM_API_KEY}"
: "${GRPO_USER_SIM_MODEL:=deepseek-v4-flash}"

if [[ "${GRPO_USER_SIM_MODEL,,}" != "deepseek-v4-flash" ]]; then
  echo "GRPO_USER_SIM_MODEL must be deepseek-v4-flash" >&2
  exit 2
fi

echo "GRPO UserBench simulator is configured as an external API; no local server was started."
