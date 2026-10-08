#!/usr/bin/env bash
# Launch live translation. Uses an Ollama on :11434 with qwen3.5:9b if one is up (e.g. on a GPU),
# otherwise starts a CPU-only Ollama container on :11435 with qwen3.5:4b (created on first run).
cd "$(dirname "$0")"
if ! curl -sf -m1 127.0.0.1:11434/api/tags >/dev/null; then
  if ! docker start live-translate-ollama >/dev/null 2>&1; then
    echo "First run: creating the CPU Ollama container and pulling qwen3.5:4b (~3 GB)…"
    docker run -d --name live-translate-ollama -p 127.0.0.1:11435:11434 \
      -v live-translate-ollama:/root/.ollama ollama/ollama >/dev/null || exit 1
    sleep 2
    docker exec live-translate-ollama ollama pull qwen3.5:4b || exit 1
  fi
  sleep 1
fi
exec .venv/bin/python live_translate.py "$@"
