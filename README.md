# live-translate

Live speech → English subtitles in your terminal, fully local. Point it at the mic (or a video file),
and every line people say shows up as a transcript plus an English translation a few seconds later.
No cloud, no API keys, and no audio is ever written to disk.

```
22:43:03 [French] Tu peux parler en français avec moi
  → You can speak French with me.
22:37:01 [Chinese] 你好，你姓什么？我姓高。高小姐你好，请问你今年芳龄？
  → Hello, what's your surname? I'm Gao. Hello, Ms. Gao. May I ask how old you are?
```

## How it works

```
mic ─► Silero VAD ─► speech recognition ─► Qwen via Ollama ─► English
       (cuts utterances)  SenseVoice-Small or Qwen3-ASR-0.6B     (skips lines already in English)
```

| Recognizer | Languages | Speed on a laptop CPU (Core Ultra 7 155H) |
|---|---|---|
| `--asr sensevoice` (default) | Mandarin, Cantonese, English, Japanese, Korean | ~20× real time |
| `--asr qwen` | 30 languages incl. most European and Asian ones + Chinese dialects; more accurate | ~1.6× real time |

Translation uses `qwen3.5:9b` from an Ollama on `127.0.0.1:11434` if one is running (best on a GPU),
otherwise `start.sh` runs a CPU-only Ollama container on `:11435` with `qwen3.5:4b` (~2.5 s per line).
Earlier lines are passed as context, so fragments and pronouns translate sensibly.

## Install

Needs Python 3.12+, `ffmpeg`, PortAudio (for `sounddevice`), and Docker (for the CPU translator).

```bash
git clone https://github.com/yaghofgm/live-translate ~/Work/live-translate
cd ~/Work/live-translate
python -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install --no-deps qwen-asr==0.0.6   # skips its gradio/flask demo extras
```

Models download on first use into `models/` (SenseVoice ~0.9 GB, Qwen3-ASR ~1.8 GB).

## Use

```bash
./start.sh                   # mic → transcript (dim) → English (cyan). Ctrl-C to stop
./start.sh --asr qwen        # Qwen3-ASR: more languages, more accurate, slower
./start.sh --list-devices    # pick another mic with --device N
./start.sh --vad 0.7         # noisy place: be pickier about what counts as speech
./start.sh --lang French     # force a language instead of auto-detecting
./start.sh --no-translate    # transcript only, no Ollama needed
./start.sh --file clip.mp4   # run on a file / downloaded video instead of the mic
```

## Omarchy bar button (optional)

On [Omarchy](https://omarchy.org), `omarchy/` has a one-click bar toggle: it opens a floating terminal
running the translator, lights up red while listening, and stops it on the next click.

```bash
mkdir -p ~/.config/omarchy/bar/scripts
cp omarchy/live-translate-* ~/.config/omarchy/bar/scripts/
```

Then add this to `bar.layout.right` in `~/.config/omarchy/shell.json`
(set `LIVE_TRANSLATE_DIR` if you cloned somewhere other than `~/Work/live-translate`):

```json
{ "id": "live-translate", "type": "command",
  "exec": "~/.config/omarchy/bar/scripts/live-translate-status", "interval": 2,
  "tooltip": "Live translate", "onClick": "~/.config/omarchy/bar/scripts/live-translate-toggle" }
```

## Remove everything

```bash
docker rm -f live-translate-ollama && docker volume rm live-translate-ollama
rm -rf ~/Work/live-translate
```

## License

MIT
