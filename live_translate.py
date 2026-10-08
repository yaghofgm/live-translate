#!/usr/bin/env python3
"""Live Chinese -> English translation from the mic.

mic -> Silero VAD (cut into utterances) -> ASR (SenseVoice or Qwen3-ASR) -> Qwen via Ollama (English)
Nothing is written to disk; audio only lives in memory until it is transcribed.
"""
import argparse
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MODELSCOPE_CACHE", os.path.join(HERE, "models"))
os.environ.setdefault("HF_HOME", os.path.join(HERE, "models", "hf"))

import logging
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import requests
import torch

SR = 16000
BLOCK = 512  # Silero VAD wants 512-sample chunks at 16 kHz

DIM, BOLD, CYAN, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[0m"

OLLAMA_CANDIDATES = [
    ("http://127.0.0.1:11434", "qwen3.5:9b"),  # e.g. a GPU Ollama
    ("http://127.0.0.1:11435", "qwen3.5:4b"),  # CPU-only container (start.sh)
]

SYSTEM_PROMPT = (
    "You translate overheard speech into natural English. The detected language is given, "
    "but it may be wrong; trust the text. "
    "The text comes from speech recognition and may contain errors or fragments; give the most "
    "plausible meaning. Earlier lines are given only as context. "
    "Output only the English translation of the LAST line, nothing else."
)


def pick_ollama(url, model):
    if url:
        return url, model or "qwen3.5:4b"
    for u, m in OLLAMA_CANDIDATES:
        try:
            tags = requests.get(f"{u}/api/tags", timeout=1).json()
            if any(t["name"] == m for t in tags.get("models", [])):
                return u, model or m
        except requests.RequestException:
            pass
    return None, None


SENSEVOICE_LANGS = {"zh": "Chinese", "yue": "Cantonese", "en": "English", "ja": "Japanese", "ko": "Korean"}


def load_sensevoice(args):
    from funasr import AutoModel

    local = os.path.join(HERE, "models", "models", "iic--SenseVoiceSmall", "snapshots", "master")
    asr = AutoModel(model=local if os.path.isdir(local) else "iic/SenseVoiceSmall", device="cpu", disable_update=True, log_level="ERROR")

    def run(seg):
        res = asr.generate(input=seg, cache={}, language=args.lang, use_itn=True, disable_pbar=True)
        raw = res[0]["text"] if res else ""
        code = (re.match(r"<\|(\w+)\|>", raw) or [None, "?"])[1]
        return SENSEVOICE_LANGS.get(code, code), re.sub(r"<\|[^|]*\|>", "", raw).strip()
    return run


def load_qwen_asr(args):
    from qwen_asr import Qwen3ASRModel
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()  # hide per-line "pad_token_id" / generation-flag chatter

    asr = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-0.6B", dtype=torch.float32, device_map="cpu",
                                        max_inference_batch_size=1, max_new_tokens=256)
    force = None if args.lang == "auto" else args.lang

    def run(seg):
        res = asr.transcribe(audio=(seg, SR), language=force)
        return (force or res[0].language or "?"), res[0].text.strip()
    return run


def asr_worker(seg_q, tr_q, args):
    logging.disable(logging.INFO)
    transcribe = (load_qwen_asr if args.asr == "qwen" else load_sensevoice)(args)
    print(f"{DIM}[{args.asr} ready — listening]{RESET}", flush=True)
    while (seg := seg_q.get()) is not None:
        lang, text = transcribe(seg)
        if len(text) < 2:
            continue
        stamp = time.strftime("%H:%M:%S")
        print(f"{DIM}{stamp} [{lang}] {text}{RESET}", flush=True)
        if lang != "English" or args.translate_all:
            tr_q.put((lang, text))
    tr_q.put(None)


def translate_worker(tr_q, url, model):
    history = deque(maxlen=3)
    session = requests.Session()
    while (item := tr_q.get()) is not None:
        lang, text = item
        ctx = "\n".join(history)
        user = f"Detected language: {lang}\n\n" + ((f"Context:\n{ctx}\n\nLAST line:\n{text}") if ctx else text)
        try:
            r = session.post(f"{url}/api/chat", timeout=60, json={
                "model": model, "stream": False, "think": False, "keep_alive": "30m",
                "options": {"temperature": 0.2},
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": user}],
            })
            out = r.json()["message"]["content"].strip()
        except Exception as e:  # keep going on a hiccup
            out = f"(translation failed: {e})"
        history.append(text)
        print(f"  {BOLD}{CYAN}→ {out}{RESET}", flush=True)


def audio_blocks(args):
    """Yield float32 blocks of BLOCK samples, from the mic or (for testing) a file."""
    if args.file:
        pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", args.file, "-f", "f32le",
                              "-ac", "1", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
        audio = np.frombuffer(pcm, dtype=np.float32)
        audio = np.concatenate([audio, np.zeros(SR, dtype=np.float32)])  # trailing silence ends last segment
        for i in range(0, len(audio) - BLOCK + 1, BLOCK):
            yield audio[i:i + BLOCK]
        return

    import sounddevice as sd

    q = queue.Queue()
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=BLOCK,
                        device=args.device, callback=lambda d, *_: q.put(d[:, 0].copy())):
        while True:
            yield q.get()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", help="input device name/index (see --list-devices)")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--file", help="run on an audio/video file instead of the mic (testing)")
    p.add_argument("--asr", choices=["sensevoice", "qwen"], default="sensevoice",
                   help="speech recognizer: sensevoice (zh/yue/en/ja/ko, fastest) or qwen (Qwen3-ASR-0.6B, 30 languages)")
    p.add_argument("--lang", default="auto",
                   help="auto, or force a language (sensevoice: zh | yue | ja | ko; qwen: Chinese | Cantonese | French ...)")
    p.add_argument("--ollama", help="Ollama URL (default: auto-detect)")
    p.add_argument("--model", help="Ollama model (default: 9b on GPU stack, 4b on CPU)")
    p.add_argument("--no-translate", action="store_true", help="only show the transcript")
    p.add_argument("--translate-all", action="store_true", help="also 'translate' English lines")
    p.add_argument("--vad", type=float, default=0.5, help="speech threshold 0-1; raise in noisy places")
    p.add_argument("--max-seconds", type=float, default=12, help="force-cut long monologues")
    args = p.parse_args()
    if args.device and args.device.isdigit():
        args.device = int(args.device)

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    from silero_vad import VADIterator, load_silero_vad

    seg_q, tr_q = queue.Queue(), queue.Queue()
    threads = [threading.Thread(target=asr_worker, args=(seg_q, tr_q, args), daemon=True)]
    if not args.no_translate:
        url, model = pick_ollama(args.ollama, args.model)
        if not url:
            sys.exit("No Ollama with a Qwen model found. Run ./start.sh (creates a CPU Ollama) "
                     "or use --no-translate.")
        print(f"{DIM}[translator: {model} @ {url}]{RESET}", flush=True)
        threads.append(threading.Thread(target=translate_worker, args=(tr_q, url, model), daemon=True))
    for t in threads:
        t.start()

    if args.asr == "sensevoice":
        torch.set_num_threads(1)  # VAD is tiny; leave the cores to SenseVoice (Qwen3-ASR needs them all)
    vad = VADIterator(load_silero_vad(), threshold=args.vad, sampling_rate=SR,
                      min_silence_duration_ms=500, speech_pad_ms=200)
    preroll = deque(maxlen=10)  # ~320 ms before speech onset
    seg, in_speech = [], False
    try:
        for block in audio_blocks(args):
            ev = vad(torch.from_numpy(block))
            if ev and "start" in ev and not in_speech:
                in_speech, seg = True, list(preroll)
            if in_speech:
                seg.append(block)
                long = len(seg) * BLOCK > args.max_seconds * SR
                if (ev and "end" in ev) or long:
                    seg_q.put(np.concatenate(seg))
                    seg = []
                    if not long:
                        in_speech = False
            preroll.append(block)
    except KeyboardInterrupt:
        pass
    seg_q.put(None)
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
