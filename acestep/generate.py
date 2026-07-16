"""Generate music from a letter (or several) with the trained pipeline.

Run:  python generate.py            # letter "a"
      python generate.py e          # letter "e"
      python generate.py merhaba    # one 2s bar per letter — a little tune

It prints the shape at every region boundary, checks each generated bar note
by note (the FFT of every beat should peak at that beat's melody frequency),
and writes the result to out.wav using only the stdlib `wave` module.
"""

import struct
import sys
import wave

import torch
from dit import DiT
from fsq import FSQBridge
from pipeline import AceStepPipeline
from planner import Planner
from text_encoder import TextEncoder
from vae import AutoencoderOobleckTiny

from data import ALPHABET, BEATS_PER_BAR, letter_notes, stoi

CHECKPOINT = "acestep.pt"


def load() -> AceStepPipeline:
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    vae = AutoencoderOobleckTiny(cfg); vae.load_state_dict(ckpt["vae"])
    fsq = FSQBridge(cfg); fsq.load_state_dict(ckpt["fsq"])
    text_encoder = TextEncoder(cfg); text_encoder.load_state_dict(ckpt["text_encoder"])
    planner = Planner(cfg); planner.load_state_dict(ckpt["planner"])
    dit = DiT(cfg); dit.load_state_dict(ckpt["dit"])
    return AceStepPipeline(cfg, vae, fsq, text_encoder, planner, dit, ckpt["latent_scale"])


def dominant_hz(beat: torch.Tensor, sample_rate: int, min_hz: float = 160.0) -> float:
    """The strongest frequency of one beat, ignoring the bass register."""
    spectrum = torch.fft.rfft(beat).abs()
    hz_per_bin = sample_rate / beat.shape[0]
    lo = int(min_hz / hz_per_bin)                      # skip DC + the 110Hz bass
    return (spectrum[lo:].argmax().item() + lo) * hz_per_bin


def check_bar(bar: torch.Tensor, letter: str, sample_rate: int) -> int:
    """FFT each beat of a generated bar against the letter's 4-note answer key."""
    beat_len = bar.shape[0] // BEATS_PER_BAR
    hits = 0
    for b, expected in enumerate(letter_notes(stoi[letter])):
        got = dominant_hz(bar[b * beat_len:(b + 1) * beat_len], sample_rate)
        ok = abs(got - expected) / expected < 0.06     # within ~a semitone
        hits += ok
        print(f"   {letter}   beat {b + 1}   {expected:7.1f} Hz   {got:7.1f} Hz   "
              f"{'ok' if ok else 'MISS'}")
    return hits


def write_wav(path: str, wave1d: torch.Tensor, sample_rate: int):
    pcm = (wave1d.clamp(-1, 1) * 32767).short().tolist()
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(b"".join(struct.pack("<h", s) for s in pcm))


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "a"
    letters = [c for c in text.lower() if c in stoi]
    if not letters:
        print(f"no usable letters in {text!r}; alphabet is: {''.join(ALPHABET)}")
        return

    pipe = load()
    sample_rate = pipe.cfg.sample_rate
    tags = torch.tensor([stoi[c] for c in letters])

    print("region shapes (one batch through the whole pipeline):")
    waves = pipe.generate(tags, verbose=True)          # [B, 1, 8000]

    print("\nletter  beat   expected      got     ok")
    hits, total = 0, 0
    segments = []
    for i, c in enumerate(letters):
        bar = waves[i, 0]
        hits += check_bar(bar, c, sample_rate)
        total += BEATS_PER_BAR
        segments.append(bar)

    write_wav("out.wav", torch.cat(segments), sample_rate)
    seconds = len(letters) * pipe.cfg.waveform_len / sample_rate
    print(f"\nnotes on pitch: {hits}/{total}")
    print(f"wrote out.wav  ({len(letters)} bar(s), {seconds:.0f}s @ {sample_rate} Hz)")


if __name__ == "__main__":
    main()
