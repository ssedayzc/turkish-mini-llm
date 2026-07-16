"""The toy "music": every Turkish letter is a tag that maps to one 2-second bar.

This is the continuous-signal analogue of the other folders' name corpus.
Names are discrete (you can't diffuse over characters), so instead each of the
29 Turkish letters becomes a *caption* whose "song" is one bar of actual music:

    letter index i  ->  4 plucked melody notes from the A-minor pentatonic
                        scale (a contour picked by i) over a bass note that
                        pulses on every beat.

The clip is 2 seconds at 4kHz — real, playable audio at 120 bpm. Notes are
synthesized with a few harmonics under an exponential decay envelope, so they
sound like a plucked string, and concatenating the letters of a word
(`python3 generate.py merhaba`) plays a little tune, one bar per letter.

Because the melody is deterministic per letter, every letter has a known
4-note answer key — which is exactly what the end-to-end test checks with an
FFT on each beat.
"""

import torch
from config import AceConfig

# --- Real ACE-Step v1.5 input, for comparison ------------------------------
# A real prompt is a caption (free text, e.g. "lofi hip hop, mellow piano"),
# lyrics with structure tags ([Intro], [Verse], [Chorus], [Instrumental], ...),
# and meta (bpm 60-180, key, duration 10s-10min). Empty lyrics -> instrumental;
# ~2-3 words/sec keeps vocals natural. Here (toy): the entire "caption + lyrics"
# is a single Turkish letter, and its "song" is one deterministic 2s bar.
# ---------------------------------------------------------------------------

# 29 Turkish lowercase letters — the tag/caption vocabulary.
ALPHABET = list("abcçdefgğhıijklmnoöprsştuüvyz")
N_LETTERS = len(ALPHABET)

stoi = {ch: i for i, ch in enumerate(ALPHABET)}   # letter -> tag id
itos = {i: ch for i, ch in enumerate(ALPHABET)}   # tag id -> letter

# A-minor pentatonic over two octaves, in semitones above the root (A3 = 220Hz).
SCALE = [0, 3, 5, 7, 10, 12, 15, 17, 19, 22]
ROOT_HZ = 220.0                # melody root: A3
BASS_HZ = 110.0                # bass root: A2, pulses on every beat
BEATS_PER_BAR = 4              # 4 beats x 0.5s = one 2-second bar at 120 bpm

# Four melodic contours (scale-degree offsets from the letter's start degree).
# start = i % 10 and contour = i // 10, so all 29 letters get distinct bars.
CONTOURS = [(0, 2, 4, 3), (0, 1, 3, 2), (0, 4, 2, 1), (0, 3, 1, 4)]

PLUCK_HARMONICS = ((1, 1.0), (2, 0.5), (3, 0.25))   # a plucked-string timbre
BASS_HARMONICS = ((1, 1.0),)                        # pure sine, keeps the FFT clean


def letter_degrees(letter_idx: int) -> list[int]:
    """The 4 scale degrees of one letter's melody — the planner's <think> plan."""
    start = letter_idx % len(SCALE)
    contour = CONTOURS[letter_idx // len(SCALE)]
    return [(start + step) % len(SCALE) for step in contour]


def degree_to_hz(degree: int) -> float:
    return ROOT_HZ * 2 ** (SCALE[degree] / 12)


def letter_notes(letter_idx: int) -> list[float]:
    """The 4 melody frequencies (Hz) of one letter's bar — the answer key."""
    return [degree_to_hz(d) for d in letter_degrees(letter_idx)]


def _pluck(freq: float, t: torch.Tensor, sample_rate: int,
           decay: float, harmonics) -> torch.Tensor:
    """One note: a few harmonics under an exponential decay envelope."""
    env = torch.exp(-decay * t / t[-1])
    wave = torch.zeros_like(t)
    for harmonic, amp in harmonics:
        if freq * harmonic < 0.45 * sample_rate:            # skip aliasing overtones
            wave = wave + amp * torch.sin(2 * torch.pi * freq * harmonic * t)
    return env * wave


def letter_song(letter_idx: int, cfg: AceConfig) -> torch.Tensor:
    """The 2-second bar for one letter: [waveform_len], peak-normalized."""
    beat_len = cfg.waveform_len // BEATS_PER_BAR
    t_beat = torch.arange(beat_len).float() / cfg.sample_rate

    beats = []
    for freq in letter_notes(letter_idx):
        melody = _pluck(freq, t_beat, cfg.sample_rate, decay=4.0,
                        harmonics=PLUCK_HARMONICS)
        bass = 0.4 * _pluck(BASS_HZ, t_beat, cfg.sample_rate, decay=3.0,
                            harmonics=BASS_HARMONICS)
        beats.append(melody + bass)                        # same bass every beat = the pulse

    wave = torch.cat(beats)
    return 0.9 * wave / wave.abs().max()                   # normalize peak to 0.9


_song_cache: dict = {}


def all_songs(cfg: AceConfig) -> torch.Tensor:
    """Every letter's bar stacked: [N_LETTERS, 1, waveform_len] (cached)."""
    key = (cfg.sample_rate, cfg.waveform_len)
    if key not in _song_cache:
        songs = [letter_song(i, cfg) for i in range(N_LETTERS)]
        _song_cache[key] = torch.stack(songs).unsqueeze(1)  # add the 1-channel axis
    return _song_cache[key]


def get_batch(cfg: AceConfig, batch_size: int):
    """Sample random letters. Returns (tag_ids [B], waveforms [B, 1, waveform_len])."""
    songs = all_songs(cfg)                                 # [N_LETTERS, 1, L]
    tag_ids = torch.randint(N_LETTERS, (batch_size,))
    return tag_ids, songs[tag_ids]
