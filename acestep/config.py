"""One config for the whole tiny ACE-Step pipeline.

ACE-Step is four models wired in series, but they all share the same little
transformer recipe (the planner LM and the DiT) and the same resolution
hierarchy, so everything fits in one dataclass. Defaults are deliberately small
so all four stages train on a laptop CPU in a few minutes.

The resolution hierarchy. The toy keeps ACE-Step's *actual* frame rates —
a true 25Hz latent and true 5Hz codes — and only shrinks the audio itself
(one 2-second bar of 4kHz mono instead of minutes of 48kHz stereo):

    waveform_len = 8000 samples   (2 seconds of real audio, the green output)
        |  VAE encoder, x160 down  (strides 4*4*2*5; the real squeeze is x1920)
    latent_len   =   50 frames    (50 frames / 2s = the real 25Hz)
        |  FSQ attention pooling, x5 down  (the exact real 25Hz -> 5Hz ratio)
    code_len     =   10 codes     (10 codes / 2s = the real 5Hz)

Real ACE-Step v1.5, for comparison (a 240-second song):
    waveform   [2, 11_520_000]   48kHz stereo
        |  AutoencoderOobleck, 1920x temporal squeeze, 64-dim latent
    latent     [64, 6000]        25Hz, 64 dims/frame
        |  FSQ attention pooling, x5 down, codebook ~64k
    codes      [1200]            5Hz, ids 0..~64000  (1200 = 240s x 5Hz)
"""

from dataclasses import dataclass
from math import prod


@dataclass
class AceConfig:
    # ---- resolution hierarchy --------------------------------------------
    sample_rate: int = 4000         # playback rate of the toy audio (real: 48kHz stereo)
    waveform_len: int = 8000        # samples in one bar = 2 seconds (real: up to 10 min)
    latent_dim: int = 16            # channels of the VAE / DiT latent (real: 64)
    latent_len: int = 50            # latent frames = waveform_len / 160 (a real 25Hz)
    code_len: int = 10              # 5Hz code frames = latent_len / 5 (a real 5Hz)
    fsq_levels: tuple = (8, 8, 8)   # FSQ levels per dim -> codebook 8*8*8 = 512
                                    # (real: 6 dims, codebook ~64k)

    # ---- VAE (tiny Oobleck) ----------------------------------------------
    vae_strides: tuple = (4, 4, 2, 5)           # per-stage downsampling, product = 160
    vae_channels: tuple = (16, 24, 32, 48, 64)  # widths: pre-conv, then after each stage
    kl_weight: float = 1e-4                     # tiny KL term keeps it a *V*AE

    # ---- shared transformer recipe (used by planner LM *and* DiT) --------
    hidden_size: int = 64           # model / embedding dimension
    num_layers: int = 2             # planner LM transformer blocks
    dit_layers: int = 4             # DiT blocks (the renderer needs a little more depth)
    num_heads: int = 4              # query heads
    num_kv_heads: int = 2           # key/value heads (GQA)
    head_dim: int = 16              # dimension per head
    intermediate_size: int = 128    # SwiGLU hidden dim
    max_seq_len: int = 32           # >= planner length (17) and DiT patch tokens (25)
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6

    # ---- DiT specifics (mirroring the real v1.5 renderer) ----------------
    patch_size: int = 2             # patchify halves the sequence (real: 25Hz -> 12.5Hz)
    sliding_window: int = 8         # odd DiT layers attend +-8 tokens (real: SWA odd layers)
    cond_dropout: float = 0.1       # drop the caption while training -> a usable null embed (CFG)

    # ---- planner LM vocabulary -------------------------------------------
    # Full vocab = [letter tags] + [<think>, </think>] + [scale-degree tokens]
    # + [audio-code tokens]. n_letters is filled in from the data at train
    # time (like ModelConfig(vocab_size=...) elsewhere).
    n_letters: int = 0
    n_degrees: int = 10             # scale-degree "metadata" tokens for the <think> plan

    # ---- diffusion sampling ----------------------------------------------
    num_inference_steps: int = 16   # Euler steps when sampling (real: 50, distilled to 8)
    shift: float = 3.0              # timestep-schedule shift (the real turbo uses shift=3)
    guidance_scale: float = 1.0     # >1 turns on classifier-free guidance when sampling

    @property
    def num_codes(self) -> int:
        """Size of the FSQ codebook = product of the per-dim levels."""
        return prod(self.fsq_levels)

    @property
    def planner_vocab(self) -> int:
        """Planner ids: letters, then <think> and </think>, then degrees, then codes."""
        return self.n_letters + 2 + self.n_degrees + self.num_codes
