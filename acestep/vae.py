"""A tiny Oobleck-style waveform VAE — the green "decode" region.

ACE-Step's AutoencoderOobleck compresses 48kHz audio into a 25Hz continuous
latent and decodes it back, working *directly on the waveform* (no mel
spectrogram). Ours does the same job at toy scale:

    encode:  waveform [B, 1, 8000] --x160 down--> latent [B, 16, 50]  (25Hz!)
    decode:  latent   [B, 16, 50]  --x160 up----> waveform [B, 1, 8000]

Both directions are a stack of strided convolutions, one stage per entry in
``cfg.vae_strides`` (4*4*2*5 = 160). Two Oobleck signatures are kept:

  * the **Snake activation** ``x + sin^2(a*x)/a`` — a periodic nonlinearity
    invented for audio nets, whose built-in oscillation makes synthesizing
    pitched waveforms much easier than ReLU-family activations;
  * a real **variational** bottleneck — the encoder emits (mean, logvar), we
    sample with the reparameterization trick and add a small KL penalty.

Kernel/stride/padding satisfy k - 2p = s, so each stage divides (encoder) or
multiplies (decoder) the length exactly by its stride — no length arithmetic
surprises anywhere in the pipeline.
"""

import torch
from config import AceConfig
from torch import nn

# --- Real ACE-Step v1.5, for comparison ------------------------------------
# The real AutoencoderOobleck compresses 48kHz *stereo* audio into a 64-dim
# latent at 25Hz — a 1920x temporal squeeze (48000 -> 25 frames/sec), trained
# with reconstruction + adversarial (GAN) + KL objectives. A 240s song:
#     waveform [2, 11_520_000]  <->  latent [64, 6000]
# Here (toy): mono 4kHz, a 16-dim latent, a x160 squeeze (8000 -> 50), trained
# with plain MSE + a tiny KL (no discriminator). The 25Hz frame rate is real:
#     waveform [1, 8000]        <->  latent [16, 50]     (50 frames / 2s = 25Hz)
# ---------------------------------------------------------------------------


class Snake(nn.Module):
    """Snake activation: x + sin^2(alpha * x) / alpha, one learned alpha per channel."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.sin(self.alpha * x) ** 2 / (self.alpha + 1e-9)


def _updown_args(stride: int):
    """Kernel/padding with k - 2p = s, so length scales exactly by the stride."""
    kernel = 2 * stride if stride % 2 == 0 else 2 * stride + 1
    return kernel, (kernel - stride) // 2


class AutoencoderOobleckTiny(nn.Module):
    def __init__(self, cfg: AceConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.latent_dim
        chs, strides = cfg.vae_channels, cfg.vae_strides   # (16,24,32,48,64), (4,4,2,5)

        # ---- encoder: waveform -> (mean, logvar) --------------------------
        enc = [nn.Conv1d(1, chs[0], kernel_size=7, padding=3)]
        for c_in, c_out, s in zip(chs[:-1], chs[1:], strides):
            k, p = _updown_args(s)
            enc += [Snake(c_in), nn.Conv1d(c_in, c_out, kernel_size=k, stride=s, padding=p)]
        enc += [Snake(chs[-1]), nn.Conv1d(chs[-1], 2 * d, kernel_size=7, padding=3)]
        self.encoder = nn.Sequential(*enc)                 # [B,1,8000] -> [B,2d,50]

        # ---- decoder: latent -> waveform (mirror image) --------------------
        dec = [nn.Conv1d(d, chs[-1], kernel_size=7, padding=3)]
        for c_in, c_out, s in zip(chs[:0:-1], chs[-2::-1], strides[::-1]):
            k, p = _updown_args(s)
            dec += [Snake(c_in), nn.ConvTranspose1d(c_in, c_out, kernel_size=k, stride=s, padding=p)]
        dec += [Snake(chs[0]), nn.Conv1d(chs[0], 1, kernel_size=7, padding=3), nn.Tanh()]
        self.decoder = nn.Sequential(*dec)                 # [B,d,50] -> [B,1,8000]

    def encode(self, waveform: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """[B, 1, 8000] -> latent [B, d, 50]. sample=True draws z ~ N(mean, var)."""
        mean, logvar = self.encoder(waveform).chunk(2, dim=1)
        if sample:
            mean = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        return mean

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)                        # [B, d, 50] -> [B, 1, 8000]

    def forward(self, waveform: torch.Tensor):
        """Training pass. Returns (reconstruction, kl) — add cfg.kl_weight * kl."""
        mean, logvar = self.encoder(waveform).chunk(2, dim=1)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)   # reparameterize
        kl = 0.5 * (mean.pow(2) + logvar.exp() - 1 - logvar).mean()
        return self.decoder(z), kl
