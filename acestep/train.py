"""Train the tiny ACE-Step pipeline, one region at a time.

Run:  python train.py

The four models are trained in series, exactly the order their outputs feed
each other — you cannot make 5Hz code targets before the VAE+FSQ exist, and
the DiT needs the VAE's latents to denoise toward:

    1. VAE       reconstruct the waveform (+ tiny KL)     (green)
    2. FSQ       round-trip the 25Hz latent through codes (blue bridge)
    3. Planner   tag -> <think> plan -> 5Hz codes         (purple)
    4. DiT       flow-match noise -> latent               (coral)

Everything is torch-only and small; all four stages finish in a few minutes
on a laptop CPU (the VAE, working on raw 8000-sample waveforms, is the slow
one).
"""

import torch
from config import AceConfig
from dit import DiT
from flow import FlowMatchScheduler
from fsq import FSQBridge
from planner import Planner, make_batch
from text_encoder import TextEncoder
from vae import AutoencoderOobleckTiny

from data import N_LETTERS, all_songs, letter_degrees

LEARNING_RATE = 3e-3
VAE_STEPS = 3000
FSQ_STEPS = 1500
PLANNER_STEPS = 1500
DIT_STEPS = 4000
SEED = 1337

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
torch.manual_seed(SEED)

cfg = AceConfig(n_letters=N_LETTERS)
vae = AutoencoderOobleckTiny(cfg).to(device)
fsq = FSQBridge(cfg).to(device)
text_encoder = TextEncoder(cfg).to(device)
planner = Planner(cfg).to(device)
dit = DiT(cfg).to(device)
flow = FlowMatchScheduler(cfg.num_inference_steps, cfg.shift)

n_params = sum(p.numel() for m in (vae, fsq, text_encoder, planner, dit)
               for p in m.parameters())
print(f"device={device}  letters={N_LETTERS}  codebook={cfg.num_codes}  parameters={n_params:,}\n")
mse = torch.nn.MSELoss()

# The dataset is deterministic and tiny (29 bars), so every stage trains
# full-batch: all letters, every step.
songs = all_songs(cfg).to(device)                    # [N, 1, 8000]
all_tags = torch.arange(N_LETTERS, device=device)    # [N]
degrees = torch.tensor([letter_degrees(i) for i in range(N_LETTERS)],
                       device=device)                # [N, 4], the <think> targets


# ---------------------------------------------------------------------------
# Stage 1 — VAE: learn to reconstruct the toy bars (MSE + tiny KL).
# ---------------------------------------------------------------------------
opt = torch.optim.AdamW(vae.parameters(), lr=LEARNING_RATE)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=VAE_STEPS)
for step in range(1, VAE_STEPS + 1):
    recon, kl = vae(songs)
    loss = mse(recon, songs) + cfg.kl_weight * kl
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 500 == 0 or step == 1:
        print(f"[1/4 vae]     step {step:5d}  recon mse {loss.item():.5f}")

# All letter latents in one shot; their std normalizes the latent for the DiT.
vae.eval()
with torch.no_grad():
    latents = vae.encode(songs)                      # [N, d, 50]
latent_scale = latents.std().item()
print(f"              latent_scale = {latent_scale:.4f}\n")


# ---------------------------------------------------------------------------
# Stage 2 — FSQ: round-trip the latent through discrete 5Hz codes.
# ---------------------------------------------------------------------------
opt = torch.optim.AdamW(fsq.parameters(), lr=LEARNING_RATE)
for step in range(1, FSQ_STEPS + 1):
    source, _ = fsq(latents)
    loss = mse(source, latents)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0 or step == 1:
        print(f"[2/4 fsq]     step {step:5d}  roundtrip mse {loss.item():.5f}")

# Each letter's 5Hz code sequence — the planner's training targets.
fsq.eval()
with torch.no_grad():
    codes_per_letter = fsq.encode(latents)           # [N, code_len]
unique = len(set(map(tuple, codes_per_letter.tolist())))
print(f"              {unique}/{N_LETTERS} letters got a distinct code sequence\n")


# ---------------------------------------------------------------------------
# Stage 3 — Planner: a tiny LM that writes tag -> <think> plan -> 5Hz codes.
# ---------------------------------------------------------------------------
opt = torch.optim.AdamW(planner.parameters(), lr=LEARNING_RATE)
for step in range(1, PLANNER_STEPS + 1):
    inp, tgt = make_batch(all_tags, degrees, codes_per_letter, cfg)
    _, loss = planner(inp, tgt)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0 or step == 1:
        print(f"[3/4 planner] step {step:5d}  plan ce {loss.item():.5f}")

planner.eval()
with torch.no_grad():
    pred_codes, pred_think = planner.generate(all_tags)
code_acc = (pred_codes == codes_per_letter).float().mean().item()
think_acc = (pred_think == degrees).float().mean().item()
print(f"              planner accuracy: codes {code_acc*100:.1f}%  <think> {think_acc*100:.1f}%\n")


# ---------------------------------------------------------------------------
# Stage 4 — DiT: flow-match noise -> latent, conditioned on caption (via
# cross-attention) + the 5Hz source latent (inside the composite input).
# ---------------------------------------------------------------------------
opt = torch.optim.AdamW(list(dit.parameters()) + list(text_encoder.parameters()),
                        lr=LEARNING_RATE)
# Cosine-decay the LR to ~0 so the velocity field settles instead of jittering.
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=DIT_STEPS)
with torch.no_grad():
    source_all = fsq.decode(codes_per_letter) / latent_scale   # conditioning skeleton
target_all = latents / latent_scale
for step in range(1, DIT_STEPS + 1):
    x_t, t, velocity = flow.add_noise(target_all)
    text_embed = text_encoder(all_tags)
    # Caption dropout: sometimes hide the tag, so the null embedding learns
    # "render *something*" — that's what CFG extrapolates against.
    drop = torch.rand(N_LETTERS, device=device) < cfg.cond_dropout
    text_embed = torch.where(drop[:, None, None],
                             text_encoder.null_embed(N_LETTERS), text_embed)
    pred = dit(x_t, t, text_embed, source_all)
    loss = mse(pred, velocity)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 500 == 0 or step == 1:
        print(f"[4/4 dit]     step {step:5d}  velocity mse {loss.item():.5f}")


# ---------------------------------------------------------------------------
# Save one checkpoint with all four models.
# ---------------------------------------------------------------------------
torch.save({
    "cfg": cfg,
    "latent_scale": latent_scale,
    "vae": vae.state_dict(),
    "fsq": fsq.state_dict(),
    "text_encoder": text_encoder.state_dict(),
    "planner": planner.state_dict(),
    "dit": dit.state_dict(),
}, "acestep.pt")
print("\nsaved checkpoint to acestep.pt")
