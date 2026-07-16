"""FlowMatchScheduler — the rule the DiT is trained and sampled under (coral).

Flow matching draws a straight line between pure noise (t=0) and a real latent
(t=1):

    x_t = (1 - t) * noise + t * latent          # a point on that line
    v   = latent - noise                        # the (constant) velocity along it

Training: pick a random t, build x_t, and ask the DiT to predict v. That's it —
no noise schedules, no variance tables. Two real ACE-Step details are kept:

  * **logit-normal t sampling** — t = sigmoid(randn()) concentrates training
    where the prediction is hardest (mid-way), instead of uniform t;
  * a **shifted timestep schedule** at sampling — with shift > 1 the Euler
    steps are packed toward the noisy end of the line, where the velocity
    field changes fastest (the real turbo model samples 8 steps at shift=3).

Sampling walks the line with Euler steps, optionally running the DiT twice per
step for **classifier-free guidance**: once with the caption, once with the
learned null caption, then extrapolating v = v_null + g * (v_cond - v_null).
"""

import torch

# --- Real ACE-Step v1.5, for comparison ------------------------------------
# Same flow-matching objective. The pipeline computes its own shifted sigma
# schedule and hands it to a FlowMatchEulerDiscreteScheduler; base/sft
# checkpoints use CFG with a learned null embedding, while the distilled turbo
# model needs neither — dynamic-shift DMD2 distillation (shift sampled from
# {1,2,3}) cuts 50 steps to 8, rendering a 240s track in ~1s on an A100.
# Here (toy): plain Euler, num_inference_steps=16, shift=3, optional CFG.
# ---------------------------------------------------------------------------


class FlowMatchScheduler:
    def __init__(self, num_inference_steps: int = 16, shift: float = 3.0):
        self.num_inference_steps = num_inference_steps
        self.shift = shift

    def add_noise(self, latent: torch.Tensor):
        """Make one training example. Returns (x_t, t, target_velocity)."""
        B = latent.shape[0]
        noise = torch.randn_like(latent)
        t = torch.sigmoid(torch.randn(B, device=latent.device))  # logit-normal in (0, 1)
        t_ = t[:, None, None]                                    # broadcast over [d, T]
        x_t = (1 - t_) * noise + t_ * latent
        velocity = latent - noise
        return x_t, t, velocity

    def timesteps(self, device) -> torch.Tensor:
        """The t grid to integrate over: 0 -> 1, packed toward the noisy end.

        sigma = 1 - t is the remaining noise level; shifting sigma by
        s' = shift*s / (1 + (shift-1)*s) keeps s'=1 at noise and s'=0 at data
        but spends more of the budget where sigma is high.
        """
        s = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, device=device)
        sigma = self.shift * s / (1 + (self.shift - 1) * s)
        return 1 - sigma                                         # [steps+1], 0 -> 1

    @torch.no_grad()
    def sample(self, dit, text_embed, source_latent, shape, device="cpu",
               guidance_scale: float = 1.0, null_embed=None):
        """Integrate noise -> latent in num_inference_steps Euler steps."""
        x = torch.randn(shape, device=device)                    # start at t = 0 (pure noise)
        ts = self.timesteps(device)
        for i in range(self.num_inference_steps):
            t = ts[i].expand(shape[0])                           # current position on the line
            v = dit(x, t, text_embed, source_latent)
            if guidance_scale != 1.0:                            # classifier-free guidance
                v_null = dit(x, t, null_embed, source_latent)
                v = v_null + guidance_scale * (v - v_null)
            x = x + v * (ts[i + 1] - ts[i])                      # Euler step toward the latent
        return x
