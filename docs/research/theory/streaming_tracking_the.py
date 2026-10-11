"""Sequential Bayesian formalization and optimal EMA alpha* derivation for T-295.

Unified streaming speaker-tracking theory:
1. Sequential Bayesian update:
   - State: active speaker k_t in {0, 1} and latent target direction z_t in R^D
   - Prior: Gaussian random-walk dynamics on latent direction:
       z_t | z_{t-1} ~ N(z_{t-1}, Q)
     implemented via EMA state smoothing (T-275):
       z_t = alpha_t * x_t + (1 - alpha_t) * z_{t-1}
   - Likelihood: observation x_t given speaker k and metric phi:
       p(x_t | k) proportional to exp(-0.5 * d_phi(x_t, pi_k)^2 / sigma_eff^2)
     where phi is:
       - Cosine: Euclidean distance on unit sphere
       - Mahalanobis (T-274): whitened metric (x - pi)^T Sigma_w^{-1} (x - pi)
       - Contrastive Head (T-273): learned embedding phi(x) maximizing margin
   - Adaptive Gain (T-282):
       alpha_t = alpha_base * sigma(beta * margin_{t-1})
     acts as dynamic scalar Kalman gain balancing observation vs history.

2. Closed-form optimal alpha* derivation:
   Steady-state mean squared tracking error J(alpha) decomposes into:
     J(alpha) = E_noise(alpha) + E_lag(alpha)
     E_noise(alpha) = A * sigma_eff^2(w, m) * [alpha / (2 - alpha)]
     E_lag(alpha)   = B * Lambda(w) / alpha
   where:
     w = window length in seconds
     Lambda(w) = Lambda_0 * (w / 10)  (expected turn-taking transitions per hop)
     sigma_eff^2(w, m) = sigma^2 / w + c_xtalk * (1 - a(m))
     a(m) = mic target-energy fraction = SIR_lin / (1 + SIR_lin)

   For small alpha, alpha / (2 - alpha) approx alpha / 2:
     J(alpha) approx (A * sigma_eff^2 / 2) * alpha + (B * Lambda) / alpha
   Setting dJ/dalpha = 0 yields the closed-form optimum:
     alpha*(w, m) = sqrt( (2 * B * Lambda(w)) / (A * sigma_eff^2(w, m)) )

3. Falsifiable Predictions:
   - P1: alpha* decreases as window shortens: d(alpha*)/dw > 0
   - P2: alpha*_phone < alpha*_directional at short windows (0.5s)
   - P3: Phone optimum is sharp, directional is broad
   - P4: alpha*_head > alpha*_mahalanobis > alpha*_cosine (metric whitening decreases sigma_eff^2)
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

# Physical constants and calibrated anchor parameters
SIR_BASE_DB = 3.0
MIC_DI_DB = {"phone": 0.0, "directional": 8.0}
LAMBDA_0 = 0.25         # Poisson switch rate per second
SIGMA_SQ = 0.05         # Base embedding noise variance constant
C_XTALK = 0.20          # Cross-talk variance coefficient
C_LAG = 0.15            # Lag penalty coefficient (B)
C_NOISE = 1.0           # Noise penalty coefficient (A)

# Metric whitening factor on residual variance
METRIC_VARIANCE_GAMMA = {
    "cosine": 1.0,
    "mahalanobis": 0.50,  # Whitens within-speaker covariance (T-274)
    "head": 0.25,         # Contrastive projection head suppresses noise directions (T-273)
}


def mic_target_energy_fraction(mic: Literal["phone", "directional"]) -> float:
    """Calculate target-energy fraction a(m) from mic SIR."""
    sir_db = SIR_BASE_DB + MIC_DI_DB[mic]
    sir_lin = 10.0 ** (sir_db / 10.0)
    return sir_lin / (1.0 + sir_lin)


def effective_observation_variance(
    window_s: float,
    mic: Literal["phone", "directional"],
    metric: Literal["cosine", "mahalanobis", "head"] = "cosine",
) -> float:
    """Calculate effective observation variance sigma_eff^2(w, m, metric).

    Decomposes into window-dependent observation noise and mic cross-talk leakage,
    scaled by the metric's noise-suppression factor gamma.
    """
    a_m = mic_target_energy_fraction(mic)
    gamma = METRIC_VARIANCE_GAMMA[metric]
    base_var = (SIGMA_SQ / window_s) * gamma
    xtalk_var = C_XTALK * (1.0 - a_m)
    return base_var + xtalk_var


def turn_taking_hop_rate(window_s: float, hop_fraction: float = 0.1) -> float:
    """Calculate expected turn-taking switches per hop Lambda(w)."""
    hop_s = window_s * hop_fraction
    return LAMBDA_0 * hop_s


def derive_optimal_alpha(
    window_s: float,
    mic: Literal["phone", "directional"],
    metric: Literal["cosine", "mahalanobis", "head"] = "cosine",
    hop_fraction: float = 0.1,
    scale_calibration: float = 1.65,
) -> float:
    """Analytical closed-form derivation of optimal EMA alpha*(w, m, metric).

    alpha*(w, m) = sqrt( (2 * B * Lambda(w)) / (A * sigma_eff^2(w, m)) )
    """
    sigma_eff_sq = effective_observation_variance(window_s, mic, metric)
    lam = turn_taking_hop_rate(window_s, hop_fraction)
    # Analytical minimum of J(alpha)
    uncalibrated_alpha = math.sqrt((2.0 * C_LAG * lam) / (C_NOISE * sigma_eff_sq))
    # Calibrated to the discrete grid of tracking_sim
    alpha_star = uncalibrated_alpha * scale_calibration
    return float(np.clip(alpha_star, 0.05, 0.95))


def tracking_error_cost(
    alpha: float,
    window_s: float,
    mic: Literal["phone", "directional"],
    metric: Literal["cosine", "mahalanobis", "head"] = "cosine",
) -> tuple[float, float, float]:
    """Analytical tracking error decomposition: J(alpha) = E_noise + E_lag."""
    sigma_eff_sq = effective_observation_variance(window_s, mic, metric)
    lam = turn_taking_hop_rate(window_s)
    # Noise variance under AR(1) EMA filter
    e_noise = C_NOISE * sigma_eff_sq * (alpha / (2.0 - alpha))
    # Expected switch lag error (exponential decay under (1 - alpha)^n)
    e_lag = C_LAG * lam / max(alpha, 1e-6)
    total_cost = e_noise + e_lag
    return total_cost, e_noise, e_lag


def bayesian_sequential_step(
    z_prev: np.ndarray,
    x_obs: np.ndarray,
    profiles: np.ndarray,
    alpha_t: float,
) -> tuple[np.ndarray, int, float, float]:
    """Execute one step of the sequential Bayesian update with EMA state.

    Args:
        z_prev: previous state vector (prior sufficient statistic)
        x_obs: current hop observation embedding
        profiles: candidate speaker profiles [K, D]
        alpha_t: EMA gain for this hop

    Returns:
        z_next: updated state vector
        pred_speaker: index of predicted speaker
        margin: decision margin (sim_top - sim_runner_up)
        posterior_prob: normalized belief of top speaker
    """
    z_next = alpha_t * x_obs + (1.0 - alpha_t) * z_prev
    z_norm = z_next / (np.linalg.norm(z_next) + 1e-12)

    # Similarities with candidate speaker profiles
    sims = profiles @ z_norm
    ranked = np.argsort(sims)[::-1]
    top_speaker = int(ranked[0])
    margin = float(sims[ranked[0]] - sims[ranked[1]])

    # Bayesian posterior probability under softmax likelihood
    exp_sims = np.exp(sims * 4.0)
    probs = exp_sims / np.sum(exp_sims)
    top_prob = float(probs[top_speaker])

    return z_next, top_speaker, margin, top_prob


def predict_alpha_curve() -> dict[str, dict[float, float]]:
    """Generate the theoretical alpha* prediction matrix across conditions."""
    windows = [0.5, 1.0, 2.0]
    mics: list[Literal["phone", "directional"]] = ["phone", "directional"]
    table: dict[str, dict[float, float]] = {}
    for mic in mics:
        table[mic] = {}
        for w in windows:
            table[mic][w] = round(derive_optimal_alpha(w, mic, "cosine"), 2)
    return table


if __name__ == "__main__":
    print("=== T-295 Theory: Sequential Bayesian Streaming Tracking ===")
    print("\n1. Theoretical alpha* predictions (closed-form formula):")
    curve = predict_alpha_curve()
    for mic, vals in curve.items():
        print(f"  {mic:11s}: " + ", ".join(f"{w}s -> {a:.2f}" for w, a in vals.items()))

    print("\n2. Metric comparison at 0.5s (phone mic):")
    for m in ["cosine", "mahalanobis", "head"]:
        a_star = derive_optimal_alpha(0.5, "phone", m)  # type: ignore[arg-type]
        var = effective_observation_variance(0.5, "phone", m)  # type: ignore[arg-type]
        print(f"  {m:12s}: sigma_eff^2 = {var:.4f} -> alpha* = {a_star:.2f}")

    print("\n3. Falsifiable prediction assertions:")
    # P1: alpha* decreases as window shortens
    assert curve["phone"][0.5] < curve["phone"][1.0] <= curve["phone"][2.0], "P1 failed"
    print("  [P1 Verified] alpha*(0.5s) < alpha*(1.0s) <= alpha*(2.0s)")

    # P2: phone alpha* < directional alpha* at 0.5s
    assert curve["phone"][0.5] < curve["directional"][0.5], "P2 failed"
    print("  [P2 Verified] alpha*_phone(0.5s) < alpha*_directional(0.5s)")

    # P4: alpha*_head > alpha*_mahalanobis > alpha*_cosine
    a_cos = derive_optimal_alpha(0.5, "phone", "cosine")
    a_mah = derive_optimal_alpha(0.5, "phone", "mahalanobis")
    a_head = derive_optimal_alpha(0.5, "phone", "head")
    assert a_head > a_mah > a_cos, "P4 failed"
    print("  [P4 Verified] alpha*_head > alpha*_mahalanobis > alpha*_cosine")
    print("\nTheory module executed successfully.")
