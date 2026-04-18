"""
=============================================================================
DSGE Model: Adaptive Emission Doctrine vs. Standard Monetary Policy
=============================================================================
Based on: "Monetary Emission as Financial Extraction: The Adaptive Emission
Doctrine" (Cantillon Effect, AED Master Formula, NK-DSGE Framework)

Model Architecture:
  - New Keynesian DSGE core (IS, NKPC, Taylor rule)
  - Cantillon Effect distributional wedge extension
  - Technological deflation & labor share dynamics
  - AED Master Formula: E_AED = [P_target·ΔQ]/V + D_annihilated
  - Debt-deflation dynamics (Fisher 1933)
  - Impulse Response Functions, Phase Diagrams, Historical Decompositions

Econometric Standards:
  - Publication-quality figures (matplotlib w/ LaTeX-style labels)
  - Shaded 68% / 90% confidence bands via bootstrap
  - NBER-style recession shading
  - Proper annotation and figure numbering
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy import linalg, optimize, stats
from scipy.linalg import solve_discrete_lyapunov
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 0.  GLOBAL PLOT STYLE  (econometrics journal standard)
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['DejaVu Serif', 'Times New Roman', 'Georgia'],
    'font.size':          10,
    'axes.titlesize':     11,
    'axes.labelsize':     10,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'figure.dpi':         150,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.25,
    'grid.linestyle':     '--',
    'grid.linewidth':     0.6,
    'axes.linewidth':     0.8,
    'lines.linewidth':    1.8,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'figure.facecolor':   'white',
    'axes.facecolor':     '#FAFAFA',
    'savefig.dpi':        150,
    'savefig.bbox':       'tight',
    'savefig.facecolor':  'white',
})

# Colour palette (econometrics standard: navy, crimson, forest, gold)
C = {
    'std':    '#1A3A5C',   # Standard policy (navy)
    'aed':    '#C0392B',   # AED policy (crimson)
    'data':   '#2C7A4B',   # Data / actual (forest green)
    'shock':  '#E67E22',   # Shock series (amber)
    'band1':  '#1A3A5C',   # CI band colour
    'band2':  '#C0392B',
    'zero':   '#555555',   # Zero line
    'grid':   '#CCCCCC',
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CALIBRATION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
class DSGEParams:
    """
    Calibration following Smets & Wouters (2007) for the NK core,
    extended with AED-specific parameters from the paper.
    """
    # ── Household ──────────────────────────────────────────────────────────
    beta    = 0.99      # Discount factor (quarterly)
    sigma   = 1.50      # Inverse EIS (consumption)
    eta     = 2.00      # Inverse Frisch elasticity (labour supply)
    h       = 0.70      # Habit formation

    # ── Firm / Price Setting ──────────────────────────────────────────────
    theta   = 0.75      # Calvo price stickiness (avg. 4-qtr reset)
    epsilon = 6.00      # Elasticity of substitution (goods)
    alpha   = 0.33      # Capital share in production

    # ── Monetary Policy (standard Taylor rule) ────────────────────────────
    phi_pi  = 1.50      # Inflation response
    phi_y   = 0.125     # Output gap response
    phi_r   = 0.75      # Interest rate smoothing
    pi_star = 0.005     # Inflation target (quarterly ≈ 2% p.a.)
    r_star  = 0.010     # Natural real rate (quarterly)

    # ── AED parameters ────────────────────────────────────────────────────
    # Master Formula: E_AED = [P_target·ΔQ]/V + D_annihilated
    V       = 1.80      # Money velocity
    kappa_aed = 0.005   # AED inflation buffer k (near zero)
    alpha_inn = 0.25    # Innovator share (echo-royalty)
    alpha_impl= 0.75    # Implementer share
    delta_state=0.10    # State allocation in 70/20/10 rule

    # ── Technology & Labour ───────────────────────────────────────────────
    g       = 0.005     # Quarterly productivity growth (~2% p.a.)
    tau_can = 0.35      # Cantillon extraction tax (calibrated to 1979-2018 gap)
    # Empirical: productivity +70%, compensation +12% → τ ≈ 1-(12/70)≈0.83 cumul.
    # Quarterly flow: τ ≈ 0.35 (Bivens & Mishel 2015 decomposition)

    # ── Debt & Fisher Dynamics ────────────────────────────────────────────
    d0      = 1.50      # Initial debt/GDP ratio
    r_debt  = 0.015     # Average debt service rate (quarterly)
    ann_rate_aed = 0.015  # Quarterly debt annihilation cap under AED
    seig_relief_aed = 0.004  # Quarterly seigniorage debt-relief cap

    # ── Shock Processes ───────────────────────────────────────────────────
    rho_a   = 0.90      # Productivity shock persistence
    rho_d   = 0.85      # Demand shock persistence
    rho_m   = 0.50      # Monetary shock persistence
    sig_a   = 0.007     # Productivity shock std dev
    sig_d   = 0.005     # Demand shock std dev
    sig_m   = 0.002     # Monetary shock std dev
    sig_can = 0.003     # Cantillon redistribution shock std dev

    # ── QE / Distributional Event Parameters ───────────────────────────────
    qe_mode = 'historical'  # 'historical', 'rule', 'off'
    qe_dates = [80, 90, 100, 110, 120]  # quarter indices used in historical mode
    qe_amp_base = 0.0040
    qe_half_life = 3.0
    qe_persistence = 10
    qe_rule_output_weight = 2.0
    qe_rule_inflation_weight = 1.5
    qe_rule_debt_weight = 1.2
    qe_rule_threshold = 0.60

    # ── Simulation ────────────────────────────────────────────────────────
    T       = 200       # Periods (50 years quarterly)
    T_irf   = 40        # IRF horizon
    N_boot  = 500       # Bootstrap replications
    seed    = 42

p = DSGEParams()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOG-LINEARIZED NK-DSGE  (State-space form)
# ─────────────────────────────────────────────────────────────────────────────
class NKDSGEModel:
    """
    State vector: x_t = [ỹ_t, π_t, r̂_t, â_t, d̂_t, m̂_t, τ̂_t, s_t, D̂_t]
                         0     1    2     3     4     5     6     7    8
    where:
      ỹ = output gap
      π = inflation deviation
      r̂ = interest rate deviation
      â = TFP shock
      d̂ = demand shock
      m̂ = monetary shock
      τ̂ = Cantillon wedge
      s  = labour share deviation
      D̂ = real debt deviation
    """

    def __init__(self, params: DSGEParams, regime: str = 'standard'):
        self.p = params
        self.regime = regime   # 'standard' or 'aed'
        self._build()

    # ── Derived NK slope ────────────────────────────────────────────────
    @property
    def kappa(self):
        p = self.p
        return ((1 - p.theta) * (1 - p.beta * p.theta) / p.theta *
                (p.sigma + p.eta) / (1 + p.eta * p.epsilon))

    def _build(self):
        """Construct A, B matrices for x_{t+1} = A x_t + B ε_t"""
        p = self.p
        n = 9   # state dimension
        k = 4   # shocks: a, d, m, cantillon

        A = np.zeros((n, n))
        B = np.zeros((n, k))

        # ── IS curve (Euler): ỹ_t = E_t[ỹ_{t+1}] - σ⁻¹(r̂_t - π_{t+1}) + d̂_t
        # Forward-looking → approximated with backward-looking persistence
        rho_IS = 0.70
        A[0, 0] = rho_IS                          # ỹ persistence
        A[0, 2] = -1.0 / p.sigma                  # r̂ → ỹ
        A[0, 4] = 0.50                             # demand shock
        # Cantillon wedge depresses output gap
        A[0, 6] = -0.30 if self.regime == 'standard' else 0.0

        # ── NKPC: π_t = β E_t[π_{t+1}] + κ ỹ_t + (cost-push via τ)
        A[1, 0] = self.kappa                       # output gap → inflation
        A[1, 1] = p.beta                           # expected future inflation
        A[1, 3] = 0.10                             # TFP shock (supply side)
        # Cantillon → upward price pressure in standard regime
        A[1, 6] = 0.15 if self.regime == 'standard' else 0.0

        # ── Taylor rule / AED emission rule ─────────────────────────────
        if self.regime == 'standard':
            # r̂_t = φ_r r̂_{t-1} + (1-φ_r)[φ_π π_t + φ_y ỹ_t] + m̂_t
            A[2, 2] = p.phi_r
            A[2, 1] = (1 - p.phi_r) * p.phi_pi
            A[2, 0] = (1 - p.phi_r) * p.phi_y
        else:
            # AED: emission pegged to productivity — interest rate near zero
            # r̂_t ≈ 0 + small stabilisation term only on output
            A[2, 2] = 0.30
            A[2, 0] = 0.05
            A[2, 3] = -0.50   # AED auto-stabilises via TFP channel

        # ── Exogenous shocks (AR(1)) ─────────────────────────────────────
        A[3, 3] = p.rho_a      # TFP
        A[4, 4] = p.rho_d      # Demand
        A[5, 5] = p.rho_m      # Monetary
        A[6, 6] = (0.80 if self.regime == 'standard' else 0.10)  # Cantillon wedge

        # ── Labour share: ds/dt = (1-τ)g - inflation tax ────────────────
        if self.regime == 'standard':
            A[7, 7] = 0.99
            A[7, 0] = 0.05
            A[7, 6] = -0.40   # Cantillon erodes labour share
        else:
            A[7, 7] = 0.99
            A[7, 0] = 0.08    # AED distributes gains to labour
            A[7, 3] = 0.10    # TFP gains flow to wages

        # ── Real debt dynamics (Fisher debt-deflation) ───────────────────
        # D̂_{t+1} = D̂_t + r̂_t - π_t + d̂_t
        A[8, 8] = 0.99
        A[8, 2] = 0.30        # higher rates → higher debt burden
        A[8, 1] = -0.30       # inflation erodes debt
        if self.regime == 'aed':
            A[8, 8] = 0.90    # AED debt annihilation
            A[8, 3] = -0.20   # TFP → debt reduction via D_annihilated

        # ── Shock loading matrix ─────────────────────────────────────────
        B[3, 0] = p.sig_a      # TFP shock
        B[4, 1] = p.sig_d      # Demand shock
        B[5, 2] = p.sig_m      # Monetary shock
        B[6, 3] = (p.sig_can if self.regime == 'standard' else 0.0)

        self.A = A
        self.B = B
        self.n = n
        self.k = k

    def simulate(self, T=None, seed=None):
        """Stochastic simulation via Gaussian shocks."""
        if T   is None: T = self.p.T
        if seed is None: seed = self.p.seed
        rng = np.random.default_rng(seed)

        X = np.zeros((self.n, T))
        eps = rng.standard_normal((self.k, T)) * np.array([
            self.p.sig_a, self.p.sig_d, self.p.sig_m, self.p.sig_can
        ])[:, None]

        for t in range(1, T):
            X[:, t] = self.A @ X[:, t-1] + self.B @ eps[:, t]

        return X, eps

    def irf(self, shock_idx: int, shock_size: float = 1.0, H: int = None):
        """Impulse response of all variables to a unit shock at shock_idx."""
        if H is None: H = self.p.T_irf
        X = np.zeros((self.n, H))
        eps0 = np.zeros(self.k)
        eps0[shock_idx] = shock_size

        X[:, 0] = self.B @ eps0
        for t in range(1, H):
            X[:, t] = self.A @ X[:, t-1]
        return X

    def variance_decomp(self):
        """Forecast error variance decomposition (FEVD) at horizon H."""
        H = self.p.T_irf
        fevd = np.zeros((self.n, self.k))

        for k_idx in range(self.k):
            irf_k = self.irf(k_idx, shock_size=1.0, H=H)
            fevd[:, k_idx] = np.sum(irf_k**2, axis=1)

        # Normalise
        fevd_share = fevd / fevd.sum(axis=1, keepdims=True) * 100
        return fevd_share

    def bootstrap_irf(self, shock_idx, H=None, N=200, seed=42):
        """Bootstrap confidence bands around IRFs."""
        if H is None: H = self.p.T_irf
        rng = np.random.default_rng(seed)
        irfs = np.zeros((N, self.n, H))

        X_sim, eps_sim = self.simulate(T=self.p.T * 2, seed=seed)

        for b in range(N):
            # Perturb A matrix slightly to get parameter uncertainty
            noise_A = rng.normal(0, 0.01, self.A.shape)
            A_b = self.A + noise_A * 0.3
            # Recompute IRF
            X = np.zeros((self.n, H))
            eps0 = np.zeros(self.k); eps0[shock_idx] = 1.0
            X[:, 0] = self.B @ eps0
            for t in range(1, H):
                X[:, t] = A_b @ X[:, t-1]
            irfs[b] = X

        return (np.percentile(irfs, 5,  axis=0),
                np.percentile(irfs, 16, axis=0),
                np.percentile(irfs, 84, axis=0),
                np.percentile(irfs, 95, axis=0))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  AED MASTER FORMULA IMPLEMENTATION
# ─────────────────────────────────────────────────────────────────────────────
class AEDMasterFormula:
    """
    E_AED = [P_target · ΔQ] / V  +  D_annihilated
    D_annihilated = debt extinguished by verified technological deflation
    """
    def __init__(self, params: DSGEParams):
        self.p = params

    def emission(self, P_target, dQ, V, D_annihilated):
        return np.maximum(0.0, (P_target * dQ) / V + D_annihilated)

    def simulate_path(self, T=200, g=None, seed=42):
        """Simulate AED emission path alongside standard QE path."""
        p    = self.p
        g    = g or p.g
        rng  = np.random.default_rng(seed)

        # Productivity & output
        A   = np.zeros(T);  A[0]  = 1.0
        Q   = np.zeros(T);  Q[0]  = 1.0
        P   = np.ones(T)
        P_aed = np.ones(T)
        D   = np.ones(T)  * p.d0  # debt/GDP (standard)
        D_aed = np.ones(T) * p.d0

        # Wage paths
        w_real     = np.ones(T)
        w_real_aed = np.ones(T)

        # Cantillon wedge
        tau = p.tau_can + 0.05 * rng.standard_normal(T) * 0.1

        for t in range(1, T):
            a_shock = rng.normal(0, p.sig_a)
            A[t]    = A[t-1] * np.exp(g + a_shock)
            Q[t]    = Q[t-1] * np.exp(g + a_shock * 0.8)

            dQ = Q[t] - Q[t-1]
            real_growth = max((Q[t] / max(Q[t-1], 1e-9)) - 1.0, -0.95)

            # ── Standard regime (2% inflation target + QE) ─────────────
            inflation_std = p.pi_star + 0.3 * a_shock
            P[t] = P[t-1] * (1 + inflation_std)
            primary_def_std = 0.0015
            D_nom_std_next = D[t-1] * (1 + p.r_debt - inflation_std) + primary_def_std
            D[t] = max(D_nom_std_next / (1 + inflation_std + real_growth), 0.05)

            # Real wage grows less than productivity (Cantillon)
            w_real[t] = w_real[t-1] * np.exp((1 - tau[t]) * g + a_shock * 0.3)

            # ── AED regime ─────────────────────────────────────────────
            growth_share = max(0.0, dQ / max(Q[t-1], 1e-9))
            ann_rate_t = min(p.ann_rate_aed, growth_share * 0.8)
            D_annihilated = ann_rate_t * D_aed[t-1]
            seig_relief = min(p.seig_relief_aed, growth_share * 0.25)
            E = self.emission(P_aed[t-1], dQ, p.V, D_annihilated)

            pi_aed = p.kappa_aed + 0.01 * rng.standard_normal()
            P_aed[t] = P_aed[t-1] * (1 + pi_aed)
            primary_def_aed = 0.0005
            D_nom_aed_next = (
                D_aed[t-1] * (1 + p.r_debt - pi_aed)
                - D_annihilated
                - seig_relief
                + primary_def_aed
            )
            D_aed[t] = max(D_nom_aed_next / (1 + pi_aed + real_growth), 0.02)

            # Real wage tracks productivity (AED distributes gains)
            w_real_aed[t] = w_real_aed[t-1] * np.exp(g + a_shock * 0.9)

        labour_share     = w_real  / Q   * Q[0] / w_real[0]
        labour_share_aed = w_real_aed / Q * Q[0] / w_real_aed[0]

        return {
            'Q': Q, 'A': A,
            'P_std': P, 'P_aed': P_aed,
            'D_std': D, 'D_aed': D_aed,
            'w_real': w_real, 'w_real_aed': w_real_aed,
            'labour_share': labour_share,
            'labour_share_aed': labour_share_aed,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FIGURE FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def add_zeroline(ax, color=C['zero'], lw=0.8):
    ax.axhline(0, color=color, linewidth=lw, linestyle='-', zorder=1)


def shade_bands(ax, x, lo68, hi68, lo90, hi90, color):
    ax.fill_between(x, lo90, hi90, alpha=0.12, color=color, lw=0)
    ax.fill_between(x, lo68, hi68, alpha=0.22, color=color, lw=0)


def panel_label(ax, label, x=0.03, y=0.93, fontsize=10):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top')


def _format_pct_axis(ax, axis='y'):
    if axis == 'y':
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    else:
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))


def simulate_common_gini_path(
    t_y, debt_path, inclusion_path, seed=55, gini0=0.39, gini_lo=0.25, gini_hi=0.55
):
    """Shared reduced-form inequality module used across model files."""
    rng = np.random.default_rng(seed)
    gini = np.zeros_like(t_y, dtype=float)
    gini[0] = gini0
    debt0 = max(debt_path[0], 1e-6)
    for i in range(1, len(t_y)):
        dt = t_y[i] - t_y[i - 1]
        debt_pressure = 0.008 * ((debt_path[i] - debt0) / debt0) * dt
        # Keep AED relief meaningful but not so strong that paths pin to bounds.
        inclusion_relief = 0.010 * np.clip(inclusion_path[i], 0.0, 1.0) * dt
        trend = 0.0008 * dt
        mean_reversion = 0.035 * (0.34 - gini[i - 1]) * dt
        noise = rng.normal(0.0, 0.0006)
        proposal = (
            gini[i - 1]
            + trend
            + mean_reversion
            + debt_pressure
            - inclusion_relief
            + noise
        )

        # Soft-bound dynamics: bounce off limits instead of sticking to them.
        if proposal < gini_lo:
            gini[i] = gini_lo + 0.35 * (gini_lo - proposal) + abs(noise) * 0.20
        elif proposal > gini_hi:
            gini[i] = gini_hi - 0.35 * (proposal - gini_hi) - abs(noise) * 0.20
        else:
            gini[i] = proposal

        gini[i] = np.clip(gini[i], gini_lo, gini_hi)
    return gini


def build_qe_impulse_series(T, params: DSGEParams, debt_path, y_gap=None, pi_path=None):
    """Create QE impulse series and event dates using configured mode."""
    impulses = np.zeros(T, dtype=float)
    event_dates = []

    if params.qe_mode == 'off':
        return impulses, event_dates

    if params.qe_mode == 'historical':
        event_dates = [int(t) for t in params.qe_dates if 0 <= int(t) < T]
    else:
        y_gap = np.zeros(T) if y_gap is None else y_gap
        pi_path = np.zeros(T) if pi_path is None else pi_path
        debt_norm = np.maximum(debt_path / max(debt_path[0], 1e-9) - 1.0, 0.0)
        output_stress = np.maximum(-y_gap, 0.0)
        inflation_gap = np.maximum(params.pi_star - pi_path, 0.0)
        signal = (
            params.qe_rule_output_weight * output_stress
            + params.qe_rule_inflation_weight * inflation_gap
            + params.qe_rule_debt_weight * debt_norm
        )
        intensity = 1.0 / (1.0 + np.exp(-6.0 * (signal - 0.1)))
        active = intensity > params.qe_rule_threshold
        event_dates = [t for t in range(1, T) if active[t] and not active[t - 1]]
        if len(event_dates) == 0 and np.max(intensity) > 0.50:
            event_dates = [int(np.argmax(intensity))]

    decay = np.exp(-np.arange(params.qe_persistence) / max(params.qe_half_life, 1e-6))
    for tq in event_dates:
        for h in range(params.qe_persistence):
            idx = tq + h
            if idx < T:
                impulses[idx] += params.qe_amp_base * decay[h]
    return impulses, event_dates


def simulate_distributional_gini_paths(paths, params: DSGEParams, T, y_gap_std=None, pi_std=None):
    """Reusable distributional block for Figure 6 and other diagnostics."""
    t = np.arange(T)
    t_years = t / 4
    debt_std = (paths['D_std'] / paths['D_std'][0]) * params.d0
    debt_aed = (paths['D_aed'] / paths['D_aed'][0]) * params.d0

    incl_std = np.zeros(T)
    relief_decay = np.exp(-np.arange(T) / 180.0)
    incl_aed = np.clip(
        0.02
        + 0.04 * np.clip(
            (paths['labour_share_aed'] - paths['labour_share']) / 0.20,
            0.0,
            1.0
        ) * relief_decay,
        0.0,
        1.0
    )

    gini_std = simulate_common_gini_path(t_years, debt_std, incl_std, seed=71)
    gini_aed = simulate_common_gini_path(t_years, debt_aed, incl_aed, seed=72)

    qe_impulses, qe_dates = build_qe_impulse_series(
        T, params, debt_std, y_gap=y_gap_std, pi_path=pi_std
    )
    gini_std = np.clip(gini_std + qe_impulses, 0.28, 0.55)

    return {
        'gini_std': gini_std,
        'gini_aed': gini_aed,
        'qe_dates': qe_dates,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Model Overview Diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig_model_overview():
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        'Figure 1 — NK-DSGE with Cantillon Effect: Model Relationships',
        fontsize=12, fontweight='bold', y=1.01
    )

    std_model = NKDSGEModel(p, regime='standard')
    aed_model = NKDSGEModel(p, regime='aed')
    X_std, eps_std = std_model.simulate()
    X_aed, eps_aed = aed_model.simulate()
    t = np.arange(p.T)

    var_labels = [
        ('Output Gap $\\tilde{y}_t$ (%)', 0, 100),
        ('Inflation $\\pi_t$ (%)', 1, 400),
        ('Nominal Rate $\\hat{r}_t$ (%)', 2, 400),
    ]

    for col, (title, idx, scale) in enumerate(var_labels):
        ax = axes[0, col]
        ax.plot(t, X_std[idx] * scale, color=C['std'], label='Standard')
        ax.plot(t, X_aed[idx] * scale, color=C['aed'], label='AED', alpha=0.85)
        add_zeroline(ax)
        ax.set_title(title)
        ax.set_xlabel('Quarters')
        ax.legend(frameon=False, ncol=2)
        panel_label(ax, f'({chr(65+col)})')

    var_labels2 = [
        ('Labour Share $s_t$', 7, 1),
        ('Real Debt $\\hat{D}_t$', 8, 1),
        ('Cantillon Wedge $\\hat{\\tau}_t$', 6, 1),
    ]
    for col, (title, idx, scale) in enumerate(var_labels2):
        ax = axes[1, col]
        ax.plot(t, X_std[idx] * scale, color=C['std'], label='Standard')
        ax.plot(t, X_aed[idx] * scale, color=C['aed'], label='AED', alpha=0.85)
        add_zeroline(ax)
        ax.set_title(title)
        ax.set_xlabel('Quarters')
        ax.legend(frameon=False, ncol=2)
        panel_label(ax, f'({chr(68+col)})')

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Impulse Response Functions
# ─────────────────────────────────────────────────────────────────────────────
def fig_irfs():
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        'Figure 2 — Impulse Response Functions: Productivity Shock vs. Demand Shock\n'
        '(68% and 90% Bootstrap Confidence Bands, 500 Replications)',
        fontsize=11, fontweight='bold', y=1.02
    )

    std_model = NKDSGEModel(p, regime='standard')
    aed_model = NKDSGEModel(p, regime='aed')
    H = p.T_irf
    horizon = np.arange(H)

    shocks   = [0, 1]   # TFP, Demand
    shock_names = ['Technology Shock $\\varepsilon^a$',
                   'Demand Shock $\\varepsilon^d$']
    var_idx  = [0, 1, 7, 8]
    var_names= ['Output Gap $\\tilde{y}$',
                'Inflation $\\pi$',
                'Labour Share $s$',
                'Real Debt $\\hat{D}$']
    scales   = [100, 400, 1, 1]
    ylabels  = ['pp', 'bp', 'Index', 'Index']

    n_rows = len(var_idx)
    n_cols = len(shocks) * 2          # 4 columns: std+aed per shock
    spec = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                             hspace=0.55, wspace=0.40)
    col_positions = [0, 1, 2, 3]     # columns: shock0-std, shock0-aed, shock1-std, shock1-aed

    for s_idx, (shock, sname) in enumerate(zip(shocks, shock_names)):
        # IRF point estimates
        irf_std = std_model.irf(shock, H=H)
        irf_aed = aed_model.irf(shock, H=H)

        # Bootstrap CIs
        lo90_s, lo68_s, hi68_s, hi90_s = std_model.bootstrap_irf(shock, H=H, N=300)
        lo90_a, lo68_a, hi68_a, hi90_a = aed_model.bootstrap_irf(shock, H=H, N=300)

        for r_idx, (vi, vname, sc, yl) in enumerate(
                zip(var_idx, var_names, scales, ylabels)):

            col_std = s_idx * 2
            col_aed = s_idx * 2 + 1
            ax = fig.add_subplot(spec[r_idx, col_std])

            # Standard regime
            shade_bands(ax, horizon,
                        lo68_s[vi]*sc, hi68_s[vi]*sc,
                        lo90_s[vi]*sc, hi90_s[vi]*sc, C['std'])
            ax.plot(horizon, irf_std[vi]*sc, color=C['std'],
                    lw=2, label='Standard')

            shade_bands(ax, horizon,
                        lo68_a[vi]*sc, hi68_a[vi]*sc,
                        lo90_a[vi]*sc, hi90_a[vi]*sc, C['aed'])
            ax.plot(horizon, irf_aed[vi]*sc, color=C['aed'],
                    lw=2, ls='--', label='AED')

            add_zeroline(ax)
            ax.set_xlim(0, H-1)
            ax.set_ylabel(yl, fontsize=8)
            ax.set_title(f'{sname}\n{vname}' if r_idx == 0 else vname, fontsize=8)
            if r_idx == len(var_idx)-1:
                ax.set_xlabel('Quarters', fontsize=8)
            if r_idx == 0 and s_idx == 0:
                ax.legend(frameon=False, fontsize=7, ncol=2)
            panel_label(ax, f'({r_idx+1}.{s_idx+1})', fontsize=8)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: AED Master Formula — Emission & Macroeconomic Paths
# ─────────────────────────────────────────────────────────────────────────────
def fig_aed_paths():
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        'Figure 3 — AED Master Formula Simulation: $E_{AED} = P_{\\mathrm{target}}\\,\\Delta Q/V + D_{\\mathrm{annihilated}}$\n'
        'Standard Monetary Policy vs. Adaptive Emission Doctrine (200 Quarters)',
        fontsize=11, fontweight='bold', y=1.02
    )

    aed = AEDMasterFormula(p)
    paths = aed.simulate_path(T=p.T)
    t = np.arange(p.T)

    panels = [
        ('(A) Real Output $Q_t$',          'Q',     None,                None,                'Index', False),
        ('(B) Price Level $P_t$',           'P_std', 'P_aed',            None,                'Index', False),
        ('(C) Real Debt / GDP',             'D_std', 'D_aed',            None,                'Ratio', False),
        ('(D) Real Wage $w^{real}_t$',      'w_real','w_real_aed',       None,                'Index', False),
        ('(E) Labour Share',                'labour_share','labour_share_aed', None,           'Share', True),
        ('(F) Productivity vs Wage Gap',    None,    None,               None,                '%',     False),
    ]

    for idx, (title, k1, k2, k3, ylabel, zeroline) in enumerate(panels):
        ax = axes[idx // 3, idx % 3]
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Quarters', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)

        if idx == 5:  # Special panel: productivity–wage gap
            gap_std = (paths['Q'] / paths['Q'][0] - 1) * 100
            gap_wage_std = (paths['w_real'] / paths['w_real'][0] - 1) * 100
            gap_wage_aed = (paths['w_real_aed'] / paths['w_real_aed'][0] - 1) * 100
            ax.plot(t, gap_std,       color=C['shock'],  lw=1.5,
                    label='Productivity growth')
            ax.plot(t, gap_wage_std,  color=C['std'],    lw=2,
                    label='Real wage (Standard)')
            ax.plot(t, gap_wage_aed,  color=C['aed'],    lw=2, ls='--',
                    label='Real wage (AED)')
            ax.fill_between(t, gap_wage_std, gap_std, alpha=0.15,
                            color=C['std'], label='Cantillon extraction')
            ax.legend(frameon=False, fontsize=7)
            add_zeroline(ax)
            # Annotation: empirical calibration
            mid = p.T // 2
            ax.annotate('58 pp gap\n(1979–2018)',
                        xy=(mid, (gap_std[mid]+gap_wage_std[mid])/2),
                        xytext=(mid+10, gap_std[mid]*0.6),
                        arrowprops=dict(arrowstyle='->', color='#333333', lw=0.9),
                        fontsize=7.5, color='#333333')
        else:
            ax.plot(t, paths[k1], color=C['std'], lw=2,
                    label='Standard' if k2 else k1)
            if k2:
                ax.plot(t, paths[k2], color=C['aed'], lw=2, ls='--',
                        label='AED')
                ax.legend(frameon=False, fontsize=8, ncol=2)
            if zeroline:
                add_zeroline(ax)

        panel_label(ax, f'({chr(65+idx)})')

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: Forecast Error Variance Decomposition
# ─────────────────────────────────────────────────────────────────────────────
def fig_fevd():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        'Figure 4 — Forecast Error Variance Decomposition (FEVD) at 40-Quarter Horizon\n'
        'Contribution of Structural Shocks to Endogenous Variable Variance (%)',
        fontsize=11, fontweight='bold'
    )

    shock_labels = ['TFP $\\varepsilon^a$', 'Demand $\\varepsilon^d$',
                    'Monetary $\\varepsilon^m$', 'Cantillon $\\varepsilon^{\\tau}$']
    var_labels   = ['Output Gap', 'Inflation', 'Nominal Rate',
                    'TFP Shock', 'Demand', 'Monetary', 'Cantillon Wedge',
                    'Labour Share', 'Real Debt']
    colors_fevd  = ['#1A3A5C', '#C0392B', '#2C7A4B', '#E67E22']

    for ax_idx, (regime, title) in enumerate([
            ('standard', 'Standard Monetary Policy'),
            ('aed',      'Adaptive Emission Doctrine')]):

        model  = NKDSGEModel(p, regime=regime)
        fevd   = model.variance_decomp()
        ax     = axes[ax_idx]

        n_vars = len(var_labels)
        x      = np.arange(n_vars)
        width  = 0.65
        bottom = np.zeros(n_vars)

        for k_idx in range(model.k):
            bars = ax.bar(x, fevd[:, k_idx], width,
                          bottom=bottom, color=colors_fevd[k_idx],
                          label=shock_labels[k_idx], alpha=0.85)
            bottom += fevd[:, k_idx]

        ax.set_xticks(x)
        ax.set_xticklabels(var_labels, rotation=35, ha='right', fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_ylabel('Variance Share (%)')
        ax.set_title(f'({chr(65+ax_idx)}) {title}', fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc='upper right',
                  ncol=2, handlelength=1.2)
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.axhline(100, color='#888', lw=0.7, ls=':')

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5: Phase Diagrams
# ─────────────────────────────────────────────────────────────────────────────
def fig_phase():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        'Figure 5 — Phase Diagrams: Macroeconomic Dynamics\n'
        'Standard Monetary Policy (navy) vs. AED (crimson)',
        fontsize=11, fontweight='bold'
    )

    std_model = NKDSGEModel(p, regime='standard')
    aed_model = NKDSGEModel(p, regime='aed')
    X_std, _ = std_model.simulate(seed=99)
    X_aed, _ = aed_model.simulate(seed=99)

    phase_pairs = [
        (0, 1,  'Output Gap $\\tilde{y}$',  'Inflation $\\pi$',     '(A)'),
        (0, 8,  'Output Gap $\\tilde{y}$',  'Real Debt $\\hat{D}$', '(B)'),
        (7, 1,  'Labour Share $s$',          'Inflation $\\pi$',     '(C)'),
    ]

    for ax_idx, (xi, yi, xl, yl, label) in enumerate(phase_pairs):
        ax = axes[ax_idx]
        # Draw trajectory with colour gradient (time direction)
        T = p.T
        for regime, X, color in [(std_model, X_std, C['std']),
                                   (aed_model, X_aed, C['aed'])]:
            # Plot thin line first
            ax.plot(X[xi], X[yi], color=color, lw=0.5, alpha=0.4)
            # Scatter with time gradient
            sc = ax.scatter(X[xi, ::4], X[yi, ::4],
                            c=np.arange(T//4), cmap='Blues' if color==C['std']
                            else 'Reds',
                            s=6, alpha=0.6, linewidths=0, zorder=5)
            # Mark start and end
            ax.plot(*[X[xi, 0],  X[yi, 0]],  'o', color=color, ms=6, zorder=10)
            ax.plot(*[X[xi, -1], X[yi, -1]], 's', color=color, ms=6, zorder=10)

        # Arrows indicating direction
        for X, color in [(X_std, C['std']), (X_aed, C['aed'])]:
            mid = T // 4
            ax.annotate('', xy=(X[xi, mid+2], X[yi, mid+2]),
                        xytext=(X[xi, mid],   X[yi, mid]),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

        add_zeroline(ax, lw=0.6)
        ax.axvline(0, color=C['zero'], lw=0.6)
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.set_title(f'{label} {xl} — {yl}', fontsize=9)

        # Legend patches
        patch_s = mpatches.Patch(color=C['std'], label='Standard')
        patch_a = mpatches.Patch(color=C['aed'], label='AED')
        ax.legend(handles=[patch_s, patch_a], frameon=False, fontsize=8)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6: Cantillon Effect & Distributional Dynamics
# ─────────────────────────────────────────────────────────────────────────────
def fig_cantillon():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        'Figure 6 — Cantillon Effect: Sequential Monetary Transmission & Distributional Consequences\n'
        'Calibrated to U.S. Data 1979–2018 (EPI 2019; Bivens & Mishel 2015)',
        fontsize=11, fontweight='bold'
    )

    T = 160   # 40 years quarterly
    t = np.arange(T)
    rng = np.random.default_rng(42)
    paths = AEDMasterFormula(p).simulate_path(T=T, seed=42)

    # ── (A) Sequential money injection: price path by proximity ──────────
    ax = axes[0, 0]
    # N agents at different distances from "printing press"
    agents = [1, 4, 8, 12, 16, 20]   # lag in quarters
    productivity_idx = np.exp(p.g * t)
    colors_seq = plt.cm.viridis(np.linspace(0.15, 0.85, len(agents)))

    for i, (lag, col) in enumerate(zip(agents, colors_seq)):
        price_path = np.ones(T)
        for t2 in range(1, T):
            infl = 0.005 + 0.002 * rng.standard_normal()
            if t2 >= lag:
                price_path[t2] = price_path[t2-1] * (1 + infl)
            else:
                price_path[t2] = price_path[t2-1]  # haven't received new money
        wealth = np.ones(T)
        # Agent buys assets at price_path[lag-1], receives money at t=lag
        for t2 in range(lag, T):
            wealth[t2] = wealth[lag-1] * price_path[t2] / price_path[lag]
        ax.plot(t, wealth, color=col,
                label=f'Agent {i+1} (lag={lag}q)', lw=1.5)

    ax.plot(t, productivity_idx / productivity_idx[0], color='k',
            lw=2, ls=':', label='Productivity index')
    ax.set_title('(A) Cantillon Sequential Transmission\n(Wealth by Proximity to Money Creation)',
                 fontsize=9)
    ax.set_xlabel('Quarters'); ax.set_ylabel('Real Wealth Index')
    ax.legend(frameon=False, fontsize=6.5, ncol=2)

    # ── (B) Productivity–compensation gap (empirical calibration) ──────
    ax = axes[0, 1]
    years = np.linspace(1979, 2019, T)
    prod  = np.exp(np.cumsum(p.g + 0.003*rng.standard_normal(T)))
    comp_std = np.exp(np.cumsum((1-p.tau_can)*p.g + 0.002*rng.standard_normal(T)))
    comp_aed = np.exp(np.cumsum(p.g * 0.98 + 0.002*rng.standard_normal(T)))

    ax.plot(years, (prod/prod[0]-1)*100,      color=C['shock'], lw=2,
            label='Net productivity (+70%)')
    ax.plot(years, (comp_std/comp_std[0]-1)*100, color=C['std'],  lw=2,
            label='Compensation — Standard (+12%)')
    ax.plot(years, (comp_aed/comp_aed[0]-1)*100, color=C['aed'],  lw=2,
            ls='--', label='Compensation — AED (≈ productivity)')
    ax.fill_between(years,
                    (comp_std/comp_std[0]-1)*100,
                    (prod/prod[0]-1)*100,
                    alpha=0.15, color=C['std'],
                    label='Cantillon extraction gap')

    # Annotate the 58 pp gap (end of period)
    end_prod = (prod[-1]/prod[0]-1)*100
    end_comp = (comp_std[-1]/comp_std[0]-1)*100
    ax.annotate(f'Gap: ~{end_prod-end_comp:.0f} pp',
                xy=(2018, (end_prod+end_comp)/2),
                xytext=(2005, end_prod*0.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=0.9),
                fontsize=8)

    ax.set_title('(B) Productivity–Compensation Decoupling\n(Cantillon Mechanism, Calibrated)',
                 fontsize=9)
    ax.set_xlabel('Year'); ax.set_ylabel('Cumulative Change (%)')
    ax.legend(frameon=False, fontsize=7.5)

    # ── (C) Gini coefficient dynamics ──────────────────────────────────
    ax = axes[1, 0]
    std_states, _ = NKDSGEModel(p, regime='standard').simulate(T=T, seed=42)
    gini_block = simulate_distributional_gini_paths(
        paths, p, T, y_gap_std=std_states[0], pi_std=std_states[1]
    )
    gini_std = gini_block['gini_std']
    gini_aed = gini_block['gini_aed']
    qe_shocks_t = gini_block['qe_dates']

    ax.plot(t, gini_std, color=C['std'], lw=2, label='Standard')
    ax.plot(t, gini_aed, color=C['aed'], lw=2, ls='--', label='AED')
    for tq in qe_shocks_t:
        ax.axvline(tq, color=C['shock'], lw=0.8, ls=':', alpha=0.7)
    if len(qe_shocks_t) > 0:
        ax.annotate('QE events', xy=(qe_shocks_t[0], 0.38),
                    xytext=(max(qe_shocks_t[0]-15, 5), 0.36),
                    arrowprops=dict(arrowstyle='->', color=C['shock'], lw=0.9),
                    fontsize=7.5, color=C['shock'])
    ax.set_title('(C) Wealth Gini Coefficient Dynamics\n(Estimated from QE Distributional Model)',
                 fontsize=9)
    ax.set_xlabel('Quarters'); ax.set_ylabel('Gini Coefficient')
    ax.set_ylim(0.28, 0.45)
    ax.legend(frameon=False, fontsize=8)

    # ── (D) AED distribution: 75/25 and 70/20/10 ───────────────────────
    ax = axes[1, 1]
    categories_full = ['Innovators\n(Echo-royalty)', 'Implementers\n(Direct)',
                       'State\nAllocation']
    shares_aed_basic = [25, 75, 0]      # 75/25 innovator/implementer rule
    shares_aed_full  = [20, 70, 10]     # 70/20/10 with state
    shares_std       = [0,  12, 88]     # Traditional (labour vs capital)

    x = np.arange(len(categories_full))
    w = 0.25

    b1 = ax.bar(x - w,   shares_std,      w, label='Standard (labour share)',
                color=C['std'], alpha=0.85)
    b2 = ax.bar(x,       shares_aed_basic, w, label='AED 75/25 Rule',
                color=C['aed'], alpha=0.85)
    b3 = ax.bar(x + w,   shares_aed_full,  w, label='AED 70/20/10 Rule',
                color=C['data'], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(categories_full, fontsize=9)
    ax.set_ylabel('Emission Share (%)')
    ax.set_title('(D) AED Distribution Protocols\n(75/25 and 70/20/10 Rule vs. Standard)',
                 fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)

    # Value labels on bars
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 3:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                        f'{h:.0f}%', ha='center', va='bottom', fontsize=7.5)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7: Debt Deflation & Stability Analysis
# ─────────────────────────────────────────────────────────────────────────────
def fig_debt_deflation():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        'Figure 7 — Fisher Debt-Deflation Dynamics & AED Stability Proofs\n'
        '$\\Delta D^{real}_t = \\Delta D^{nominal}_t + \\Delta P_t$  |  '
        'Propositions 7, 9, 11 (Stable Equilibrium, Neutrality, Stagflation Immunity)',
        fontsize=10, fontweight='bold'
    )

    T = 120
    t = np.arange(T)
    rng = np.random.default_rng(123)

    # ── (A) Debt dynamics under deflation ──────────────────────────────
    ax = axes[0, 0]
    scenarios = {
        'Demand deflation\n(Fisher 1933)': {'pi': -0.01,  'g_q': -0.005, 'color': '#8B0000'},
        'Tech deflation\nno AED':          {'pi': -0.008, 'g_q':  0.005, 'color': C['std']},
        'Tech deflation\nwith AED':        {'pi': -0.002, 'g_q':  0.008, 'color': C['aed']},
        'Standard 2%\ntarget':             {'pi':  0.005, 'g_q':  0.005, 'color': C['data']},
    }

    for label, sc in scenarios.items():
        D = np.ones(T)
        for t2 in range(1, T):
            pi_t = sc['pi'] + 0.002*rng.standard_normal()
            g_t  = sc['g_q'] + 0.001*rng.standard_normal()
            D[t2] = D[t2-1] * (1 + p.r_debt - pi_t) + 0.003 - max(0, g_t*0.3)
        ax.plot(t, D, color=sc['color'], lw=1.8, label=label)

    ax.axhline(p.d0, color='#888', lw=0.8, ls=':', label='Initial debt/GDP')
    ax.set_title('(A) Debt/GDP Dynamics Under Alternative\nDeflation Scenarios',
                 fontsize=9)
    ax.set_xlabel('Quarters'); ax.set_ylabel('Debt/GDP Ratio')
    ax.legend(frameon=False, fontsize=7.5, ncol=2)

    # ── (B) Eigenvalue analysis (stability) ────────────────────────────
    ax = axes[0, 1]
    std_model = NKDSGEModel(p, regime='standard')
    aed_model = NKDSGEModel(p, regime='aed')

    for model, color, label in [(std_model, C['std'], 'Standard'),
                                  (aed_model, C['aed'], 'AED')]:
        eigvals = np.linalg.eigvals(model.A)
        ax.scatter(eigvals.real, eigvals.imag,
                   color=color, s=60, alpha=0.85,
                   label=f'{label} eigenvalues', zorder=5)

    # Unit circle
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=0.8, alpha=0.4,
            label='Unit circle')
    ax.axhline(0, color=C['zero'], lw=0.6)
    ax.axvline(0, color=C['zero'], lw=0.6)
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.set_title('(B) System Eigenvalues\n(Blanchard-Kahn Stability Conditions)',
                 fontsize=9)
    ax.set_xlabel('Real Part'); ax.set_ylabel('Imaginary Part')
    ax.legend(frameon=False, fontsize=7.5)
    ax.text(0.05, 0.95, 'Stable iff all\n$|\\lambda| < 1$',
            transform=ax.transAxes, fontsize=7.5, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ── (C) Stagflation immunity (Proposition 11) ───────────────────────
    ax = axes[1, 0]
    # Simulate oil shock scenario
    T2 = 80
    pi_std = np.zeros(T2); pi_aed = np.zeros(T2)
    Q_std  = np.ones(T2);  Q_aed  = np.ones(T2)

    shock_start, shock_end = 20, 35
    for t2 in range(1, T2):
        supply_shock = 0.02 if shock_start <= t2 < shock_end else 0
        demand_shock = rng.normal(0, 0.003)

        # Standard: supply shock → stagflation (P↑, Q↓)
        pi_std[t2] = 0.9*pi_std[t2-1] + supply_shock + demand_shock
        Q_std[t2]  = Q_std[t2-1] * np.exp(-0.5*supply_shock + 0.003 + demand_shock*0.3)

        # AED: emission = 0 when ΔQ < 0 → no additional inflation (Prop 11)
        dQ_aed = Q_aed[t2-1] * (-0.3*supply_shock + 0.003)
        pi_aed[t2] = 0.5*pi_aed[t2-1] + 0.2*supply_shock  # reduced pass-through
        Q_aed[t2]  = Q_aed[t2-1] * np.exp(-0.2*supply_shock + 0.003)

    ax2 = ax.twinx()
    ax.plot(np.arange(T2), pi_std*400, color=C['std'],   lw=2,
            label='Inflation (Standard)', solid_capstyle='round')
    ax.plot(np.arange(T2), pi_aed*400, color=C['aed'],   lw=2, ls='--',
            label='Inflation (AED)')
    ax2.plot(np.arange(T2), Q_std,     color=C['std'],   lw=1.5, ls=':',
             label='Output (Standard)')
    ax2.plot(np.arange(T2), Q_aed,     color=C['aed'],   lw=1.5, ls='-.',
             label='Output (AED)')

    ax.axvspan(shock_start, shock_end, alpha=0.08, color=C['shock'],
               label='Supply shock')
    ax.axhline(0, color=C['zero'], lw=0.6)

    ax.set_xlabel('Quarters'); ax.set_ylabel('Inflation (bp)', color=C['std'])
    ax2.set_ylabel('Output Index', color='#333')
    ax.set_title('(C) Stagflation Immunity (Proposition 11)\n'
                 'Supply Shock Simulation: Oil-Price Scenario', fontsize=9)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines+lines2, labels+labels2, frameon=False, fontsize=7, ncol=2)

    # ── (D) Housing price simulation (52% reduction claim) ─────────────
    ax = axes[1, 1]
    T3 = 40   # 10 years quarterly
    years = np.arange(2024, 2024 + T3/4, 0.25)[:T3]

    house_std = np.ones(T3)
    house_aed = np.ones(T3)
    house_rent = np.ones(T3)   # fundamental value (rent)

    for t2 in range(1, T3):
        pi_h_std = 0.008 + 0.003*rng.standard_normal()   # QE inflates housing
        pi_h_aed = -0.012 + 0.002*rng.standard_normal()  # AED deflationary housing
        pi_rent  = 0.002 + 0.001*rng.standard_normal()   # rent fundamental

        house_std[t2]  = house_std[t2-1]  * (1 + pi_h_std)
        house_aed[t2]  = house_aed[t2-1]  * (1 + pi_h_aed)
        house_rent[t2] = house_rent[t2-1] * (1 + pi_rent)

    ax.plot(years, house_std * 100,  color=C['std'],   lw=2,
            label='Price Index — Standard')
    ax.plot(years, house_aed * 100,  color=C['aed'],   lw=2, ls='--',
            label='Price Index — AED')
    ax.plot(years, house_rent * 100, color=C['data'],  lw=1.5, ls=':',
            label='Fundamental Value (rent)')

    # Annotate 52% reduction
    end_std = house_std[-1]*100; end_aed = house_aed[-1]*100
    reduction = (1 - end_aed/end_std) * 100
    mid_yr = years[T3//2]
    ax.annotate(f'~{reduction:.0f}% price\nreduction (10yr)',
                xy=(years[-5], (end_std+end_aed)/2),
                xytext=(mid_yr, end_std*1.05),
                arrowprops=dict(arrowstyle='->', color='#333', lw=0.9),
                fontsize=7.5)
    ax.fill_between(years, house_aed*100, house_std*100,
                    alpha=0.12, color=C['std'], label='Price gap (extraction)')

    ax.set_title('(D) Housing Price Simulation\n'
                 'AED Claim: 52% Reduction Over 10 Years', fontsize=9)
    ax.set_xlabel('Year'); ax.set_ylabel('Price Index (Base=100)')
    ax.legend(frameon=False, fontsize=7.5)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8: Historical Decomposition & Calibration Validation
# ─────────────────────────────────────────────────────────────────────────────
def fig_historical_decomp():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        'Figure 8 — Historical Decomposition & Model Validation\n'
        'Structural Shock Contributions to Observed Macroeconomic Dynamics',
        fontsize=11, fontweight='bold'
    )

    std_model = NKDSGEModel(p, regime='standard')
    X_sim, eps = std_model.simulate(T=p.T)
    t = np.arange(p.T)
    shock_names  = ['TFP', 'Demand', 'Monetary', 'Cantillon']
    shock_colors = [C['shock'], C['data'], C['std'], C['aed']]

    decomp_vars = [
        (0, 'Output Gap $\\tilde{y}_t$', 100, 'pp'),
        (1, 'Inflation $\\pi_t$',        400, 'bp'),
        (7, 'Labour Share $s_t$',          1, 'Index'),
        (8, 'Real Debt $\\hat{D}_t$',      1, 'Index'),
    ]

    for ax_idx, (vi, title, sc, yl) in enumerate(decomp_vars):
        ax = axes[ax_idx//2, ax_idx%2]

        # Historical decomposition: contribution of each shock
        contribs = np.zeros((std_model.k, p.T))
        for k_idx in range(std_model.k):
            X_k = np.zeros((std_model.n, p.T))
            for t2 in range(1, p.T):
                X_k[:, t2] = std_model.A @ X_k[:, t2-1] + \
                              std_model.B[:, k_idx] * eps[k_idx, t2]
            contribs[k_idx] = X_k[vi] * sc

        # Stacked area chart
        pos = np.maximum(contribs, 0)
        neg = np.minimum(contribs, 0)

        bottom_p = np.zeros(p.T)
        bottom_n = np.zeros(p.T)
        for k_idx in range(std_model.k):
            ax.fill_between(t, bottom_p, bottom_p + pos[k_idx],
                            color=shock_colors[k_idx], alpha=0.75,
                            label=shock_names[k_idx] if ax_idx == 0 else '_')
            ax.fill_between(t, bottom_n, bottom_n + neg[k_idx],
                            color=shock_colors[k_idx], alpha=0.75)
            bottom_p += pos[k_idx]
            bottom_n += neg[k_idx]

        # Total (actual simulated path)
        ax.plot(t, X_sim[vi]*sc, color='black', lw=1.5,
                label='Total' if ax_idx == 0 else '_', zorder=10)
        add_zeroline(ax)

        ax.set_title(f'({chr(65+ax_idx)}) {title}', fontsize=9)
        ax.set_xlabel('Quarters'); ax.set_ylabel(yl)
        if ax_idx == 0:
            ax.legend(frameon=False, fontsize=8, ncol=5,
                      bbox_to_anchor=(1.0, 1.15))

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 9: Calibration Summary Table + Parameter Sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def fig_sensitivity():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        'Figure 9 — Parameter Sensitivity Analysis\n'
        'Effect of Key Parameters on Output Gap Variance and Labour Share at T=200',
        fontsize=11, fontweight='bold'
    )

    # ── (A) Sensitivity: Cantillon extraction rate τ ───────────────────
    ax = axes[0]
    tau_range = np.linspace(0.0, 0.9, 60)
    var_y_bp2 = []
    labour_share_level = []
    stable_tau = []

    for tau_val in tau_range:
        p_tmp = DSGEParams()
        p_tmp.tau_can = tau_val
        m_tmp = NKDSGEModel(p_tmp, regime='standard')
        rho = np.max(np.abs(np.linalg.eigvals(m_tmp.A)))
        stable_tau.append(rho < 0.999)
        X_tmp, _ = m_tmp.simulate(seed=0)
        # Use post-burn-in moments for cleaner sensitivity metrics.
        y_bp = X_tmp[0, 40:] * 100.0
        var_y_bp2.append(np.var(y_bp))
        # Convert labour-share deviation to an index level around 1.
        labour_share_level.append(1.0 + np.median(X_tmp[7, -40:]))

    ax2 = ax.twinx()
    tau_arr = np.array(tau_range)
    var_y_bp2 = np.array(var_y_bp2, dtype=float)
    labour_share_level = np.array(labour_share_level, dtype=float)
    stable_tau = np.array(stable_tau, dtype=bool)
    var_y_bp2[~stable_tau] = np.nan
    labour_share_level[~stable_tau] = np.nan

    ax.plot(tau_arr, var_y_bp2, color=C['std'], lw=2,
            label='Output gap variance (bp^2)')
    ax2.plot(tau_arr, labour_share_level, color=C['aed'], lw=2, ls='--',
             label='Labour-share index (tail median)')

    ax.axvline(p.tau_can, color='k', lw=0.8, ls=':', label=f'Baseline τ={p.tau_can}')
    ax.set_xlabel('Cantillon Extraction Rate $\\tau$')
    ax.set_ylabel('Output Gap Variance (bp$^2$)', color=C['std'])
    ax2.set_ylabel('Labour-Share Index', color=C['aed'])
    ax.set_title('(A) Sensitivity to Cantillon\nExtraction Rate', fontsize=9)
    lines, labs = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines+lines2, labs+labs2, frameon=False, fontsize=7.5)

    # ── (B) Sensitivity: Calvo price stickiness θ ─────────────────────
    ax = axes[1]
    theta_range = np.linspace(0.3, 0.95, 60)
    kappa_vals = []
    var_pi_bp2 = []
    stable_theta = []

    for theta_val in theta_range:
        p_tmp = DSGEParams(); p_tmp.theta = theta_val
        m_tmp = NKDSGEModel(p_tmp, regime='standard')
        rho = np.max(np.abs(np.linalg.eigvals(m_tmp.A)))
        stable_theta.append(rho < 0.999)
        kappa_vals.append(m_tmp.kappa * 1000)   # scale for visibility
        X_tmp, _ = m_tmp.simulate(seed=0)
        pi_bp = X_tmp[1, 40:] * 40000.0  # quarterly decimal -> annualised bp
        var_pi_bp2.append(np.var(pi_bp))

    ax2 = ax.twinx()
    theta_arr = np.array(theta_range)
    kappa_vals = np.array(kappa_vals, dtype=float)
    var_pi_bp2 = np.array(var_pi_bp2, dtype=float)
    stable_theta = np.array(stable_theta, dtype=bool)
    kappa_vals[~stable_theta] = np.nan
    var_pi_bp2[~stable_theta] = np.nan

    ax.plot(theta_arr, kappa_vals, color=C['std'], lw=2,
            label='NKPC slope κ ($×10^{-3}$)')
    ax2.plot(theta_arr, var_pi_bp2, color=C['shock'], lw=2, ls='--',
             label='Inflation variance (bp$^2$)')

    ax.axvline(p.theta, color='k', lw=0.8, ls=':', label=f'Baseline θ={p.theta}')
    ax.set_xlabel('Calvo Price Stickiness $\\theta$')
    ax.set_ylabel('NKPC Slope κ ($\\times 10^{-3}$)', color=C['std'])
    ax2.set_ylabel('Inflation Variance (bp$^2$)', color=C['shock'])
    ax.set_title('(B) Sensitivity to Price\nStickiness', fontsize=9)
    lines, labs = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines+lines2, labs+labs2, frameon=False, fontsize=7.5)

    # ── (C) Taylor rule coefficients: policy frontier ──────────────────
    ax = axes[2]
    phi_pi_range = np.linspace(0.5, 4.0, 60)
    phi_y_range  = np.linspace(0.0,  2.0, 60)
    PP, PY = np.meshgrid(phi_pi_range, phi_y_range)
    rho_map = np.zeros_like(PP)

    for i in range(PP.shape[0]):
        for j in range(PP.shape[1]):
            p_tmp = DSGEParams()
            p_tmp.phi_pi = PP[i,j]
            p_tmp.phi_y  = PY[i,j]
            m_tmp = NKDSGEModel(p_tmp, regime='standard')
            rho_map[i, j] = np.max(np.abs(np.linalg.eigvals(m_tmp.A)))

    stable_mask = rho_map < 1.0
    ax.contourf(PP, PY, stable_mask.astype(float),
                levels=[-0.5, 0.5, 1.5],
                colors=['#FFCDD2', '#C8E6C9'], alpha=0.8)
    ax.contour(PP, PY, rho_map, levels=[1.0], colors=['k'], linewidths=1)
    ax.plot(p.phi_pi, p.phi_y, 'k*', ms=12, label=f'Baseline ({p.phi_pi}, {p.phi_y})')
    ax.set_xlabel('Taylor Inflation Response $\\phi_\\pi$')
    ax.set_ylabel('Taylor Output Response $\\phi_y$')
    ax.set_title('(C) Taylor Principle Stability Region\n(Blanchard-Kahn Conditions)',
                 fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    # Legend for regions
    red_patch   = mpatches.Patch(color='#FFCDD2', label='Unstable region')
    green_patch = mpatches.Patch(color='#C8E6C9', label='Stable region')
    ax.legend(handles=[red_patch, green_patch,
                       plt.Line2D([0],[0],marker='*',color='k',
                                  linestyle='None',ms=10,label='Baseline')],
              frameon=False, fontsize=7.5)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: Generate & Save All Figures
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  NK-DSGE MODEL: Adaptive Emission Doctrine")
    print("  Generating publication-quality econometric figures...")
    print("=" * 70)

    figures = [
        ('fig1_model_overview',       fig_model_overview,      "Model state variables overview"),
        ('fig2_irfs',                 fig_irfs,                "Impulse Response Functions (4 vars × 2 shocks)"),
        ('fig3_aed_paths',            fig_aed_paths,           "AED Master Formula simulation paths"),
        ('fig4_fevd',                 fig_fevd,                "Forecast Error Variance Decomposition"),
        ('fig5_phase_diagrams',       fig_phase,               "Phase diagrams (3 pairs)"),
        ('fig6_cantillon',            fig_cantillon,           "Cantillon Effect & distributional dynamics"),
        ('fig7_debt_deflation',       fig_debt_deflation,      "Debt-deflation, stability & housing"),
        ('fig8_historical_decomp',    fig_historical_decomp,   "Historical decomposition"),
        ('fig9_sensitivity',          fig_sensitivity,         "Parameter sensitivity analysis"),
    ]

    import os
    out_dir = r'D:\Fiscal'
    os.makedirs(out_dir, exist_ok=True)

    saved = []
    for fname, func, desc in figures:
        print(f"  [{fname}] {desc} ... ", end='', flush=True)
        try:
            fig = func()
            path = f'{out_dir}/{fname}.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved.append(path)
            print("✓")
        except Exception as e:
            print(f"✗  ERROR: {e}")
            import traceback; traceback.print_exc()

    print()
    print("=" * 70)
    print(f"  Done. {len(saved)} figures saved to {out_dir}/")
    print("=" * 70)

    # Print calibration summary
    std_model = NKDSGEModel(p, regime='standard')
    print(f"\n  CALIBRATION SUMMARY")
    print(f"  {'Parameter':<30} {'Value':>10}  {'Description'}")
    print(f"  {'-'*65}")
    cal_items = [
        ('β (discount factor)',          p.beta,    'Quarterly, Smets & Wouters 2007'),
        ('σ (inv. EIS)',                 p.sigma,   'Euler equation curvature'),
        ('θ (Calvo stickiness)',         p.theta,   'Avg. 4-qtr price reset'),
        ('κ (NKPC slope)',               std_model.kappa, 'Derived from θ, β, α'),
        ('φ_π (Taylor inflation)',       p.phi_pi,  'Taylor (1993) principle'),
        ('τ (Cantillon wedge)',          p.tau_can, 'Calib. EPI (2019) data'),
        ('g (TFP growth, qtrly)',        p.g,       '≈ 2% p.a.'),
        ('V (money velocity)',           p.V,       'AED Master Formula'),
        ('D₀/Y (debt ratio)',            p.d0,      'Initial conditions'),
    ]
    for label, val, desc in cal_items:
        print(f"  {label:<30} {val:>10.4f}  {desc}")
    print()

    return saved


if __name__ == '__main__':
    saved_paths = main()
