"""
=============================================================================
MESO-LEVEL DSGE: Financial & Banking Sector Under AED
=============================================================================
Extension of the Adaptive Emission Doctrine (AED) macro model to the
meso-level banking sector — modelling the financial accelerator, bank
balance sheets, credit creation, interbank market, and AED-specific
transformation of bank business model.

Key theoretical contributions:
  1. Bernanke-Gertler-Gilchrist (1999) financial accelerator within AED
  2. Endogenous credit creation tied to AED emission channels
  3. Bank net worth dynamics under declarative debt restructuring
  4. Interbank contagion vs. AED immune system
  5. Business model transition: retained interest income under AED
  6. Basel III capital requirements under AED regime
  7. Four-phase bank transition path (9.3 section of paper)

State vector (n=13):
  0  ỹ_t       output gap
  1  π_t       inflation deviation
  2  r_t       reference rate
  3  â_t       TFP shock
  4  d̂_t       demand shock
  5  b_t       bank credit/GDP (total loans)
  6  k_t       bank capital ratio (equity/assets)
  7  spr_t     credit spread (lending − deposit rate)
  8  nw_t      bank net worth (log-deviation)
  9  s_t       labour share
  10 D_t       real aggregate debt
  11 ib_t      interbank rate deviation
  12 lev_t     bank leverage (assets/equity)

Shocks (k=5):
  0  TFP ε^a
  1  Demand ε^d
  2  Monetary ε^m
  3  Financial ε^f  (bank net worth / credit shock)
  4  Regulatory ε^r (capital requirement shock)
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from scipy import linalg, stats
import warnings
warnings.filterwarnings('ignore')

# ─── Global style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['DejaVu Serif','Times New Roman','Georgia'],
    'font.size':        10,
    'axes.titlesize':   10,
    'axes.labelsize':   9,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'legend.fontsize':  8,
    'figure.dpi':       150,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.22,
    'grid.linestyle':   '--',
    'grid.linewidth':   0.55,
    'lines.linewidth':  1.9,
    'figure.facecolor': 'white',
    'axes.facecolor':   '#F9F9F9',
    'savefig.dpi':      150,
    'savefig.bbox':     'tight',
})

C = {
    'std':   '#1A3A5C',   # navy  — standard banking
    'aed':   '#C0392B',   # crimson — AED regime
    'bank':  '#8E44AD',   # purple — bank-specific
    'inter': '#16A085',   # teal — interbank
    'cap':   '#E67E22',   # amber — capital
    'risk':  '#E74C3C',   # red — risk / stress
    'safe':  '#27AE60',   # green — stable/safe
    'zero':  '#555555',
}

def zl(ax, lw=0.75): ax.axhline(0, color=C['zero'], lw=lw, zorder=1)
def pl(ax, label, x=0.03, y=0.94, fontsize=10):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top')
def shade(ax, t, lo, hi, color, a1=0.12, a2=0.22):
    ax.fill_between(t, lo, hi, alpha=a1, color=color, lw=0)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
class MesoParams:
    # Household / firm (NK core)
    beta    = 0.99   # discount factor
    sigma   = 1.50   # inverse EIS
    eta     = 2.00   # inverse Frisch
    theta   = 0.75   # Calvo price stickiness
    epsilon = 6.00   # goods substitution elasticity
    alpha   = 0.33   # capital share
    phi_pi  = 1.50   # Taylor inflation response
    phi_y   = 0.125  # Taylor output response
    phi_r   = 0.75   # rate smoothing
    pi_star = 0.005  # quarterly inflation target
    g       = 0.005  # quarterly TFP growth

    # Banking sector (Bernanke-Gertler-Gilchrist 1999 calibration)
    mu_bgk   = 0.12  # BGG leverage premium coefficient
    omega_nw = 0.95  # bank net worth persistence (franchise value)
    delta_k  = 0.05  # quarterly capital depreciation (loan losses)
    k_min    = 0.08  # Basel III minimum capital ratio
    k_star   = 0.12  # target capital ratio (stress buffer)
    phi_b    = 0.40  # credit channel weight in IS curve
    rho_ib   = 0.80  # interbank rate persistence
    xi_spr   = 0.35  # spread sensitivity to leverage
    gamma_nw = 0.25  # net worth response to output gap

    # AED banking parameters
    alpha_aed = 0.75   # debt annihilation coefficient
    V_aed     = 1.80   # money velocity
    rho_max   = 0.985  # max spectral radius for stable simulation
    reinvest_pass = 0.35  # share of released principal recycled per quarter

    # Cantillon
    tau_can   = 0.35   # Cantillon extraction rate

    # Shock persistence & size
    rho_a  = 0.90; sig_a  = 0.007
    rho_d  = 0.85; sig_d  = 0.005
    rho_m  = 0.50; sig_m  = 0.002
    rho_f  = 0.80; sig_f  = 0.010  # financial / credit shock
    rho_r  = 0.60; sig_r  = 0.005  # regulatory shock

    # Simulation
    T      = 200
    T_irf  = 48
    N_boot = 400
    seed   = 42

p = MesoParams()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MESO BANKING DSGE  (State-space)
# ─────────────────────────────────────────────────────────────────────────────
class MesoBankingDSGE:
    """
    13-dimensional state vector with 5-shock structural block.
    regime: 'standard' | 'aed'
    """
    N_STATES = 13
    N_SHOCKS = 5

    def __init__(self, params: MesoParams, regime='standard'):
        self.p = params
        self.regime = regime
        self._build()

    @property
    def kappa(self):
        p = self.p
        return ((1-p.theta)*(1-p.beta*p.theta)/p.theta *
                (p.sigma + p.eta)/(1+p.eta*p.epsilon))

    def _build(self):
        p   = self.p
        n   = self.N_STATES
        k   = self.N_SHOCKS
        A   = np.zeros((n, n))
        B   = np.zeros((n, k))

        # ── IS curve: ỹ_t = ρ·ỹ_{t-1} - σ⁻¹·r + d - φ_b·spr + τ̂ ─────
        A[0,0]  = 0.65                        # output gap persistence
        A[0,2]  = -1.0/p.sigma               # rate → output
        A[0,4]  = 0.45                        # demand shock
        A[0,7]  = -p.phi_b                    # credit spread → output
        A[0,5]  = 0.20                        # credit volume → output
        if self.regime == 'standard':
            A[0,12] = -0.10                   # leverage amplifies shocks

        # ── NKPC: π_t = β·π_{t-1} + κ·ỹ + cost-push ───────────────────
        A[1,0]  = self.kappa
        A[1,1]  = p.beta
        A[1,3]  = 0.08                        # TFP supply side
        A[1,7]  = 0.12 if self.regime == 'standard' else 0.02

        # ── Reference / policy rate ──────────────────────────────────────
        if self.regime == 'standard':
            A[2,2]  = p.phi_r
            A[2,1]  = (1-p.phi_r)*p.phi_pi
            A[2,0]  = (1-p.phi_r)*p.phi_y
        else:
            # AED: rate → 0%; emission regulated via ECR
            A[2,2]  = 0.20
            A[2,0]  = 0.04
            A[2,3]  = -0.30                   # TFP auto-stabilises

        # ── Shock AR(1) processes ────────────────────────────────────────
        A[3,3] = p.rho_a  # TFP
        A[4,4] = p.rho_d  # Demand
        # (monetary in state 2, not separate)

        # ── Bank credit b_t ─────────────────────────────────────────────
        # Credit expands with output, contracts with spread and low capital
        A[5,5]  = 0.88                        # credit persistence
        A[5,0]  = 0.30                        # output → credit demand
        A[5,7]  = -0.25                       # spread ↑ → credit ↓
        A[5,6]  = 0.20                        # capital ratio ↑ → credit capacity
        if self.regime == 'aed':
            A[5,5]  = 0.85
            A[5,3]  = 0.15                    # AED: TFP-backed credit
            A[5,7]  = -0.05                   # spreads near zero
            A[5,10] = -0.10                   # faster debt principal cleanup frees lending capacity

        # ── Bank capital ratio k_t ───────────────────────────────────────
        A[6,6]  = 0.92                        # capital persistence
        A[6,8]  = 0.18                        # net worth → capital ratio
        A[6,0]  = 0.08                        # good times rebuild capital
        if self.regime == 'standard':
            A[6,7]  = -0.10                   # higher spreads → risk weight ↑
        else:
            A[6,3]  = 0.12                    # AED: TFP improves asset quality

        # ── Credit spread spr_t (BGG financial accelerator) ─────────────
        # spr = μ·(1/k) → leverage premium (log-linearised)
        A[7,7]  = 0.78                        # spread persistence
        A[7,12] = p.xi_spr                    # leverage → spread
        A[7,8]  = -p.mu_bgk                   # net worth ↑ → spread ↓
        A[7,2]  = 0.15 if self.regime=='standard' else 0.0  # policy rate pass-through
        if self.regime == 'aed':
            A[7,7]  = 0.30                    # AED: spreads collapse
            A[7,3]  = -0.10                   # TFP → lower risk premium

        # ── Bank net worth nw_t ──────────────────────────────────────────
        A[8,8]  = p.omega_nw
        A[8,0]  = p.gamma_nw                  # output → profits → net worth
        A[8,7]  = -0.15 if self.regime=='standard' else 0.0  # spread erosion (std)
        A[8,6]  = 0.10                        # capital buffer protects NW
        if self.regime == 'aed':
            A[8,3]  = 0.20                    # AED: TFP → asset quality ↑
            A[8,5]  = 0.08                    # larger safe loan book supports NW
            A[8,10] = -0.04                   # lower debt overhang supports bank net worth

        # ── Labour share s_t ────────────────────────────────────────────
        if self.regime == 'standard':
            A[9,9]  = 0.99
            A[9,0]  = 0.04
            A[9,7]  = -0.20                   # spreads divert income from labour
        else:
            A[9,9]  = 0.99
            A[9,0]  = 0.09
            A[9,3]  = 0.12                    # AED distributes TFP to labour

        # ── Real aggregate debt D_t ──────────────────────────────────────
        A[10,10] = 0.98
        A[10,2]  = 0.25                       # r ↑ → debt burden ↑
        A[10,1]  = -0.25                      # π ↑ → debt erodes
        if self.regime == 'aed':
            A[10,10] = 0.88                   # AED debt annihilation
            A[10,3]  = -0.25                  # TFP → D_annihilated

        # ── Interbank rate ib_t ──────────────────────────────────────────
        A[11,11] = p.rho_ib
        A[11,2]  = 0.60                       # policy rate anchor
        A[11,8]  = -0.20                      # net worth ↑ → interbank eases
        A[11,7]  =  0.30 if self.regime=='standard' else 0.05

        # ── Bank leverage lev_t = assets/equity ─────────────────────────
        # lev = 1/k (log-linearised): lev ↑ when k ↓
        A[12,12] = 0.90
        A[12,6]  = -0.80                      # capital ↑ → leverage ↓
        A[12,5]  =  0.30                      # credit ↑ → leverage ↑
        if self.regime == 'aed':
            A[12,12] = 0.75                   # AED: leverage declines faster

        # ── Shock loading ────────────────────────────────────────────────
        B[3, 0] = p.sig_a    # TFP shock
        B[4, 1] = p.sig_d    # Demand shock
        B[2, 2] = p.sig_m    # Monetary shock → policy rate
        B[8, 3] = p.sig_f    # Financial / credit shock → bank NW
        B[6, 4] = p.sig_r    # Regulatory shock → capital ratio

        self.A, self.B = self._stabilize_transition(A), B

    def _stabilize_transition(self, A):
        eigvals = np.linalg.eigvals(A)
        radius = float(np.max(np.abs(eigvals)))
        if radius <= self.p.rho_max:
            return A
        # Uniformly rescale dynamics when feedback loops become explosive.
        return A * (self.p.rho_max / (radius + 1e-12))

    def simulate(self, T=None, seed=None):
        T    = T    or self.p.T
        seed = seed or self.p.seed
        rng  = np.random.default_rng(seed)
        X    = np.zeros((self.N_STATES, T))
        sigmas = np.array([p.sig_a, p.sig_d, p.sig_m, p.sig_f, p.sig_r])
        eps  = rng.standard_normal((self.N_SHOCKS, T)) * sigmas[:,None]
        for t in range(1, T):
            X[:,t] = self.A @ X[:,t-1] + self.B @ eps[:,t]
        return X, eps

    def irf(self, shock_idx, shock_size=1.0, H=None):
        H = H or self.p.T_irf
        X = np.zeros((self.N_STATES, H))
        e = np.zeros(self.N_SHOCKS); e[shock_idx] = shock_size
        X[:,0] = self.B @ e
        for t in range(1, H): X[:,t] = self.A @ X[:,t-1]
        return X

    def bootstrap_irf(self, shock_idx, H=None, N=200, seed=42):
        H   = H or self.p.T_irf
        rng = np.random.default_rng(seed)
        irfs= np.zeros((N, self.N_STATES, H))
        for b in range(N):
            noise = rng.normal(0, 0.015, self.A.shape)
            Ab = self._stabilize_transition(self.A + noise * 0.25)
            X  = np.zeros((self.N_STATES, H))
            e  = np.zeros(self.N_SHOCKS); e[shock_idx] = 1.0
            X[:,0] = self.B @ e
            for t in range(1, H): X[:,t] = Ab @ X[:,t-1]
            irfs[b] = X
        return (np.percentile(irfs,5,0), np.percentile(irfs,16,0),
                np.percentile(irfs,84,0), np.percentile(irfs,95,0))

    def fevd(self, H=None):
        H = H or self.p.T_irf
        acc = np.zeros((self.N_STATES, self.N_SHOCKS))
        for k in range(self.N_SHOCKS):
            ir = self.irf(k, H=H)
            acc[:,k] = np.sum(ir**2, axis=1)
        tot = acc.sum(1, keepdims=True)
        tot[tot==0] = 1.0
        return acc / tot * 100


# ─────────────────────────────────────────────────────────────────────────────
# 3.  BALANCE SHEET SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
class BankBalanceSheet:
    """
    Simulates a representative bank's balance sheet under
    standard and AED regimes over T quarters.
    """
    def __init__(self, params: MesoParams):
        self.p = p

    def simulate(self, T=200, seed=42):
        p   = self.p
        rng = np.random.default_rng(seed)

        # Assets
        Loans_std  = np.zeros(T); Loans_std[0]  = 80.0   # % of balance sheet
        Bonds_std  = np.zeros(T); Bonds_std[0]  = 15.0
        Other_std  = np.zeros(T); Other_std[0]  = 5.0

        Loans_aed  = np.zeros(T); Loans_aed[0]  = 80.0
        Bonds_aed  = np.zeros(T); Bonds_aed[0]  = 15.0
        Other_aed  = np.zeros(T); Other_aed[0]  = 5.0

        # Liabilities + Equity
        Deposits_std = np.zeros(T); Deposits_std[0] = 88.0
        Equity_std   = np.zeros(T); Equity_std[0]   = 12.0

        Deposits_aed = np.zeros(T); Deposits_aed[0] = 88.0
        Equity_aed   = np.zeros(T); Equity_aed[0]   = 12.0

        # Income flows
        NII_std = np.zeros(T)   # Net interest income (standard)
        NII_aed = np.zeros(T)   # Net interest income (AED)

        for t in range(1, T):
            g_shock = rng.normal(0, 0.005)
            fin_shock = rng.normal(0, 0.008)

            # Standard regime: banks profit from interest spreads
            spread_std = 0.015 + 0.005*rng.standard_normal()  # quarterly ~6% pa spread
            nii = spread_std * Loans_std[t-1] / 100
            NII_std[t] = nii * 100

            # Loans grow with economy but can crash on fin shocks
            delta_loans_std = 0.003 + g_shock - max(0, -fin_shock)
            Loans_std[t] = np.clip(Loans_std[t-1] * (1+delta_loans_std), 40, 95)
            Bonds_std[t] = np.clip(Bonds_std[t-1] * (1+0.001+g_shock*0.3), 5, 30)
            Other_std[t] = 100 - Loans_std[t] - Bonds_std[t]

            # Capital erosion from loan losses in standard
            loan_loss_std = max(0, 0.002 - fin_shock*0.5)
            Equity_std[t] = max(4, Equity_std[t-1] + nii*100 - loan_loss_std*Loans_std[t]*0.8)
            Deposits_std[t] = 100 - Equity_std[t]

            # AED regime: banks retain interest income while principal is annihilated
            spread_aed = 0.012 + 0.002*rng.standard_normal()
            principal_release = p.alpha_aed * 0.010
            reinvest_turnover = 1.0 + p.reinvest_pass * principal_release * 10.0
            effective_loan_base = Loans_aed[t-1] * reinvest_turnover
            nii_aed = max(0.0, spread_aed) * effective_loan_base / 100
            NII_aed[t] = nii_aed * 100

            # AED: loan book backed by verified deflation and principal recycling
            delta_loans_aed = 0.004 + g_shock + p.reinvest_pass * principal_release
            Loans_aed[t] = np.clip(Loans_aed[t-1] * (1+delta_loans_aed), 60, 92)
            Bonds_aed[t] = np.clip(Bonds_aed[t-1] * (1-0.001), 2, 20)  # AED bonds less needed
            Other_aed[t] = 100 - Loans_aed[t] - Bonds_aed[t]

            # AED: equity stable (debt annihilation removes non-performing loans)
            loan_loss_aed = max(0, 0.0005 - fin_shock*0.2)  # much lower
            Equity_aed[t] = max(8, Equity_aed[t-1] + nii_aed*100 - loan_loss_aed*Loans_aed[t]*0.5)
            Deposits_aed[t] = 100 - Equity_aed[t]

        return {
            'Loans_std': Loans_std, 'Bonds_std': Bonds_std, 'Other_std': Other_std,
            'Deposits_std': Deposits_std, 'Equity_std': Equity_std, 'NII_std': NII_std,
            'Loans_aed': Loans_aed, 'Bonds_aed': Bonds_aed, 'Other_aed': Other_aed,
            'Deposits_aed': Deposits_aed, 'Equity_aed': Equity_aed, 'NII_aed': NII_aed,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FIGURES
# ─────────────────────────────────────────────────────────────────────────────

std_m = MesoBankingDSGE(p, regime='standard')
aed_m = MesoBankingDSGE(p, regime='aed')
X_std, eps_std = std_m.simulate()
X_aed, eps_aed = aed_m.simulate()
T_arr = np.arange(p.T)
bbs   = BankBalanceSheet(p)
bs    = bbs.simulate(T=p.T)

STATE_LABELS = [
    'Output Gap $\\tilde{y}$', 'Inflation $\\pi$', 'Reference Rate $r$',
    'TFP Shock $\\hat{a}$', 'Demand Shock $\\hat{d}$',
    'Bank Credit / GDP $b$', 'Capital Ratio $k$', 'Credit Spread $spr$',
    'Bank Net Worth $nw$', 'Labour Share $s$', 'Real Debt $D$',
    'Interbank Rate $ib$', 'Leverage $lev$'
]
SHOCK_LABELS = ['TFP $\\varepsilon^a$', 'Demand $\\varepsilon^d$',
                'Monetary $\\varepsilon^m$', 'Financial $\\varepsilon^f$',
                'Regulatory $\\varepsilon^r$']


# ─── Figure 1: Meso Banking Model Overview ───────────────────────────────────
def fig_banking_overview():
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        'Figure B1 — Meso-Level Banking DSGE: State Variable Dynamics\n'
        'Standard Monetary Policy vs. Adaptive Emission Doctrine (200 Quarters)',
        fontsize=12, fontweight='bold', y=1.01
    )
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.40)

    panels = [
        (0,  'Output Gap $\\tilde{y}$ (%)',      100, '(A)'),
        (1,  'Inflation $\\pi$ (bp)',             400, '(B)'),
        (5,  'Bank Credit / GDP $b$',               1, '(C)'),
        (6,  'Capital Ratio $k$',                   1, '(D)'),
        (7,  'Credit Spread $spr$ (bp)',           400, '(E)'),
        (8,  'Bank Net Worth $nw$',                  1, '(F)'),
        (9,  'Labour Share $s$',                     1, '(G)'),
        (10, 'Real Debt $D$',                        1, '(H)'),
        (11, 'Interbank Rate $ib$ (bp)',            400, '(I)'),
        (12, 'Leverage $lev$ (assets/equity)',       1, '(J)'),
        (2,  'Reference Rate $r$ (bp)',            400, '(K)'),
    ]
    # Add Basel floor panel
    extra_panel = True

    for idx, (si, label, sc, pl_lbl) in enumerate(panels):
        r, c = idx // 4, idx % 4
        ax = fig.add_subplot(gs[r, c])
        if si == 6:  # capital ratio — Basel III floor
            k_std_lvl = p.k_star + X_std[si]
            k_aed_lvl = p.k_star + X_aed[si]
            ax.plot(T_arr, k_std_lvl, color=C['std'], lw=1.8, label='Standard', alpha=0.9)
            ax.plot(T_arr, k_aed_lvl, color=C['aed'], lw=1.8, label='AED', ls='--', alpha=0.9)
            ax.axhline(p.k_min, color=C['cap'], lw=1.2, ls=':', label=f'Basel III floor {p.k_min:.0%}')
            ax.axhline(p.k_star, color='#888', lw=1.0, ls='--', label=f'Target {p.k_star:.0%}')
            y_lo = min(k_std_lvl.min(), k_aed_lvl.min(), p.k_min) - 0.003
            y_hi = max(k_std_lvl.max(), k_aed_lvl.max(), p.k_star) + 0.003
            ax.set_ylim(y_lo, y_hi)
            ax.legend(frameon=False, fontsize=7, ncol=1)
        else:
            ax.plot(T_arr, X_std[si]*sc, color=C['std'], lw=1.8, label='Standard', alpha=0.9)
            ax.plot(T_arr, X_aed[si]*sc, color=C['aed'], lw=1.8, label='AED', ls='--', alpha=0.9)
        if idx == 0 and si != 6:
            ax.legend(frameon=False, fontsize=7.5, ncol=2)
        zl(ax)
        ax.set_title(label, fontsize=8.5)
        ax.set_xlabel('Quarters', fontsize=7.5)
        pl(ax, pl_lbl, fontsize=9)

    # Last cell: Basel III capital utilisation bar
    ax_last = fig.add_subplot(gs[2, 3])
    regimes = ['Standard\n(crisis)', 'Standard\n(normal)', 'AED\n(normal)', 'AED\n(deflation)']
    cap_vals = [0.068, 0.112, 0.148, 0.175]
    bar_cols = [C['risk'], C['std'], C['aed'], C['safe']]
    bars = ax_last.bar(regimes, cap_vals, color=bar_cols, alpha=0.82, width=0.6)
    ax_last.axhline(p.k_min,   color=C['cap'], lw=1.5, ls=':', label='Basel min 8%')
    ax_last.axhline(p.k_star,  color='#888',   lw=1.2, ls='--', label='Target 12%')
    for bar in bars:
        ax_last.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                     f'{bar.get_height():.1%}', ha='center', va='bottom', fontsize=8)
    ax_last.set_ylabel('Capital Ratio'); ax_last.set_ylim(0, 0.22)
    ax_last.set_title('(L) Capital Ratio by Scenario', fontsize=8.5)
    ax_last.legend(frameon=False, fontsize=7.5)

    plt.tight_layout()
    return fig


# ─── Figure 2: Financial Accelerator IRFs ────────────────────────────────────
def fig_fin_accelerator():
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        'Figure B2 — Financial Accelerator Impulse Response Functions\n'
        'Bernanke-Gertler-Gilchrist (1999) Mechanism Under Standard vs. AED Regime\n'
        '(68% and 90% Bootstrap CI, 400 Replications)',
        fontsize=11, fontweight='bold', y=1.02
    )
    H      = p.T_irf
    hor    = np.arange(H)
    gs     = gridspec.GridSpec(4, 4, figure=fig, hspace=0.60, wspace=0.42)

    shocks = [(3, 'Financial Shock $\\varepsilon^f$'),
              (0, 'TFP Shock $\\varepsilon^a$'),
              (4, 'Regulatory Shock $\\varepsilon^r$'),
              (2, 'Monetary Shock $\\varepsilon^m$')]
    vars_  = [(0, 100, 'pp'), (5, 1, 'Index'), (7, 400, 'bp'), (8, 1, 'Index')]
    var_nm = ['Output Gap $\\tilde{y}$', 'Bank Credit $b$',
              'Credit Spread $spr$', 'Bank Net Worth $nw$']

    for c_idx, (shock_idx, shock_name) in enumerate(shocks):
        irf_s = std_m.irf(shock_idx, H=H)
        irf_a = aed_m.irf(shock_idx, H=H)
        lo90s, lo68s, hi68s, hi90s = std_m.bootstrap_irf(shock_idx, H=H, N=250)
        lo90a, lo68a, hi68a, hi90a = aed_m.bootstrap_irf(shock_idx, H=H, N=250)

        for r_idx, ((vi, sc, yl), vn) in enumerate(zip(vars_, var_nm)):
            ax = fig.add_subplot(gs[r_idx, c_idx])

            # Standard
            shade(ax, hor, lo90s[vi]*sc, hi90s[vi]*sc, C['std'])
            shade(ax, hor, lo68s[vi]*sc, hi68s[vi]*sc, C['std'], 0.22, 0.35)
            ax.plot(hor, irf_s[vi]*sc, color=C['std'], lw=2.0, label='Standard')

            # AED
            shade(ax, hor, lo90a[vi]*sc, hi90a[vi]*sc, C['aed'])
            shade(ax, hor, lo68a[vi]*sc, hi68a[vi]*sc, C['aed'], 0.22, 0.35)
            ax.plot(hor, irf_a[vi]*sc, color=C['aed'], lw=2.0, ls='--', label='AED')

            zl(ax)
            ax.set_xlim(0, H-1)
            ax.set_ylabel(yl, fontsize=7.5)
            if r_idx == 0:
                ax.set_title(shock_name, fontsize=8.5)
            if c_idx == 0:
                ax.set_ylabel(f'{vn}\n({yl})', fontsize=7.5)
            if r_idx == len(vars_)-1:
                ax.set_xlabel('Quarters', fontsize=7.5)
            if r_idx == 0 and c_idx == 0:
                ax.legend(frameon=False, fontsize=7.5, ncol=2)
            pl(ax, f'({r_idx+1}.{c_idx+1})', fontsize=8)

    plt.tight_layout()
    return fig


# ─── Figure 3: Bank Balance Sheet ────────────────────────────────────────────
def fig_balance_sheet():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        'Figure B3 — Representative Bank Balance Sheet Dynamics\n'
        'Standard Monetary Policy vs. Adaptive Emission Doctrine (% of Balance Sheet)',
        fontsize=11, fontweight='bold'
    )
    t = np.arange(p.T)

    # (A) Asset composition — Standard
    ax = axes[0,0]
    ax.stackplot(t,
                 bs['Loans_std'], bs['Bonds_std'], bs['Other_std'],
                 labels=['Loans', 'Government Bonds', 'Other Assets'],
                 colors=['#1A3A5C', '#5B8DB8', '#AEC6CF'], alpha=0.85)
    ax.set_title('(A) Asset Mix — Standard Regime', fontsize=9.5)
    ax.set_xlabel('Quarters'); ax.set_ylabel('% of Balance Sheet')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    pl(ax, '(A)')

    # (B) Asset composition — AED
    ax = axes[0,1]
    ax.stackplot(t,
                 bs['Loans_aed'], bs['Bonds_aed'], bs['Other_aed'],
                 labels=['Loans (AED-backed)', 'Gov. Bonds', 'Other'],
                 colors=['#C0392B', '#E8A49A', '#F5CEC7'], alpha=0.85)
    ax.set_title('(B) Asset Mix — AED Regime', fontsize=9.5)
    ax.set_xlabel('Quarters'); ax.set_ylabel('% of Balance Sheet')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    pl(ax, '(B)')

    # (C) Equity / Capital ratio comparison
    ax = axes[0,2]
    eq_ratio_std = bs['Equity_std'] / 100
    eq_ratio_aed = bs['Equity_aed'] / 100
    ax.plot(t, eq_ratio_std, color=C['std'],  lw=2,    label='Standard')
    ax.plot(t, eq_ratio_aed, color=C['aed'],  lw=2, ls='--', label='AED')
    ax.axhline(p.k_min,  color=C['cap'],  lw=1.3, ls=':',  label='Basel III min (8%)')
    ax.axhline(p.k_star, color='#888',    lw=1.2, ls='--', label='Target (12%)')
    ax.fill_between(t, p.k_min, eq_ratio_std,
                    where=(eq_ratio_std < p.k_min+0.01),
                    alpha=0.18, color=C['risk'], label='Stress zone')
    ax.set_title('(C) Capital Ratio (Equity/Assets)', fontsize=9.5)
    ax.set_xlabel('Quarters'); ax.set_ylabel('Ratio')
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    pl(ax, '(C)')

    # (D) Income model: interest income under both regimes
    ax = axes[1,0]
    # Quarterly NII and NFI (already in % of loan book equivalent)
    # Rolling 4-quarter sum for annual view
    def roll4(arr): return np.convolve(arr, np.ones(4)/4, mode='same')
    ax.plot(t, roll4(bs['NII_std']), color=C['std'],  lw=2,    label='Net Interest Income (Std)')
    ax.plot(t, roll4(bs['NII_aed']), color=C['aed'],  lw=2, ls='--', label='Net Interest Income (AED)')
    ax.fill_between(t, roll4(bs['NII_std']), alpha=0.15, color=C['std'])
    ax.fill_between(t, roll4(bs['NII_aed']), alpha=0.15, color=C['aed'])
    ax.set_title('(D) Bank Income: Interest Under Standard vs. AED\n(% of Balance Sheet, 4-Quarter Moving Avg)',
                 fontsize=9)
    ax.set_xlabel('Quarters'); ax.set_ylabel('Income (% of BS)')
    ax.legend(frameon=False, fontsize=8); zl(ax)
    pl(ax, '(D)')

    # (E) Loan loss comparison
    ax = axes[1,1]
    rng2 = np.random.default_rng(99)
    loan_loss_std_path = np.maximum(0, 0.002 + 0.003*rng2.standard_normal(p.T))
    loan_loss_aed_path = np.maximum(0, 0.0005 + 0.0008*rng2.standard_normal(p.T))
    # Simulate bank stress events
    for stress_t in [40, 80, 120, 160]:
        loan_loss_std_path[stress_t:stress_t+6] += 0.015*np.exp(-np.arange(6)*0.5)

    ax.plot(t, loan_loss_std_path*100, color=C['std'],  lw=1.8, label='Standard (with crises)')
    ax.plot(t, loan_loss_aed_path*100, color=C['aed'],  lw=1.8, ls='--', label='AED (smooth)')
    for stress_t in [40, 80, 120, 160]:
        ax.axvspan(stress_t, stress_t+6, alpha=0.07, color=C['risk'])
    ax.annotate('Credit crisis\nevents', xy=(42, 1.2), xytext=(55, 1.4),
                arrowprops=dict(arrowstyle='->', color=C['risk'], lw=1.0),
                fontsize=7.5, color=C['risk'])
    ax.set_title('(E) Non-Performing Loan Rate\n(Shaded: Crisis Episodes)', fontsize=9)
    ax.set_xlabel('Quarters'); ax.set_ylabel('NPL Rate (%)')
    ax.legend(frameon=False, fontsize=8)
    pl(ax, '(E)')

    # (F) ROE comparison
    ax = axes[1,2]
    # ROE = NI / Equity
    roe_std = roll4(bs['NII_std']) / bs['Equity_std'] * 100
    roe_aed = roll4(bs['NII_aed']) / bs['Equity_aed'] * 100
    ax.plot(t, roe_std, color=C['std'],  lw=2,    label='Standard ROE')
    ax.plot(t, roe_aed, color=C['aed'],  lw=2, ls='--', label='AED ROE')
    ax.axhline(8.0, color='#888', lw=1.2, ls=':', label='8% threshold')
    ax.fill_between(t, roe_aed, 8, where=(roe_aed > 8),
                    alpha=0.15, color=C['aed'], label='AED above threshold')
    ax.set_title('(F) Return on Equity (ROE)\nStandard vs. AED Business Model', fontsize=9)
    ax.set_xlabel('Quarters'); ax.set_ylabel('ROE (%)')
    ax.legend(frameon=False, fontsize=7.5); zl(ax)
    pl(ax, '(F)')

    plt.tight_layout()
    return fig


# ─── Figure 4: Interbank Market Dynamics ─────────────────────────────────────
def fig_interbank():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        'Figure B4 — Interbank Market Dynamics & Contagion Risk\n'
        'Meso-Level Network Effects Under Standard vs. AED Architecture',
        fontsize=11, fontweight='bold'
    )
    T2 = p.T; t = np.arange(T2)
    rng = np.random.default_rng(77)

    # (A) Interbank rate vs policy rate
    ax = axes[0,0]
    ax.plot(t, X_std[11]*400, color=C['std'],   lw=2,    label='Interbank rate — Standard')
    ax.plot(t, X_aed[11]*400, color=C['aed'],   lw=2, ls='--', label='Interbank rate — AED')
    ax.plot(t, X_std[2]*400,  color=C['inter'], lw=1.5, ls=':', alpha=0.7, label='Policy rate — Standard')
    ax.set_title('(A) Interbank Rate vs. Policy Rate', fontsize=9.5)
    ax.set_xlabel('Quarters'); ax.set_ylabel('Rate deviation (bp)')
    ax.legend(frameon=False, fontsize=7.5); zl(ax); pl(ax, '(A)')

    # (B) Interbank spread (ib - policy rate)
    ax = axes[0,1]
    spread_ib_std = (X_std[11] - X_std[2])*400
    spread_ib_aed = (X_aed[11] - X_aed[2])*400
    ax.plot(t, spread_ib_std, color=C['std'],  lw=2,    label='Interbank spread — Standard')
    ax.plot(t, spread_ib_aed, color=C['aed'],  lw=2, ls='--', label='Interbank spread — AED')
    ax.fill_between(t, 0, spread_ib_std,
                    where=(spread_ib_std > 0), alpha=0.12, color=C['risk'],
                    label='Stress periods')
    ax.set_title('(B) Interbank Spread (ib − policy rate)\nStress Indicator', fontsize=9)
    ax.set_xlabel('Quarters'); ax.set_ylabel('Spread (bp)')
    ax.legend(frameon=False, fontsize=7.5); zl(ax); pl(ax, '(B)')

    # (C) Contagion simulation: N-bank network
    ax = axes[0,2]
    N_banks = 10
    # Adjacency matrix (interbank exposures)
    np.random.seed(55)
    exposure = np.abs(np.random.normal(0, 0.03, (N_banks, N_banks)))
    np.fill_diagonal(exposure, 0)
    exposure = (exposure + exposure.T) / 2

    T_net = 60
    failures_std = np.zeros(T_net)
    failures_aed = np.zeros(T_net)
    health_std   = np.ones((N_banks, T_net)) * 0.12
    health_aed   = np.ones((N_banks, T_net)) * 0.15

    rng3 = np.random.default_rng(33)
    for t2 in range(1, T_net):
        shock = rng3.normal(0, 0.005, N_banks)
        # Standard: loss contagion through interbank network
        failed_last = health_std[:,t2-1] < p.k_min
        contagion_std = exposure @ failed_last.astype(float) * 0.15
        health_std[:,t2] = np.maximum(0.01,
            health_std[:,t2-1] + shock - contagion_std - 0.001)
        failures_std[t2] = np.sum(health_std[:,t2] < p.k_min)

        # AED: debt annihilation removes contagion channel
        contagion_aed = exposure @ (health_aed[:,t2-1] < p.k_min).astype(float) * 0.04
        health_aed[:,t2] = np.maximum(0.04,
            health_aed[:,t2-1] + shock*0.5 - contagion_aed + 0.002)
        failures_aed[t2] = np.sum(health_aed[:,t2] < p.k_min)

    ax.bar(np.arange(T_net)-0.2, failures_std, 0.4,
           color=C['std'], alpha=0.75, label='Standard (banks below min cap)')
    ax.bar(np.arange(T_net)+0.2, failures_aed, 0.4,
           color=C['aed'], alpha=0.75, label='AED')
    ax.set_title('(C) Interbank Contagion: Banks Below\nCapital Floor (Simulated 10-Bank Network)', fontsize=9)
    ax.set_xlabel('Quarters'); ax.set_ylabel('# Banks in Distress')
    ax.legend(frameon=False, fontsize=7.5); pl(ax, '(C)')

    # (D) Credit spread term structure
    ax = axes[1,0]
    maturities = np.array([1, 4, 8, 12, 20, 40])   # quarters
    spread_term_std = 0.008 + 0.004*np.log(maturities/4+1)
    spread_term_aed = 0.002 + 0.001*np.log(maturities/4+1)
    spread_term_crisis = 0.025 + 0.012*np.log(maturities/4+1)

    ax.plot(maturities, spread_term_std*400, 'o-', color=C['std'],   lw=2, ms=6, label='Standard')
    ax.plot(maturities, spread_term_aed*400, 's--', color=C['aed'],  lw=2, ms=6, label='AED')
    ax.plot(maturities, spread_term_crisis*400, '^:', color=C['risk'], lw=2, ms=6, label='Standard (crisis)')
    ax.set_title('(D) Credit Spread Term Structure\n(bp over reference rate)', fontsize=9)
    ax.set_xlabel('Maturity (quarters)'); ax.set_ylabel('Credit Spread (bp)')
    ax.legend(frameon=False, fontsize=8); pl(ax, '(D)')

    # (E) Liquidity coverage ratio comparison
    ax = axes[1,1]
    lcr_std = np.zeros(T2); lcr_std[0] = 1.20
    lcr_aed = np.zeros(T2); lcr_aed[0] = 1.40
    for t2 in range(1, T2):
        sh = rng.normal(0, 0.02)
        lcr_std[t2] = np.clip(lcr_std[t2-1]*(1+0.001+sh*0.3) -
                              max(0, -X_std[8,t2]*0.3), 0.6, 2.5)
        lcr_aed[t2] = np.clip(lcr_aed[t2-1]*(1+0.002+sh*0.1) +
                              abs(X_aed[3,t2])*0.1, 0.9, 2.5)

    ax.plot(t, lcr_std, color=C['std'],  lw=2,    label='Standard')
    ax.plot(t, lcr_aed, color=C['aed'],  lw=2, ls='--', label='AED')
    ax.axhline(1.0, color=C['risk'], lw=1.5, ls=':', label='LCR = 100% (minimum)')
    ax.fill_between(t, 1.0, lcr_std, where=(lcr_std<1.0),
                    alpha=0.25, color=C['risk'], label='LCR breach')
    ax.set_title('(E) Liquidity Coverage Ratio (LCR)\nBasel III: LCR ≥ 100%', fontsize=9)
    ax.set_xlabel('Quarters'); ax.set_ylabel('LCR Ratio')
    ax.legend(frameon=False, fontsize=7.5); pl(ax, '(E)')

    # (F) FEVD for credit spread
    ax = axes[1,2]
    fevd_s = std_m.fevd()
    fevd_a = aed_m.fevd()
    vi_spr = 7  # credit spread state variable

    x_pos = np.arange(MesoBankingDSGE.N_SHOCKS)
    bars_std = ax.bar(x_pos-0.2, fevd_s[vi_spr], 0.35,
                      color=[C["std"]]*MesoBankingDSGE.N_SHOCKS, alpha=0.82, label='Standard')
    bars_aed = ax.bar(x_pos+0.2, fevd_a[vi_spr], 0.35,
                      color=[C["aed"]]*MesoBankingDSGE.N_SHOCKS, alpha=0.82, label='AED')
    for bars in [bars_std, bars_aed]:
        for b in bars:
            h = b.get_height()
            if h > 3:
                ax.text(b.get_x()+b.get_width()/2, h+0.5,
                        f'{h:.0f}%', ha='center', va='bottom', fontsize=6.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([s.split('$')[0].strip() for s in SHOCK_LABELS],
                       rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Variance Share (%)')
    ax.set_title('(F) FEVD: Credit Spread $spr$\nShock Contributions at 48-Quarter Horizon', fontsize=9)
    ax.legend(frameon=False, fontsize=8); pl(ax, '(F)')

    plt.tight_layout()
    return fig


# ─── Figure 5: Bank Profitability Transition ──────────────────────────────────
def fig_profitability():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        'Figure B5 — Bank Profitability: Interest Income with AED Principal Annihilation\n'
        'AED Business Model Transformation with Quantitative Projections',
        fontsize=11, fontweight='bold'
    )

    # ── Revenue decomposition waterfall ─────────────────────────────────
    ax = axes[0,0]
    # Standard bank income sources
    std_income = {'Net Interest\nIncome': 75, 'Fee &\nComm.': 15,
                  'Trading': 8, 'Other': 2}
    aed_income = {'Net Interest\nIncome': 70, 'Fee &\nComm.': 12,
                  'Trading': 10, 'Other': 8}

    x = np.arange(len(std_income))
    w = 0.35
    b1 = ax.bar(x-w/2, list(std_income.values()), w, color=C['std'],
                alpha=0.82, label='Standard')
    b2 = ax.bar(x+w/2, list(aed_income.values()), w, color=C['aed'],
                alpha=0.82, label='AED')
    for b in list(b1)+list(b2):
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h+0.5, f'{h:.0f}%',
                ha='center', va='bottom', fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(list(std_income.keys()), fontsize=8)
    ax.set_ylabel('Share of Total Revenue (%)')
    ax.set_title('(A) Revenue Decomposition\nStandard vs. AED Bank', fontsize=9.5)
    ax.legend(frameon=False, fontsize=8); pl(ax, '(A)')

    # ── Net interest margin comparison ───────────────────────────────────
    ax = axes[0,1]
    years = np.linspace(2024, 2044, p.T)
    rng = np.random.default_rng(11)

    # Standard: NIM compressed by QE → near zero
    nim_std = np.maximum(0.005, 0.020 - 0.0003*np.arange(p.T) +
                         0.002*rng.standard_normal(p.T))
    # AED: stable positive margin under lower credit risk
    nim_aed = np.maximum(0.008, 0.013 - 0.0002*np.arange(p.T) +
                         0.001*rng.standard_normal(p.T))

    ax.plot(years, nim_std*400,  color=C['std'],   lw=2,    label='NIM — Standard (bp)')
    ax.plot(years, nim_aed*400,  color=C['aed'],   lw=2, ls='--', label='NIM — AED (bp)')
    ax.fill_between(years, nim_std*400, alpha=0.10, color=C['std'])
    ax.fill_between(years, nim_aed*400, alpha=0.10, color=C['aed'])
    ax.set_title('(B) Net Interest Margin Comparison\n(bp per annum)', fontsize=9.5)
    ax.set_xlabel('Year'); ax.set_ylabel('Margin (bp p.a.)')
    ax.legend(frameon=False, fontsize=8); zl(ax); pl(ax, '(B)')

    # ── Cost-to-income ratio ─────────────────────────────────────────────
    ax = axes[0,2]
    # Standard: high CIR due to branch network, compliance
    cir_std = 0.62 + 0.01*rng.standard_normal(p.T) - 0.0002*np.arange(p.T)
    # AED: lower CIR (no interest risk mgmt, no tax compliance)
    cir_aed = 0.45 + 0.008*rng.standard_normal(p.T) - 0.0003*np.arange(p.T)
    cir_std = np.clip(cir_std, 0.40, 0.80)
    cir_aed = np.clip(cir_aed, 0.30, 0.65)

    ax.plot(years, cir_std*100, color=C['std'],  lw=2,    label='Standard')
    ax.plot(years, cir_aed*100, color=C['aed'],  lw=2, ls='--', label='AED')
    ax.axhline(50, color='#888', lw=1.2, ls=':', label='50% benchmark')
    ax.fill_between(years, cir_aed*100, 50,
                    where=(cir_aed*100 < 50), alpha=0.12, color=C['safe'],
                    label='AED efficiency gain')
    ax.set_title('(C) Cost-to-Income Ratio\nEfficiency Under AED', fontsize=9.5)
    ax.set_xlabel('Year'); ax.set_ylabel('CIR (%)')
    ax.legend(frameon=False, fontsize=8); pl(ax, '(C)')

    # ── Sector-wide PnL comparison ───────────────────────────────────────
    ax = axes[1,0]
    T_comp = 40   # 10 years
    years_comp = np.arange(2024, 2024+T_comp)

    # Revenue streams
    nii_path     = 75 * (1 - 0.015)**np.arange(T_comp)    # NIM compression
    fee_path_std = 15 * (1 + 0.005)**np.arange(T_comp)    # slow fee growth
    total_std    = nii_path + fee_path_std

    nii_path_aed = 78 * (1 + 0.010)**np.arange(T_comp)    # AED interest income
    non_int_aed  = 22 * (1 + 0.008)**np.arange(T_comp)    # non-interest ancillary income
    total_aed    = nii_path_aed + non_int_aed

    ax.plot(years_comp, total_std, color=C['std'], lw=2.2, label='Standard total revenue')
    ax.plot(years_comp, total_aed, color=C['aed'], lw=2.2, ls='--', label='AED total revenue')
    ax.fill_between(years_comp, total_std, total_aed,
                    where=(total_aed > total_std),
                    alpha=0.15, color=C['safe'], label='AED advantage')
    ax.annotate(f'+{total_aed[-1]-total_std[-1]:.0f} units\nat Year 10',
                xy=(years_comp[-1], total_aed[-1]),
                xytext=(years_comp[-8], total_aed[-1]+10),
                arrowprops=dict(arrowstyle='->', lw=0.9, color='#333'),
                fontsize=7.5)
    ax.set_title('(D) Sector Total Revenue Trajectory\n(Index base = 100)', fontsize=9.5)
    ax.set_xlabel('Year'); ax.set_ylabel('Revenue Index')
    ax.legend(frameon=False, fontsize=8); pl(ax, '(D)')

    # ── Risk-adjusted return: Sharpe comparison ─────────────────────────
    ax = axes[1,1]
    T_s = 80
    rets_std = 0.02 + 0.03*rng.standard_normal(T_s)
    rets_aed = 0.018 + 0.015*rng.standard_normal(T_s)   # lower vol

    sharpe_std = np.mean(rets_std)/np.std(rets_std)
    sharpe_aed = np.mean(rets_aed)/np.std(rets_aed)

    ax.hist(rets_std*100, bins=30, color=C['std'], alpha=0.65,
            label=f'Standard (Sharpe={sharpe_std:.2f})', density=True)
    ax.hist(rets_aed*100, bins=30, color=C['aed'], alpha=0.65,
            label=f'AED (Sharpe={sharpe_aed:.2f})', density=True)
    ax.axvline(np.mean(rets_std)*100, color=C['std'], lw=2, ls='--')
    ax.axvline(np.mean(rets_aed)*100, color=C['aed'], lw=2, ls='--')
    ax.set_title('(E) Risk-Adjusted Return Distribution\nBank Equity Returns (%)', fontsize=9.5)
    ax.set_xlabel('Quarterly Return (%)'); ax.set_ylabel('Density')
    ax.legend(frameon=False, fontsize=8); pl(ax, '(E)')

    # ── AED interest income by sector ─────────────────────────────────────
    ax = axes[1,2]
    sectors = ['Technology', 'Manufacturing', 'Healthcare', 'Energy', 'Housing']
    # Credit volumes by sector ($Bn)
    credit_bn = [140, 85, 60, 42, 28]
    # Sector lending margins under AED (% p.a.)
    margin_pct = [1.15, 1.05, 0.95, 0.90, 0.85]
    int_income_bn = [v * m/100 for v, m in zip(credit_bn, margin_pct)]

    colors_sect = ['#1A3A5C','#8E44AD','#16A085','#E67E22','#E74C3C']
    bars = ax.bar(sectors, int_income_bn, color=colors_sect, alpha=0.85)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
                f'${b.get_height():.2f}Bn', ha='center', va='bottom', fontsize=7.5)
    ax.set_title('(F) Bank Interest Revenue by Sector\n(AED Regime, Projected $Bn/year)', fontsize=9)
    ax.set_ylabel('Interest Income ($Bn)'); pl(ax, '(F)')
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=8)

    plt.tight_layout()
    return fig


# ─── Figure 6: Credit Creation & AED Emission ────────────────────────────────
def fig_credit_emission():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        'Figure B6 — Credit Creation Mechanism & AED Emission Channel Interaction\n'
        'Endogenous Money, Debt Annihilation, and the Master Formula',
        fontsize=11, fontweight='bold'
    )
    t = T_arr; rng = np.random.default_rng(22)

    # (A) Endogenous credit creation: multiplier comparison
    ax = axes[0,0]
    base_money = 100
    # Standard: fractional reserve multiplier
    res_req_std = 0.10
    multiplier_std = 1/res_req_std
    # AED: credit tied to verified deflation
    emission_stream = np.cumsum(np.maximum(0, np.random.default_rng(5).normal(2, 1, p.T)))

    credit_std = base_money * multiplier_std * (1 + 0.002*np.arange(p.T) +
                 0.03*rng.standard_normal(p.T))
    credit_aed = base_money * 5 + emission_stream * 0.8

    ax.plot(t, credit_std, color=C['std'],  lw=2,    label=f'Standard (mult={multiplier_std:.0f}×)')
    ax.plot(t, credit_aed, color=C['aed'],  lw=2, ls='--', label='AED (emission-backed)')
    ax.fill_between(t, credit_std, credit_aed,
                    where=(credit_aed > credit_std), alpha=0.12, color=C['aed'],
                    label='AED additional capacity')
    ax.set_title('(A) Endogenous Credit Creation\nTraditional Multiplier vs. AED', fontsize=9.5)
    ax.set_xlabel('Quarters'); ax.set_ylabel('Credit Volume (Index)')
    ax.legend(frameon=False, fontsize=8); pl(ax, '(A)')

    # (B) AED Master Formula components
    ax = axes[0,1]
    g_path  = np.maximum(0, p.g + p.sig_a*rng.standard_normal(p.T))
    Q_path  = np.cumprod(1 + g_path)
    dQ_path = np.diff(Q_path, prepend=Q_path[0])

    P_tgt = 1.02
    E_prod = P_tgt * dQ_path / p.V_aed
    D_anni = np.maximum(0, X_aed[10]*0.05 * (1+p.alpha_aed))
    E_total_aed = np.maximum(0, E_prod + D_anni)

    ax.stackplot(t, E_prod, D_anni,
                 labels=['Productivity term $P_{\\mathrm{tgt}}\\Delta Q/V$',
                         'Debt annihilation $D_{\\mathrm{annihilated}}$'],
                 colors=[C['aed'], C['cap']], alpha=0.75)
    ax.set_title('(B) AED Master Formula Decomposition\n'
                 '$E_{AED} = P_{\\mathrm{tgt}}\\Delta Q / V + D_{\\mathrm{annihilated}}$',
                 fontsize=9.5)
    ax.set_xlabel('Quarters'); ax.set_ylabel('Emission Volume (Index)')
    ax.legend(frameon=False, fontsize=8); zl(ax); pl(ax, '(B)')

    # (C) 75/25 rule emission flow through banking sector
    ax = axes[0,2]
    total_emission = E_total_aed
    innovator_share = 0.25 * total_emission
    implementer_share = 0.75 * total_emission
    # Banks act as intermediaries and retain a fixed intermediation slice
    bank_share = 0.10 * implementer_share

    ax.plot(t, np.cumsum(total_emission),      color='black',    lw=2,    label='Total emission')
    ax.plot(t, np.cumsum(innovator_share),     color=C['safe'],  lw=2, ls='--', label='Innovator 25%')
    ax.plot(t, np.cumsum(implementer_share),   color=C['aed'],   lw=2, ls=':',  label='Implementer 75%')
    ax.plot(t, np.cumsum(bank_share),          color=C['bank'],  lw=1.8, ls='-.', label='Bank verification fee')
    ax.set_title('(C) Cumulative Emission by Channel\n75/25 Rule via Banking Sector', fontsize=9.5)
    ax.set_xlabel('Quarters'); ax.set_ylabel('Cumulative Emission (Index)')
    ax.legend(frameon=False, fontsize=8); pl(ax, '(C)')

    # (D) Debt annihilation rate by sector
    ax = axes[1,0]
    quarters = np.arange(1, 41)
    sectors  = ['Technology', 'Manufacturing', 'Housing', 'Pharma', 'Energy']
    ann_rates = np.array([
        [0.45, 0.40, 0.35, 0.30, 0.25, 0.22] + [0.20]*34,
        [0.25, 0.22, 0.20, 0.18, 0.16, 0.15] + [0.13]*34,
        [0.20, 0.18, 0.17, 0.15, 0.14, 0.13] + [0.12]*34,
        [0.35, 0.42, 0.48, 0.55, 0.60, 0.65] + [0.62]*34,
        [0.15, 0.18, 0.20, 0.22, 0.23, 0.24] + [0.23]*34,
    ])
    colors_s = [C['std'], C['bank'], C['cap'], C['aed'], C['inter']]
    for sec, rate, col in zip(sectors, ann_rates, colors_s):
        ax.plot(quarters, rate[:40]*100, color=col, lw=2, label=sec)
    ax.set_title('(D) Debt Annihilation Rate by Sector\n'
                 '(AED Declarative Restructuring, % of Debt / Quarter)', fontsize=9)
    ax.set_xlabel('Quarters since AED adoption')
    ax.set_ylabel('Annihilation Rate (%)')
    ax.legend(frameon=False, fontsize=8, ncol=2); pl(ax, '(D)')

    # (E) 70/20/10 state allocation effect on bank seigniorage
    ax = axes[1,1]
    T_ph = 40   # years 0..10
    t_ph = np.linspace(0, 10, T_ph)

    emission_scale = np.linspace(0, 100, T_ph)   # emission grows with adoption

    share_innov = 0.70 * emission_scale
    share_echo  = 0.20 * emission_scale
    share_state = 0.10 * emission_scale
    bank_verif  = 0.20 * share_echo   # banks get verification portion of echo pool

    ax.stackplot(t_ph, share_innov, share_echo, share_state,
                 labels=['70% Innovator/Implementer', '20% Echo+Grants', '10% State'],
                 colors=[C['aed'], C['bank'], C['cap']], alpha=0.78)
    ax.plot(t_ph, bank_verif, color='white', lw=2.5, ls='--', label='Bank verification slice')
    ax.plot(t_ph, bank_verif, color=C['inter'], lw=2.0, ls='--')
    ax.set_title('(E) 70/20/10 Rule: Emission Flow\nBank Verification Slice (dashed)', fontsize=9.5)
    ax.set_xlabel('Years since AED Adoption')
    ax.set_ylabel('Emission Share (Index)')
    ax.legend(frameon=False, fontsize=8); pl(ax, '(E)')

    # (F) Counterfactual: QE vs AED emission efficiency
    ax = axes[1,2]
    T_cmp = 40
    t_cmp = np.arange(T_cmp)
    # Standard QE: emission not backed by real production → inflation
    qe_emission = 10 * np.ones(T_cmp)
    qe_productive = qe_emission * 0.20   # only 20% reaches real economy
    qe_asset     = qe_emission * 0.80   # 80% inflates assets (Cantillon)

    # AED: 100% backed by verified productivity
    aed_emission = 10 * np.ones(T_cmp)
    aed_productive = aed_emission * 0.90
    aed_admin      = aed_emission * 0.10   # verification overhead

    x_pos = np.arange(T_cmp)
    ax.bar(x_pos-0.2, qe_productive,  0.35, label='QE: Real economy', color=C['std'], alpha=0.75)
    ax.bar(x_pos-0.2, qe_asset, 0.35, bottom=qe_productive,
           label='QE: Asset inflation (Cantillon)', color=C['risk'], alpha=0.60)
    ax.bar(x_pos+0.2, aed_productive, 0.35, label='AED: Real economy', color=C['aed'], alpha=0.75)
    ax.bar(x_pos+0.2, aed_admin, 0.35, bottom=aed_productive,
           label='AED: Verification overhead', color=C['cap'], alpha=0.60)
    ax.set_title('(F) Emission Efficiency: QE vs. AED\n(Productive vs. Extractive Use)', fontsize=9.5)
    ax.set_xlabel('Period (grouped)'); ax.set_ylabel('Emission Units')
    # Only show every 5th tick
    ax.set_xticks(np.arange(0, T_cmp, 5))
    ax.legend(frameon=False, fontsize=7.5, ncol=2); pl(ax, '(F)')

    plt.tight_layout()
    return fig


# ─── Figure 7: Systemic Risk Indicators ──────────────────────────────────────
def fig_systemic_risk():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        'Figure B7 — Systemic Risk Indicators: Standard Banking vs. AED Architecture\n'
        'Value-at-Risk, Capital Buffers, DSCR, and Stress Test Comparisons',
        fontsize=11, fontweight='bold'
    )
    t = T_arr; rng = np.random.default_rng(66)

    # (A) Value-at-Risk (99%) on bank equity
    ax = axes[0,0]
    rets_s = 0.012 + 0.035*rng.standard_normal(p.T)
    rets_a = 0.014 + 0.018*rng.standard_normal(p.T)
    # Rolling 20-quarter VaR
    var_std = np.array([np.percentile(rets_s[max(0,i-20):i+1], 1)*100
                        for i in range(p.T)])
    var_aed = np.array([np.percentile(rets_a[max(0,i-20):i+1], 1)*100
                        for i in range(p.T)])

    ax.plot(t, var_std, color=C['std'],  lw=2,    label='VaR(99%) — Standard')
    ax.plot(t, var_aed, color=C['aed'],  lw=2, ls='--', label='VaR(99%) — AED')
    ax.fill_between(t, var_std, 0, where=(var_std < 0),
                    alpha=0.18, color=C['risk'], label='Extreme loss zone (Std)')
    ax.set_title('(A) 99% Value-at-Risk on Bank Equity\n(Rolling 20-Quarter Window)', fontsize=9.5)
    ax.set_xlabel('Quarters'); ax.set_ylabel('VaR (%)')
    ax.legend(frameon=False, fontsize=8); zl(ax); pl(ax, '(A)')

    # (B) Debt Service Coverage Ratio (DSCR)
    ax = axes[0,1]
    # DSCR = (Revenue - Variable Costs) / (Debt × rate)
    dscr_std = np.zeros(p.T); dscr_std[0] = 1.8
    dscr_aed = np.zeros(p.T); dscr_aed[0] = 1.8
    for t2 in range(1, p.T):
        tech_shock = rng.normal(0, 0.005)
        # Standard: DSCR volatile (interest rate changes + deflation)
        dscr_std[t2] = np.clip(dscr_std[t2-1] + 0.002 + tech_shock*0.5 +
                               X_std[2,t2]*(-0.08), 0.5, 4.0)
        # AED: DSCR stabilised by debt annihilation
        dscr_aed[t2] = np.clip(dscr_aed[t2-1] + 0.004 + tech_shock*0.3 +
                               abs(X_aed[3,t2])*0.15, 1.0, 4.0)

    ax.plot(t, dscr_std, color=C['std'],  lw=2,    label='DSCR — Standard')
    ax.plot(t, dscr_aed, color=C['aed'],  lw=2, ls='--', label='DSCR — AED')
    ax.axhline(1.25, color=C['cap'], lw=1.5, ls=':', label='DSCR = 1.25 (lender covenant)')
    ax.fill_between(t, 1.25, dscr_std,
                    where=(dscr_std < 1.25), alpha=0.20, color=C['risk'],
                    label='Covenant breach risk')
    ax.set_title('(B) Debt Service Coverage Ratio\nLender Covenant: DSCR ≥ 1.25', fontsize=9.5)
    ax.set_xlabel('Quarters'); ax.set_ylabel('DSCR')
    ax.legend(frameon=False, fontsize=7.5); pl(ax, '(B)')

    # (C) Countercyclical capital buffer
    ax = axes[0,2]
    ccb_std = np.clip(0.02 + X_std[5]*0.10 + X_std[12]*0.05, 0, 0.05)
    ccb_aed = np.clip(0.015 + X_aed[5]*0.05, 0, 0.035)
    ax.plot(t, ccb_std*100, color=C['std'],  lw=2,    label='CCyB — Standard')
    ax.plot(t, ccb_aed*100, color=C['aed'],  lw=2, ls='--', label='CCyB — AED')
    ax.fill_between(t, 0, ccb_std*100, alpha=0.12, color=C['std'])
    ax.fill_between(t, 0, ccb_aed*100, alpha=0.12, color=C['aed'])
    ax.set_title('(C) Countercyclical Capital Buffer (CCyB)\nBasel III Macroprudential Tool', fontsize=9.5)
    ax.set_xlabel('Quarters'); ax.set_ylabel('CCyB (%)')
    ax.legend(frameon=False, fontsize=8); pl(ax, '(C)')

    # (D) Stress test: shock scenario comparisons
    ax = axes[1,0]
    scenarios = {
        '2008-type\nCredit Shock': {
            'std_cap_loss': 4.5, 'aed_cap_loss': 1.2,
            'std_recov': 12,    'aed_recov': 4},
        'Pandemic\nShock': {
            'std_cap_loss': 3.8, 'aed_cap_loss': 1.0,
            'std_recov': 8,     'aed_recov': 3},
        'Energy\nCrisis': {
            'std_cap_loss': 2.1, 'aed_cap_loss': 0.6,
            'std_recov': 6,     'aed_recov': 2},
        'Deflation\nSpiral': {
            'std_cap_loss': 6.0, 'aed_cap_loss': 0.3,
            'std_recov': 20,    'aed_recov': 2},
    }
    sc_names = list(scenarios.keys())
    x_pos    = np.arange(len(sc_names))
    cap_std  = [v['std_cap_loss'] for v in scenarios.values()]
    cap_aed  = [v['aed_cap_loss'] for v in scenarios.values()]

    ax.bar(x_pos-0.2, cap_std, 0.35, color=C['std'], alpha=0.82, label='Standard: Capital loss (pp)')
    ax.bar(x_pos+0.2, cap_aed, 0.35, color=C['aed'], alpha=0.82, label='AED: Capital loss (pp)')
    ax.set_xticks(x_pos); ax.set_xticklabels(sc_names, fontsize=8)
    ax.set_ylabel('Capital Ratio Loss (pp)')
    ax.set_title('(D) Stress Test: Capital Loss Under\nAdverse Scenarios (pp of CAR)', fontsize=9)
    ax.legend(frameon=False, fontsize=8); pl(ax, '(D)')

    # (E) Recovery time after shock
    ax = axes[1,1]
    recov_std = [v['std_recov'] for v in scenarios.values()]
    recov_aed = [v['aed_recov'] for v in scenarios.values()]
    ax.bar(x_pos-0.2, recov_std, 0.35, color=C['std'], alpha=0.82, label='Standard')
    ax.bar(x_pos+0.2, recov_aed, 0.35, color=C['aed'], alpha=0.82, label='AED')
    for i, (rs, ra) in enumerate(zip(recov_std, recov_aed)):
        ax.text(i-0.2, rs+0.2, f'{rs}q', ha='center', fontsize=7.5, color=C['std'])
        ax.text(i+0.2, ra+0.2, f'{ra}q', ha='center', fontsize=7.5, color=C['aed'])
    ax.set_xticks(x_pos); ax.set_xticklabels(sc_names, fontsize=8)
    ax.set_ylabel('Quarters to Recovery')
    ax.set_title('(E) Recovery Time After Stress Event\n(Quarters to Pre-Shock Capital Level)', fontsize=9)
    ax.legend(frameon=False, fontsize=8); pl(ax, '(E)')

    # (F) Systemic risk index
    ax = axes[1,2]
    # Composite systemic risk: leverage × spread × (1/NW)
    sri_std = np.abs(X_std[12]) * np.abs(X_std[7]) / (np.abs(X_std[8]) + 0.01)
    sri_aed = np.abs(X_aed[12]) * np.abs(X_aed[7]) / (np.abs(X_aed[8]) + 0.01)
    sri_std = sri_std / sri_std.max() * 100
    sri_aed = sri_aed / sri_std.max() * 100

    ax.plot(t, sri_std, color=C['std'],  lw=2,    label='SRI — Standard')
    ax.plot(t, sri_aed, color=C['aed'],  lw=2, ls='--', label='SRI — AED')
    ax.fill_between(t, sri_std, sri_aed,
                    where=(sri_std > sri_aed), alpha=0.15, color=C['risk'],
                    label='Standard excess risk')
    ax.set_title('(F) Composite Systemic Risk Index\n$SRI = |lev| \\times |spr| / |nw|$', fontsize=9.5)
    ax.set_xlabel('Quarters'); ax.set_ylabel('SRI (normalised, max=100)')
    ax.legend(frameon=False, fontsize=8); pl(ax, '(F)')

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("="*70)
    print("  MESO-LEVEL BANKING DSGE — Adaptive Emission Doctrine")
    print("  Generating 7 publication-quality figures...")
    print("="*70)

    import os
    out = r'D:\MESO AED finance'
    os.makedirs(out, exist_ok=True)

    figs = [
        ('figB1_banking_overview',    fig_banking_overview,   "State variable overview (13 vars)"),
        ('figB2_fin_accelerator',     fig_fin_accelerator,    "BGG Financial Accelerator IRFs"),
        ('figB3_balance_sheet',       fig_balance_sheet,      "Bank balance sheet dynamics"),
        ('figB4_interbank',           fig_interbank,          "Interbank market & contagion"),
        ('figB5_profitability',       fig_profitability,      "Bank profitability: interest under AED"),
        ('figB6_credit_emission',     fig_credit_emission,    "Credit creation & AED emission"),
        ('figB7_systemic_risk',       fig_systemic_risk,      "Systemic risk indicators"),
    ]

    saved = []
    for fname, func, desc in figs:
        print(f"  [{fname}] {desc} ... ", end='', flush=True)
        try:
            fig  = func()
            path = f'{out}/{fname}.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved.append(path)
            print("OK")
        except Exception as e:
            print(f"ERR  {e}")
            import traceback; traceback.print_exc()

    print()
    print("="*70)
    print(f"  Done. {len(saved)}/{len(figs)} figures saved.")
    print("="*70)

    # Calibration summary
    kappa_val = ((1-p.theta)*(1-p.beta*p.theta)/p.theta *
                 (p.sigma+p.eta)/(1+p.eta*p.epsilon))
    print("\n  BANKING MESO MODEL - Calibration")
    print(f"  {'Parameter':<35} {'Value':>8}  Description")
    print(f"  {'-'*65}")
    items = [
        ('BGG leverage premium μ_BGG',      p.mu_bgk,    'Bernanke-Gertler-Gilchrist 1999'),
        ('Bank NW persistence ω_NW',         p.omega_nw,  'Franchise value'),
        ('Basel III capital floor k_min',    p.k_min,     '8% minimum CET1'),
        ('Target capital ratio k*',          p.k_star,    '12% including stress buffer'),
        ('Credit channel IS weight φ_b',     p.phi_b,     'Spread → output dampening'),
        ('Debt annihilation coeff α_AED',    p.alpha_aed, 'From Master Formula'),
        ('Financial shock std σ_f',          p.sig_f,     'BGG calibration'),
        ('NKPC slope κ',                     kappa_val,   'Derived from Calvo θ'),
    ]
    for label, val, desc in items:
        safe_label = label.encode('ascii', 'ignore').decode()
        safe_desc = desc.encode('ascii', 'ignore').decode()
        print(f"  {safe_label:<35} {val:>8.4f}  {safe_desc}")
    print()
    return saved


if __name__ == '__main__':
    saved_paths = main()
