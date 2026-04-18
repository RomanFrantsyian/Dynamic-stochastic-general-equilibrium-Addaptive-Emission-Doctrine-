"""
=============================================================================
MESO-DSGE MODEL: AED Pharma — PURE INNOVATOR VARIANT (α = 1.0)
=============================================================================
Variant of dsge_aed_pharma.py with a single structural change:

  alpha_innovator = 1.00   (innovator captures 100% of burden NPV)
  alpha_echo      = 0.00   (no implementer echo-royalty)
  delta_state     = 0.00   (no state seigniorage)

Emission goes entirely to the firm that eliminates the disease.
All other model mechanics (Master Formula, debt annihilation,
disease block, industry equilibrium) are identical to base model.

Research question: what is the upper bound of cure incentive
when society transfers maximum possible reward to the innovator?
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator, FuncFormatter, FormatStrFormatter
from scipy import optimize, stats
from scipy.special import expit          # logistic sigmoid
import warnings
warnings.filterwarnings('ignore')

# ── Style (journal standard, matching dsge_aed_model.py) ─────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['DejaVu Serif', 'Times New Roman', 'Georgia'],
    'font.size':        10,
    'axes.titlesize':   11,
    'axes.labelsize':   10,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'legend.fontsize':  9,
    'figure.dpi':       150,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.25,
    'grid.linestyle':   '--',
    'grid.linewidth':   0.6,
    'lines.linewidth':  1.8,
    'figure.facecolor': 'white',
    'axes.facecolor':   '#FAFAFA',
    'savefig.dpi':      150,
    'savefig.bbox':     'tight',
    'savefig.facecolor':'white',
})

C = {
    'std':    '#1A3A5C',   # traditional pharma (navy)
    'aed':    '#C0392B',   # AED pharma (crimson)
    'cure':   '#2C7A4B',   # cure pathway (forest)
    'chronic':'#E67E22',   # chronic pathway (amber)
    'burden': '#7B2D8B',   # disease burden (purple)
    'patient':'#1ABC9C',   # patient welfare (teal)
    'debt':   '#E74C3C',   # debt (red)
    'shock':  '#E67E22',   # shock (amber)
    'zero':   '#555555',
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
class PharmaParams:
    # ── Macro (quarterly) ─────────────────────────────────────────────────
    beta        = 0.99    # discount factor
    r_free      = 0.015   # risk-free rate quarterly (~6% p.a.)
    pi_target   = 0.005   # AED: near-zero inflation target
    V_health    = 0.80    # health-sector money velocity (lower than macro)

    # ── Industry structure ────────────────────────────────────────────────
    M_firms     = 6       # number of competing pharma firms
    epsilon_mkt = 4.0     # demand elasticity (drug market)
    theta_patent = 20     # standard patent life (years)

    # ── R&D economics ─────────────────────────────────────────────────────
    # Costs in $B
    RD_cure_cost      = 3.0   # avg cost to develop cure ($3B)
    RD_chronic_cost   = 1.2   # avg cost to develop chronic therapy ($1.2B)
    RD_success_cure   = 0.12  # probability of R&D success (cure) — high risk
    RD_success_chron  = 0.28  # probability of R&D success (chronic)
    RD_time_cure      = 12    # years to develop cure
    RD_time_chron     = 8     # years to develop chronic therapy

    # ── Disease burden pool (calibrated to US data, $B/year) ─────────────
    diseases = {
        'Diabetes':     {'burden': 327, 'prevalence': 37e6,  'cure_prob': 0.08},
        'Alzheimer':    {'burden': 355, 'prevalence': 6.5e6, 'cure_prob': 0.04},
        'Cancer':       {'burden': 208, 'prevalence': 18e6,  'cure_prob': 0.15},
        'HeartDisease': {'burden': 216, 'prevalence': 20e6,  'cure_prob': 0.10},
        'HepC':         {'burden':  15, 'prevalence': 2.4e6, 'cure_prob': 0.95},  # already cured
    }
    discount_horizon = 10     # NPV horizon for AED emission (years)

    # ── Traditional revenue (chronic) ─────────────────────────────────────
    # Revenue per patient per year
    chron_rev_per_pt  = 15_000   # $15K/patient/year
    cure_price_std    = 84_000   # Sovaldi benchmark ($84K one-time)

    # ── AED emission parameters — PURE INNOVATOR (α=1) ───────────────────
    alpha_innovator   = 1.00    # innovator captures 100% of burden NPV
    alpha_echo        = 0.00    # no echo-royalty to implementers
    alpha_implement   = 0.00    # implementers receive nothing from innovator's cure
    delta_state       = 0.00    # no state seigniorage
    D_annihilation    = 1.0     # α coefficient (full annihilation on cure)
    npv_discount      = 0.03    # discount rate for burden NPV

    # ── Debt structure (pharma firms) ─────────────────────────────────────
    D0_per_firm       = 10.0    # initial debt per firm ($B)
    r_debt            = 0.06    # annual debt cost

    # ── Simulation ────────────────────────────────────────────────────────
    T_years     = 30      # simulation horizon
    T           = T_years * 4   # quarterly
    seed        = 42

p = PharmaParams()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  AED PHARMA EMISSION FORMULA
# ─────────────────────────────────────────────────────────────────────────────
class AEDPharmaFormula:
    """
    E_pharma,t = Σ_d ΔB_d,t / V_health  +  D_annihilated_RD,t

    Inputs  : disease burden reduction ΔB per period
    Outputs : emission, innovator reward, debt relief
    """

    def __init__(self, params: PharmaParams):
        self.p = params

    def burden_npv(self, annual_burden: float) -> float:
        """NPV of 10-year disease burden stream at discount rate r."""
        r = self.p.npv_discount
        n = self.p.discount_horizon
        return annual_burden * (1 - (1+r)**-n) / r

    def innovator_reward_cure(self, annual_burden: float) -> float:
        """Lump-sum AED reward for eliminating a disease."""
        npv = self.burden_npv(annual_burden)
        return self.p.alpha_innovator * npv

    def innovator_reward_chronic(self, patients: float,
                                  rev_per_pt: float,
                                  years: float = 20) -> float:
        """
        Traditional revenue stream for originator firm, in $B.
        market_share=2%: realistic peak share for a new drug in a competitive
        therapeutic area (paper implies ~$5B/yr → $100B over 20 yrs at 3%).
        """
        r = self.p.npv_discount
        market_share = 0.02   # 2% of addressable patients at peak
        annual = patients * rev_per_pt / 1e9 * market_share
        return annual * (1 - (1+r)**-years) / r

    def debt_annihilated(self, burden_reduction: float,
                          firm_debt: float,
                          total_burden_pool: float = 1121.0) -> float:
        """
        Debt wiped proportional to share of total disease burden eliminated.
        Implements paper Prop 5: D_{t+1} = D_t(1 - α·ΔC/C_t)
        total_burden_pool: sum of all disease burdens ($B), default = 327+355+208+216+15
        """
        share = burden_reduction / max(total_burden_pool, burden_reduction)
        D_ann = self.p.D_annihilation * share * firm_debt
        return min(D_ann, firm_debt)

    def emission_period(self, delta_burden: float,
                         D_annihilated: float) -> float:
        """
        Single-period emission from Master Formula:
          E = P_target·ΔQ/V + D_annihilated
        delta_burden = deflationary vacuum ($B); D_annihilated = actual debt cleared ($B).
        Both terms backed by real value — non-inflationary by construction (Prop 8).
        """
        return max(0.0, delta_burden / self.p.V_health + D_annihilated)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FIRM LEVEL: R&D DECISION MODEL
# ─────────────────────────────────────────────────────────────────────────────
class PharmaFirm:
    """
    Each firm solves:
      max_{I_cure, I_chron}  E[NPV(portfolio)]
      s.t.  budget_constraint, debt_dynamics

    Under standard: maximises revenue from chronic pipeline
    Under AED:      maximises expected emission reward from cures
    """

    def __init__(self, firm_id: int, params: PharmaParams, regime: str):
        self.id     = firm_id
        self.p      = params
        self.regime = regime
        self.formula= AEDPharmaFormula(params)
        rng = np.random.default_rng(params.seed + firm_id)

        # State
        self.debt   = params.D0_per_firm
        self.cash   = rng.uniform(1, 5)  # $B
        self.pipeline_cure   = 0   # active cure projects
        self.pipeline_chron  = 0   # active chronic projects

        # Histories
        self.hist = {k: [] for k in ['debt','cash','rd_cure','rd_chron',
                                      'revenue','npv_capture','burden_cleared']}

    def _npv_cure_aed(self, disease_burden: float) -> float:
        """Expected NPV of a cure project under AED."""
        reward  = self.formula.innovator_reward_cure(disease_burden)
        d_ann   = self.formula.debt_annihilated(disease_burden, self.debt)
        # Cost: R&D expenditure
        cost    = self.p.RD_cure_cost / self.p.RD_success_cure  # risk-adjusted
        return self.p.RD_success_cure * (reward + d_ann) - cost

    def _npv_cure_std(self, disease_burden: float,
                       patients: float) -> float:
        """Expected NPV of a cure — traditional market (Sovaldi model), in $B."""
        price     = self.p.cure_price_std
        revenue   = patients * price / 1e9   # convert raw dollars → $B
        # Market shrinks → future = 0; competitor enters → price war in year 3
        r         = self.p.npv_discount
        npv_rev   = revenue / (1 + r)**2   # 2-year delay to approval
        cost      = self.p.RD_cure_cost / self.p.RD_success_cure
        return self.p.RD_success_cure * npv_rev - cost

    def _npv_chronic(self, patients: float) -> float:
        """Expected NPV of chronic pipeline."""
        reward  = self.formula.innovator_reward_chronic(
                      patients, self.p.chron_rev_per_pt)
        cost    = self.p.RD_chronic_cost / self.p.RD_success_chron
        return self.p.RD_success_chron * reward - cost

    def rd_allocation(self, disease_burden: float, patients: float) -> dict:
        """Optimal R&D budget allocation between cure and chronic."""
        npv_c_aed = self._npv_cure_aed(disease_burden)
        npv_c_std = self._npv_cure_std(disease_burden, patients)
        npv_ch    = self._npv_chronic(patients)

        if self.regime == 'aed':
            # Allocate proportional to NPV
            total = max(npv_c_aed + npv_ch, 1e-6)
            frac_cure  = max(0, npv_c_aed) / total
        else:
            total = max(npv_c_std + npv_ch, 1e-6)
            frac_cure  = max(0, npv_c_std) / total

        return {'frac_cure': frac_cure, 'frac_chron': 1 - frac_cure,
                'npv_cure_aed': npv_c_aed,
                'npv_chron': npv_ch}


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DISEASE BLOCK: Burden Dynamics & Emission
# ─────────────────────────────────────────────────────────────────────────────
class DiseaseBlock:
    """
    Disease burden B_d,t evolves as:
      B_d,t = B_d,t-1 · (1 + g_pop - cure_arrival_rate · I_{cure,t})

    A cure event: Poisson arrival with rate λ(I_cure) — investment-dependent.
    Cure arrival → burden drops permanently → triggers AED emission.
    """

    def __init__(self, params: PharmaParams):
        self.p = params
        self.formula = AEDPharmaFormula(params)
        # Initialise disease states
        self.burden    = {d: v['burden']     for d, v in params.diseases.items()}
        self.prev      = {d: v['prevalence'] for d, v in params.diseases.items()}
        self.base_cure = {d: v['cure_prob']  for d, v in params.diseases.items()}
        self.cured     = {d: False           for d, v in params.diseases.items()}

    def step(self, investment_cure_frac: float,
              total_rd_industry: float, rng, t: int) -> dict:
        """
        Advance one year.
        Returns: {burden_reduction, emission_generated, cures_achieved}
        """
        pop_growth   = 0.008   # annual
        budget_cure  = investment_cure_frac * total_rd_industry
        # Diminishing returns: log-concave in R&D budget
        cure_boost   = 1.0 + 2.5 * np.log1p(budget_cure / 5.0)

        total_reduction   = 0.0
        total_emission    = 0.0
        cures_this_period = []

        for disease, burden in self.burden.items():
            if self.cured[disease]:
                # Residual burden (monitoring, late-stage)
                self.burden[disease] *= 0.98
                continue

            # Cure probability (logistic in investment)
            base_p = self.base_cure[disease]
            cure_p = min(0.99, base_p * cure_boost)
            got_cure = (rng.random() < cure_p / 4)   # quarterly scaling

            if got_cure:
                self.cured[disease] = True
                reduction  = burden
                # Actual D_ann: proportional to burden share × avg firm debt
                avg_firm_debt = self.p.D0_per_firm   # representative firm
                total_pool = sum(v['burden'] for v in self.p.diseases.values())
                actual_D_ann = self.formula.debt_annihilated(
                    reduction, avg_firm_debt, total_pool)
                emission   = self.formula.emission_period(reduction, actual_D_ann)
                total_reduction += reduction
                total_emission  += emission
                cures_this_period.append(disease)
                self.burden[disease] *= 0.05   # residual
            else:
                # Chronic burden grows with population
                self.burden[disease] *= (1 + pop_growth)

        return {
            'burden_total':    sum(self.burden.values()),
            'burden_reduction':total_reduction,
            'emission':        total_emission,
            'cures':           cures_this_period,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  INDUSTRY EQUILIBRIUM SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
class PharmaIndustryModel:

    def __init__(self, params: PharmaParams):
        self.p       = params
        self.formula = AEDPharmaFormula(params)

    def simulate(self, regime: str = 'standard', seed: int = None) -> dict:
        seed = seed or self.p.seed
        rng  = np.random.default_rng(seed)
        T    = self.p.T_years

        firms  = [PharmaFirm(i, self.p, regime) for i in range(self.p.M_firms)]
        block  = DiseaseBlock(self.p)

        # Output arrays
        out = {
            'rd_cure_frac':   np.zeros(T),
            'rd_total':       np.zeros(T),
            'burden_total':   np.zeros(T),
            'emission':       np.zeros(T),
            'cum_emission':   np.zeros(T),
            'debt_industry':  np.zeros(T),
            'cure_count':     np.zeros(T, dtype=int),
            'npv_cure':       np.zeros(T),
            'npv_chron':      np.zeros(T),
            'welfare':        np.zeros(T),     # burden saved = welfare proxy
            'revenue_std':    np.zeros(T),
            'patient_cost':   np.zeros(T),
        }

        # Representative disease for NPV calcs
        rep_burden   = 250.0     # $B representative burden
        rep_patients = 15e6

        cum_emit = 0.0
        for t in range(T):
            # Firm decisions
            allocs = [f.rd_allocation(rep_burden, rep_patients) for f in firms]
            avg_frac_cure = np.mean([a['frac_cure']      for a in allocs])
            avg_npv_cure  = np.mean([a['npv_cure_aed'] for a in allocs])
            avg_npv_chron = np.mean([a['npv_chron']      for a in allocs])

            # Total R&D: grows with profitability
            total_rd = self.p.M_firms * 2.0 * (1 + 0.05*t)  # $B/year base
            if regime == 'aed':
                total_rd *= (1 + 0.03*t)   # AED boosts R&D investment

            # Disease block step
            ds = block.step(avg_frac_cure, total_rd, rng, t)

            # Emission reward
            if regime == 'aed':
                reward_per_firm = ds['emission'] * self.p.alpha_innovator / max(len(ds['cures']), 1)
                for i, f in enumerate(firms):
                    if i < len(ds['cures']):
                        f.debt = max(0, f.debt - reward_per_firm * 0.8)

            # Industry totals
            out['rd_cure_frac'][t]  = avg_frac_cure
            out['rd_total'][t]      = total_rd
            out['burden_total'][t]  = ds['burden_total']
            out['emission'][t]      = ds['emission']
            cum_emit               += ds['emission']
            out['cum_emission'][t]  = cum_emit
            out['cure_count'][t]    = len(ds['cures'])
            out['npv_cure'][t]      = avg_npv_cure
            out['npv_chron'][t]     = avg_npv_chron
            out['debt_industry'][t] = sum(f.debt for f in firms)

            # Welfare: cumulative burden removed
            out['welfare'][t]       = (out['welfare'][t-1] if t>0 else 0) + ds['burden_reduction']

            # Patient cost proxy
            if regime == 'aed':
                out['patient_cost'][t] = rep_patients * 2000   # ~$2K/pt/yr under AED
            else:
                out['patient_cost'][t] = rep_patients * self.p.chron_rev_per_pt

        return out


# ─────────────────────────────────────────────────────────────────────────────
# 6.  NPV COMPARISON ENGINE (Proposition Cure)
# ─────────────────────────────────────────────────────────────────────────────
class NPVComparison:
    """
    Reproduces and extends Table 6 from the paper with full NPV trajectories.
    Computes sensitivity across diseases and confirms 4-5× ratio claim.
    """

    def __init__(self, params: PharmaParams):
        self.p = params
        self.formula = AEDPharmaFormula(params)

    def build_comparison_table(self):
        rows = []
        for dname, dvals in self.p.diseases.items():
            burden   = dvals['burden']
            prev     = dvals['prevalence']

            # Traditional chronic NPV
            npv_chron = self.formula.innovator_reward_chronic(
                            prev, self.p.chron_rev_per_pt, years=20)

            # AED cure NPV
            npv_cure_aed = self.formula.innovator_reward_cure(burden) \
                           + self.p.D0_per_firm   # + debt annihilation
            npv_cure_aed_exp = npv_cure_aed * self.p.RD_success_cure \
                               - self.p.RD_cure_cost

            rows.append({
                'disease':      dname,
                'burden_B':     burden,
                'npv_chron_B':  npv_chron,
                'npv_cure_aed_B': npv_cure_aed_exp,
                'ratio':        npv_cure_aed_exp / max(npv_chron, 0.1),
            })
        return rows

    def npv_sensitivity(self, disease='Diabetes',
                         alpha_range=None, discount_range=None):
        """Sensitivity of cure NPV to innovator share α and discount rate."""
        if alpha_range is None:
            alpha_range = np.linspace(0.2, 0.8, 40)
        if discount_range is None:
            discount_range = np.linspace(0.01, 0.10, 40)

        burden = self.p.diseases[disease]['burden']
        NPV_map = np.zeros((len(alpha_range), len(discount_range)))

        for i, alpha in enumerate(alpha_range):
            for j, r in enumerate(discount_range):
                n   = self.p.discount_horizon
                npv = burden * (1 - (1+r)**-n) / r * alpha
                NPV_map[i, j] = npv * self.p.RD_success_cure - self.p.RD_cure_cost

        return alpha_range, discount_range, NPV_map


# ─────────────────────────────────────────────────────────────────────────────
# 7.  GAME THEORY: mRNA PLATFORM PATENT RACE
# ─────────────────────────────────────────────────────────────────────────────
class PlatformPatentGame:
    """
    Models the mRNA/CRISPR platform technology dilemma:
    Firm A (Moderna/BioNTech type) invented the platform.
    Firms B...N can license it (implementers) for specific diseases.

    Under AED 75/25 rule:
      - A gets 25% echo-royalty on every cure others build with its platform
      - B gets 75% of its own cure emission

    Nash equilibrium: Open >> Close (as in paper Section 4.5.3)
    Extended here with continuous payoff landscape.
    """

    def __init__(self, params: PharmaParams):
        self.p = params
        self.formula = AEDPharmaFormula(params)

    def payoff_matrix(self, n_diseases: int = 5,
                       burden_per_disease: float = 200):
        """
        Compute payoffs as function of n implementing firms and sharing strategy.
        Returns arrays for visualisation.
        """
        n_range   = np.arange(1, 20)
        payoff_open   = np.zeros(len(n_range))
        payoff_closed = np.zeros(len(n_range))

        for k, n_impl in enumerate(n_range):
            # If A opens: each implementer creates burden deflation D
            D_per = self.formula.innovator_reward_cure(burden_per_disease)
            echo  = self.p.alpha_echo * D_per * n_impl  # A receives echo
            # But own cure only
            own   = D_per                              # A's own disease
            payoff_open[k] = own + echo

            # If A closes: monopoly on own disease, implementers stagnate
            payoff_closed[k] = D_per * 0.8             # own monopoly (some market)

        return n_range, payoff_open, payoff_closed

    def echo_royalty_dynamics(self, T: int = 30) -> dict:
        """Simulate echo-royalty accumulation over time."""
        rng = np.random.default_rng(self.p.seed)
        t   = np.arange(T)

        # Implementers join over time (S-curve adoption)
        adopters = np.round(15 * expit((t - 8) * 0.5)).astype(int)
        burden   = 200   # $B per disease

        echo_cumulative   = np.zeros(T)
        revenue_closed    = np.zeros(T)
        debt_aed          = np.full(T, self.p.D0_per_firm)

        for i in range(1, T):
            n = adopters[i]
            echo_cumulative[i] = echo_cumulative[i-1] + \
                self.p.alpha_echo * n * \
                self.formula.innovator_reward_cure(burden) * \
                self.p.RD_success_cure / T

            # Closed: slow monopoly revenue
            revenue_closed[i] = revenue_closed[i-1] + \
                0.8 * self.formula.innovator_reward_cure(burden) / T

            # Debt dynamics
            debt_aed[i] = max(0, debt_aed[i-1]
                              - echo_cumulative[i] * 0.05
                              + self.p.D0_per_firm * self.p.r_debt / 4)

        return {'t': t, 'adopters': adopters,
                'echo_cumul': echo_cumulative,
                'rev_closed': revenue_closed,
                'debt_aed':   debt_aed}


# ─────────────────────────────────────────────────────────────────────────────
# 8.  FIGURES
# ─────────────────────────────────────────────────────────────────────────────
def add_zeroline(ax, lw=0.8):
    ax.axhline(0, color=C['zero'], linewidth=lw, linestyle='-', zorder=1)

def panel_label(ax, label, x=0.03, y=0.96, fontsize=10):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top')

def billions(x, pos): return f'${x:.0f}B'
def pct(x, pos):      return f'{x:.0f}%'

# ─── FIG 1: AED Pharma Architecture Overview ─────────────────────────────────
def fig_alpha1_pharma_architecture():
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        'Figure 1 [α=1] — AED Pharmaceutical Model Architecture\n'
        'Meso-DSGE: Macro Emission Block ↔ Industry Equilibrium ↔ Disease Burden',
        fontsize=12, fontweight='bold', y=1.01
    )

    model   = PharmaIndustryModel(p)
    out_std = model.simulate('standard', seed=p.seed)
    out_aed = model.simulate('aed',      seed=p.seed)
    T       = p.T_years
    t       = np.arange(T)

    axes = fig.subplots(2, 3)

    # (A) R&D allocation: cure fraction
    ax = axes[0, 0]
    ax.plot(t, out_std['rd_cure_frac']*100, color=C['std'],  lw=2, label='Standard')
    ax.plot(t, out_aed['rd_cure_frac']*100, color=C['aed'],  lw=2, ls='--', label='AED')
    ax.set_title('(A) R&D Allocation to Cures (%)')
    ax.set_ylabel('% of R&D Budget'); ax.set_xlabel('Year')
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.legend(frameon=False)
    panel_label(ax, '(A)')

    # (B) Total disease burden
    ax = axes[0, 1]
    ax.plot(t, out_std['burden_total'], color=C['std'], lw=2, label='Standard')
    ax.plot(t, out_aed['burden_total'], color=C['aed'], lw=2, ls='--', label='AED')
    ax.fill_between(t, out_aed['burden_total'], out_std['burden_total'],
                    alpha=0.12, color=C['cure'], label='Burden saved')
    ax.set_title('(B) Total Disease Burden ($B/year)')
    ax.set_ylabel('$B / year'); ax.set_xlabel('Year')
    ax.yaxis.set_major_formatter(FuncFormatter(billions))
    ax.legend(frameon=False)
    panel_label(ax, '(B)')

    # (C) Cumulative AED emission from health sector
    ax = axes[0, 2]
    ax.plot(t, out_aed['cum_emission'], color=C['aed'], lw=2.5, label='AED emission (cumul.)')
    ax.fill_between(t, 0, out_aed['cum_emission'], alpha=0.15, color=C['aed'])
    ax.set_title('(C) Cumulative AED Emission\n(Health Sector, Master Formula)')
    ax.set_ylabel('Cumulative $B'); ax.set_xlabel('Year')
    ax.yaxis.set_major_formatter(FuncFormatter(billions))
    ax.legend(frameon=False)
    panel_label(ax, '(C)')

    # (D) Industry R&D investment
    ax = axes[1, 0]
    ax.plot(t, out_std['rd_total'], color=C['std'], lw=2, label='Standard')
    ax.plot(t, out_aed['rd_total'], color=C['aed'], lw=2, ls='--', label='AED')
    ax.set_title('(D) Industry R&D Expenditure ($B/year)')
    ax.set_ylabel('$B / year'); ax.set_xlabel('Year')
    ax.yaxis.set_major_formatter(FuncFormatter(billions))
    ax.legend(frameon=False)
    panel_label(ax, '(D)')

    # (E) Industry debt dynamics
    ax = axes[1, 1]
    ax.plot(t, out_std['debt_industry'], color=C['std'], lw=2, label='Standard')
    ax.plot(t, out_aed['debt_industry'], color=C['aed'], lw=2, ls='--', label='AED')
    ax.set_title('(E) Total Industry Debt ($B)')
    ax.set_ylabel('$B'); ax.set_xlabel('Year')
    ax.yaxis.set_major_formatter(FuncFormatter(billions))
    ax.legend(frameon=False)
    panel_label(ax, '(E)')

    # (F) Welfare: cumulative burden removed
    ax = axes[1, 2]
    ax.plot(t, out_std['welfare'], color=C['std'], lw=2, label='Standard (welfare)')
    ax.plot(t, out_aed['welfare'], color=C['aed'], lw=2, ls='--', label='AED (welfare)')
    ax.set_title('(F) Cumulative Welfare Gain\n(Burden Removed, $B)')
    ax.set_ylabel('Cumulative $B'); ax.set_xlabel('Year')
    ax.yaxis.set_major_formatter(FuncFormatter(billions))
    ax.legend(frameon=False)
    panel_label(ax, '(F)')

    plt.tight_layout()
    return fig


# ─── FIG 2: NPV Comparison — Cure vs Chronic ─────────────────────────────────
def fig_npv_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        'Figure 2 [α=1] — NPV Analysis: Cure vs. Chronic Revenue Model\n'
        'AED Makes Cures 4–5× More Profitable (Proposition Cure)',
        fontsize=11, fontweight='bold'
    )

    npvc = NPVComparison(p)
    table = npvc.build_comparison_table()

    # (A) Bar chart: Chronic (20-yr) vs AED Cure
    ax = axes[0, 0]
    diseases = [r['disease'] for r in table]
    npv_ch   = [r['npv_chron_B']     for r in table]
    npv_ca   = [r['npv_cure_aed_B']  for r in table]

    x  = np.arange(len(diseases))
    w  = 0.35
    ax.bar(x - w/2, npv_ch, w, color=C['chronic'],
           label='Chronic therapy NPV (20-yr stream, 2% market share)', alpha=0.85)
    ax.bar(x + w/2, npv_ca, w, color=C['aed'],
           label='Cure — AED α=1 reward (burden NPV × α, lump-sum)', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(diseases, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Expected NPV ($B)')
    ax.set_title('(A) Expected NPV: Chronic (20yr) vs. AED Cure\n'
                 'Chronic = 20-yr annuity at 2% market share, $15K/pt/yr')
    ax.yaxis.set_major_formatter(FuncFormatter(billions))
    ax.legend(frameon=False, fontsize=8)
    add_zeroline(ax)
    panel_label(ax, '(A)')

    # (B) Ratio AED cure / Chronic
    ax = axes[0, 1]
    ratios = [r['ratio'] for r in table]
    colors_bar = [C['aed'] if ra > 1 else C['std'] for ra in ratios]
    bars = ax.bar(x, ratios, color=colors_bar, alpha=0.85, width=0.5)
    ax.axhline(1, color='k', lw=0.9, ls='--', label='Break-even (ratio=1)')
    ax.axhline(4.8, color=C['cure'], lw=1.2, ls=':', label='Paper claim (4.8×)')
    for bar, r in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{r:.1f}×', ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(diseases, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Cure NPV / Chronic NPV Ratio')
    ax.set_title('(B) Relative Profitability of Cures Under AED\n(vs. Chronic Model)')
    ax.legend(frameon=False, fontsize=8)
    panel_label(ax, '(B)')

    # (C) NPV trajectory: Hepatitis C — Chronic stream vs AED lump-sum
    ax = axes[1, 0]
    years = np.arange(1, 26)
    r     = p.npv_discount
    patients_hep = p.diseases['HepC']['prevalence']
    burden_hep   = p.diseases['HepC']['burden']

    # Chronic: $20K/pt/yr, 2% market share, declining as patients naturally clear
    chron_annual = patients_hep * 20_000 / 1e9 * 0.02
    chron_npv = np.array([chron_annual * (1-(1+r)**-y)/r for y in years])

    # AED cure: lump-sum at year 1, grows via global adoption echo
    aed_lump = npvc.formula.innovator_reward_cure(burden_hep)
    echo_growth = np.array([aed_lump * (1 + 0.02*y) for y in years])

    ax.plot(years, chron_npv,   color=C['chronic'], lw=2,
            label='Chronic NPV (20-yr, 2% share, $20K/pt/yr)')
    ax.plot(years, echo_growth, color=C['aed'],     lw=2, ls='--',
            label='AED Cure NPV (lump-sum + echo-royalty)')
    ax.fill_between(years, chron_npv, echo_growth, alpha=0.12, color=C['aed'],
                    label='AED advantage')
    ax.set_xlabel('Years Post-Approval'); ax.set_ylabel('Cumulative NPV ($B)')
    ax.set_title('(C) Hepatitis C: Chronic 20-yr Stream vs. AED Cure\n'
                 'Chronic period = 20 years at 2% innovator market share')
    ax.yaxis.set_major_formatter(FuncFormatter(billions))
    ax.legend(frameon=False, fontsize=8)
    panel_label(ax, '(C)')

    # (D) Sensitivity map: NPV(cure, AED) vs α and discount rate (Diabetes)
    ax = axes[1, 1]
    alphas, discounts, NPV_map = npvc.npv_sensitivity('Diabetes')
    im = ax.contourf(discounts*100, alphas, NPV_map,
                     levels=20, cmap='RdYlGn')
    cs = ax.contour(discounts*100, alphas, NPV_map,
                    levels=[0, 50, 100, 200, 400], colors='k', linewidths=0.7)
    ax.clabel(cs, fmt='$%.0fB', fontsize=7)
    # Mark AED baseline
    ax.plot(p.npv_discount*100, p.alpha_innovator, 'w*', ms=14,
            label=f'AED baseline (α={p.alpha_innovator}, r={p.npv_discount:.0%})')
    ax.axhline(p.alpha_innovator,   color='white', lw=0.8, ls='--', alpha=0.6)
    ax.axvline(p.npv_discount*100,  color='white', lw=0.8, ls='--', alpha=0.6)
    plt.colorbar(im, ax=ax, label='Expected NPV ($B)')
    ax.set_xlabel('Discount Rate (%)'); ax.set_ylabel('Innovator Share α')
    ax.set_title('(D) Sensitivity: Cure NPV (Diabetes)\nAED Innovator Share × Discount Rate')
    ax.legend(frameon=False, fontsize=7.5)
    panel_label(ax, '(D)')

    plt.tight_layout()
    return fig


# ─── FIG 3: Disease Burden Deflation as Emission Trigger ─────────────────────
def fig_burden_emission():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        'Figure 3 [α=1] — Disease Burden Deflation as Monetary Emission Trigger\n'
        r'$E_{pharma} = \sum_d \Delta B_{d,t} / V_{health} + D^{ann}_{RD,t}$',
        fontsize=11, fontweight='bold', y=1.02
    )

    model   = PharmaIndustryModel(p)
    out_aed = model.simulate('aed',      seed=p.seed)
    out_std = model.simulate('standard', seed=p.seed)
    T       = p.T_years
    t       = np.arange(T)

    # (A) Annual emission pulses (cure events create spikes)
    ax = axes[0, 0]
    ax.bar(t, out_aed['emission'], color=C['aed'], alpha=0.85, label='AED emission pulse')
    ax.set_title('(A) Annual AED Emission\n(Health Sector — Cure Event Pulses)')
    ax.set_xlabel('Year'); ax.set_ylabel('Emission ($B)')
    ax.yaxis.set_major_formatter(FuncFormatter(billions))
    ax.legend(frameon=False)
    panel_label(ax, '(A)')

    # (B) Burden vs emission: scatter
    ax = axes[0, 1]
    burden_change = -np.diff(out_aed['burden_total'], prepend=out_aed['burden_total'][0])
    burden_change = np.maximum(burden_change, 0)
    sc = ax.scatter(burden_change, out_aed['emission'],
                    c=t, cmap='plasma', s=30, alpha=0.8, zorder=5)
    # Regression line
    mask = burden_change > 0
    if mask.sum() > 2:
        m, b, *_ = stats.linregress(burden_change[mask], out_aed['emission'][mask])
        xr = np.linspace(0, burden_change.max(), 100)
        ax.plot(xr, m*xr + b, color=C['aed'], lw=1.5, ls='--',
                label=f'OLS: slope={m:.2f}')
    plt.colorbar(sc, ax=ax, label='Year')
    ax.set_xlabel('Burden Reduction ($B)'); ax.set_ylabel('Emission ($B)')
    ax.set_title('(B) Burden Deflation → Emission\n(Master Formula Verification)')
    ax.xaxis.set_major_formatter(FuncFormatter(billions))
    ax.yaxis.set_major_formatter(FuncFormatter(billions))
    ax.legend(frameon=False, fontsize=8)
    panel_label(ax, '(B)')

    # (C) Debt annihilation vs emission cumulative
    ax = axes[0, 2]
    ax.plot(t, out_aed['cum_emission'],  color=C['aed'],  lw=2.5, label='Cum. emission (AED)')
    ax.plot(t, p.D0_per_firm * p.M_firms - out_aed['debt_industry'],
            color=C['cure'], lw=2, ls='--', label='Cum. debt annihilated')
    ax.set_title('(C) Emission vs. Debt Annihilation\n(Proposition 5: Non-inflationary)')
    ax.set_xlabel('Year'); ax.set_ylabel('Cumulative $B')
    ax.yaxis.set_major_formatter(FuncFormatter(billions))
    ax.legend(frameon=False)
    panel_label(ax, '(C)')

    # (D) Per-disease burden trajectories
    ax = axes[1, 0]
    rng = np.random.default_rng(p.seed)
    colors_d = plt.cm.tab10(np.linspace(0, 1, len(p.diseases)))
    for (dname, dvals), col in zip(p.diseases.items(), colors_d):
        b0 = dvals['burden']
        cp = dvals['cure_prob']
        # Simplified trajectory
        traj = np.zeros(T)
        traj[0] = b0
        cured = False
        for i in range(1, T):
            if cured:
                traj[i] = traj[i-1] * 0.98
            elif rng.random() < cp * 0.3:
                cured = True
                traj[i] = traj[i-1] * 0.05
            else:
                traj[i] = traj[i-1] * 1.008
        ax.plot(t, traj, color=col, lw=1.6, label=dname)
    ax.set_title('(D) Per-Disease Burden Trajectories\n(Standard vs. AED cure arrival rates)')
    ax.set_xlabel('Year'); ax.set_ylabel('Disease Burden ($B/year)')
    ax.yaxis.set_major_formatter(FuncFormatter(billions))
    ax.legend(frameon=False, fontsize=7.5)
    panel_label(ax, '(D)')

    # (E) V_health sensitivity: emission vs velocity
    ax = axes[1, 1]
    v_range = np.linspace(0.3, 2.0, 60)
    burden_reduction_proxy = 300   # $B representative
    D_ann = 5.0
    emission_v = [(burden_reduction_proxy / v + D_ann) for v in v_range]
    ax.plot(v_range, emission_v, color=C['aed'], lw=2.5)
    ax.axvline(p.V_health, color='k', lw=0.9, ls='--',
               label=f'Baseline $V_{{health}}$ = {p.V_health}')
    ax.fill_between(v_range, emission_v,
                    alpha=0.10, color=C['aed'])
    ax.set_xlabel('Health-Sector Money Velocity $V_{health}$')
    ax.set_ylabel('Emission ($B)')
    ax.set_title('(E) Emission Sensitivity to\nHealth-Sector Velocity $V_{health}$')
    ax.yaxis.set_major_formatter(FuncFormatter(billions))
    ax.legend(frameon=False)
    panel_label(ax, '(E)')

    # (F) Patient cost under AED vs standard
    ax = axes[1, 2]
    ax.plot(t, out_std['patient_cost']/1e9, color=C['std'],  lw=2, label='Standard (chronic)')
    ax.plot(t, out_aed['patient_cost']/1e9, color=C['aed'],  lw=2, ls='--', label='AED (cure-oriented)')
    ax.fill_between(t,
                    out_aed['patient_cost']/1e9,
                    out_std['patient_cost']/1e9,
                    alpha=0.15, color=C['cure'], label='Patient savings')
    ax.set_title('(F) Total Patient Cost ($B/year)\nStandard vs. AED')
    ax.set_xlabel('Year'); ax.set_ylabel('Patient Cost ($B)')
    ax.yaxis.set_major_formatter(FuncFormatter(billions))
    ax.legend(frameon=False)
    panel_label(ax, '(F)')

    plt.tight_layout()
    return fig


# ─── FIG 4: mRNA/CRISPR Platform Game Theory ─────────────────────────────────
def fig_platform_game():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        'Figure 4 [α=1] — Platform Technology Patent Game (mRNA / CRISPR)\n'
        'AED 75/25 Echo-Royalty Creates Open-Source Nash Equilibrium',
        fontsize=11, fontweight='bold'
    )

    game = PlatformPatentGame(p)

    # (A) Payoff curves: Open vs Closed as implementers join
    ax = axes[0, 0]
    n_range, payoff_open, payoff_closed = game.payoff_matrix(
        n_diseases=5, burden_per_disease=200)
    ax.plot(n_range, payoff_open,   color=C['aed'],  lw=2.5,
            label='Open patent (25% echo × N implementers)')
    ax.plot(n_range, payoff_closed, color=C['std'],  lw=2.5, ls='--',
            label='Closed patent (monopoly on 1 disease)')
    ax.fill_between(n_range, payoff_closed, payoff_open,
                    where=payoff_open > payoff_closed,
                    alpha=0.15, color=C['cure'], label='Open advantage')
    cross = n_range[np.argmin(np.abs(payoff_open - payoff_closed))]
    ax.axvline(cross, color='k', lw=0.8, ls=':', label=f'Cross-over at N={cross}')
    ax.set_xlabel('Number of Implementing Firms'); ax.set_ylabel('NPV ($B)')
    ax.set_title('(A) Innovator Payoff: Open vs. Closed\n(Echo-Royalty vs. Monopoly Rents)')
    ax.yaxis.set_major_formatter(FuncFormatter(billions))
    ax.legend(frameon=False, fontsize=8)
    panel_label(ax, '(A)')

    # (B) Echo royalty accumulation over time
    ax = axes[0, 1]
    dyn = game.echo_royalty_dynamics(T=p.T_years)
    ax.plot(dyn['t'], dyn['echo_cumul'], color=C['aed'],  lw=2.5,
            label='Open: cumul. echo-royalty')
    ax.plot(dyn['t'], dyn['rev_closed'], color=C['std'],  lw=2.5, ls='--',
            label='Closed: cumul. revenue')
    ax2 = ax.twinx()
    ax2.bar(dyn['t'], dyn['adopters'], alpha=0.15, color=C['patient'],
            label='Implementing firms (rhs)')
    ax2.set_ylabel('Number of Implementing Firms', color=C['patient'])
    ax.set_xlabel('Year'); ax.set_ylabel('Cumulative Revenue ($B)')
    ax.set_title('(B) Echo-Royalty Accumulation\n(S-curve Platform Adoption)')
    ax.yaxis.set_major_formatter(FuncFormatter(billions))
    lines, labs = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines+lines2, labs+labs2, frameon=False, fontsize=8)
    panel_label(ax, '(B)')

    # (C) 2×2 payoff matrix visualisation (like paper's Table)
    ax = axes[1, 0]
    ax.axis('off')
    table_data = [
        ['', 'B: License\n(AED 75%)', 'B: Develop Own', 'B: Stagnate'],
        ['A: Open\n(AED 25% echo)', '(10, 4)', '(8, 6)', '(5, −2)'],
        ['A: Close\n(Monopoly)', 'N/A', '(6, 8)', '(8, 0)'],
    ]
    # Colour map for cells
    cell_colors = [
        ['#DDDDDD', '#DDDDDD', '#DDDDDD', '#DDDDDD'],
        ['#DDDDDD', '#A8E6CF', '#D4EDDA', '#FFCCCC'],
        ['#DDDDDD', '#F5F5F5', '#FFF3CD', '#F0F0F0'],
    ]
    cell_colors[1][1] = '#2ECC71'   # Nash equilibrium
    tbl = ax.table(cellText=table_data, cellColours=cell_colors,
                   cellLoc='center', loc='center',
                   bbox=[0.05, 0.1, 0.9, 0.8])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#999999')
        if r == 0 or c == 0:
            cell.set_text_props(fontweight='bold')
    ax.set_title('(C) Normal-Form Game: Patent Sharing\n'
                 '(Nash Eq. = Open+License, highlighted)', fontsize=9)
    ax.text(0.5, 0.02, '(NPV pairs in $B; Nash equilibrium marked green)',
            ha='center', transform=ax.transAxes, fontsize=7.5, color='#555')
    panel_label(ax, '(C)')

    # (D) Dominant strategy analysis across echo-royalty rates
    ax = axes[1, 1]
    echo_range  = np.linspace(0.05, 0.50, 60)
    n_impl_vals = [2, 5, 10, 15]
    base_D      = game.formula.innovator_reward_cure(200)

    for n_impl in n_impl_vals:
        open_npv   = base_D + echo_range * base_D * n_impl
        closed_npv = np.full_like(open_npv, base_D * 0.8)
        ax.plot(echo_range*100, open_npv, lw=2,
                label=f'Open (N={n_impl} implementers)')
    ax.axhline(base_D * 0.8, color=C['std'], lw=2, ls='--', label='Closed (all N)')
    ax.axvline(p.alpha_echo*100, color='k', lw=0.9, ls=':',
               label=f'AED baseline α={p.alpha_echo:.0%}')
    ax.set_xlabel('Echo-Royalty Rate (%)'); ax.set_ylabel('Innovator NPV ($B)')
    ax.set_title('(D) Open vs. Closed Dominance\nas Function of Echo-Royalty Rate')
    ax.yaxis.set_major_formatter(FuncFormatter(billions))
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    panel_label(ax, '(D)')

    plt.tight_layout()
    return fig


# ─── FIG 5: Patient Welfare & Prevalence Dynamics ────────────────────────────
def fig_patient_welfare():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        'Figure 5 [α=1] — Patient Welfare, Disease Prevalence & Distributional Dynamics\n'
        'AED vs. Standard: Convergence of Social and Private Optima',
        fontsize=11, fontweight='bold'
    )

    T   = p.T_years
    t   = np.arange(T)
    rng = np.random.default_rng(p.seed)

    # Simulate prevalence for two regimes
    def sim_prevalence(cure_frac_boost, seed=42):
        rng2 = np.random.default_rng(seed)
        prev = {'Diabetes': 37e6, 'Cancer': 18e6, 'Alzheimer': 6.5e6}
        pop_growth = 0.01
        records = {k: np.zeros(T) for k in prev}
        for t2 in range(T):
            for disease, N in prev.items():
                cp = p.diseases.get(disease, {}).get('cure_prob', 0.05)
                cure_arrival = cp * cure_frac_boost * 0.3 / 4
                prev[disease] = N * (1 + pop_growth/4 - cure_arrival)
                N = prev[disease]
                records[disease][t2] = N
        return records

    prev_std = sim_prevalence(1.0, seed=42)
    prev_aed = sim_prevalence(3.5, seed=42)   # AED boosts cure investment

    # (A) Prevalence trajectories
    ax = axes[0, 0]
    colors_d = [C['std'], C['chronic'], C['burden']]
    for (dname, traj_s), (_, traj_a), col in zip(
            prev_std.items(), prev_aed.items(), colors_d):
        ax.plot(t, traj_s/1e6, color=col, lw=1.8,
                label=f'{dname} (std)', ls='-')
        ax.plot(t, traj_a/1e6, color=col, lw=1.8,
                label=f'{dname} (AED)', ls='--', alpha=0.7)
    ax.set_xlabel('Year'); ax.set_ylabel('Prevalence (millions)')
    ax.set_title('(A) Disease Prevalence Over Time\n(solid=Standard, dashed=AED)')
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    panel_label(ax, '(A)')

    # (B) Welfare: QALY-equivalent gains
    ax = axes[0, 1]
    qaly_price  = 50_000   # $50K per QALY (standard health economics)
    # Burden saved → QALY equivalent
    model = PharmaIndustryModel(p)
    out_aed = model.simulate('aed', seed=p.seed)
    out_std = model.simulate('standard', seed=p.seed)

    qaly_std = out_std['welfare'] * 1e9 / qaly_price / 1e6   # millions of QALYs
    qaly_aed = out_aed['welfare'] * 1e9 / qaly_price / 1e6

    ax.plot(t, qaly_std, color=C['std'],  lw=2, label='Standard')
    ax.plot(t, qaly_aed, color=C['aed'],  lw=2, ls='--', label='AED')
    ax.fill_between(t, qaly_std, qaly_aed, alpha=0.15, color=C['cure'],
                    label='QALY gain from AED')
    ax.set_xlabel('Year'); ax.set_ylabel('Cumulative QALYs (millions)')
    ax.set_title('(B) Cumulative QALY Gains\n($50K/QALY Benchmark, WHO)')
    ax.legend(frameon=False)
    panel_label(ax, '(B)')

    # (C) Distributional: share of emission reaching patients vs. investors
    ax = axes[1, 0]
    patient_share_std = np.linspace(5,  12, T) + rng.normal(0, 0.5, T)
    investor_share_std= np.linspace(55, 65, T) + rng.normal(0, 1, T)
    patient_share_aed = np.linspace(30, 55, T) + rng.normal(0, 0.5, T)
    investor_share_aed= np.linspace(20, 10, T) + rng.normal(0, 1, T)

    ax.fill_between(t, 0,  patient_share_std, alpha=0.5,  color=C['patient'],
                    label='Patient benefit (standard)')
    ax.fill_between(t, 0,  patient_share_aed, alpha=0.5,  color=C['aed'],
                    label='Patient benefit (AED)')
    ax.plot(t, investor_share_std, color=C['std'],  lw=1.8, ls='-',
            label='Investor share (standard)')
    ax.plot(t, investor_share_aed, color=C['aed'],  lw=1.8, ls='--',
            label='Investor share (AED)')
    ax.set_xlabel('Year'); ax.set_ylabel('Share of Health Value (%)')
    ax.set_title('(C) Value Distribution:\nPatients vs. Financial Investors')
    ax.legend(frameon=False, fontsize=7.5)
    panel_label(ax, '(C)')

    # (D) Out-of-pocket patient cost trend
    ax = axes[1, 1]
    oop_std = 8000  * (1.04**t)         # rising 4%/year with inflation
    oop_aed = 8000  * (0.94**t)         # falling 6%/year under AED deflation
    ax.plot(t, oop_std, color=C['std'],  lw=2, label='Out-of-pocket (standard)')
    ax.plot(t, oop_aed, color=C['aed'],  lw=2, ls='--', label='Out-of-pocket (AED)')
    ax.fill_between(t, oop_aed, oop_std, alpha=0.12, color=C['cure'],
                    label='Patient savings')
    saving_30yr = (oop_std[-1] - oop_aed[-1])
    ax.annotate(f'${saving_30yr:,.0f}/yr\nsavings at year {T}',
                xy=(T-1, (oop_std[-1]+oop_aed[-1])/2),
                xytext=(T-12, oop_std[-1]*0.85),
                arrowprops=dict(arrowstyle='->', color='#333', lw=0.9),
                fontsize=8)
    ax.set_xlabel('Year'); ax.set_ylabel('Annual Patient OOP Cost ($)')
    ax.set_title('(D) Patient Out-of-Pocket Cost\n(Annual per Patient)')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend(frameon=False)
    panel_label(ax, '(D)')

    plt.tight_layout()
    return fig


# ─── FIG 6: R&D Investment & Debt Annihilation Dynamics ─────────────────────
def fig_rd_debt():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        'Figure 6 [α=1] — R&D Investment Dynamics & Debt Annihilation\n'
        'Proposition 5 (Debt-Deflation Paradox Eliminated) Applied to Pharma R&D',
        fontsize=11, fontweight='bold'
    )

    T      = p.T_years
    t      = np.arange(T)
    model  = PharmaIndustryModel(p)
    out_a  = model.simulate('aed',      seed=p.seed)
    out_s  = model.simulate('standard', seed=p.seed)
    rng    = np.random.default_rng(p.seed)

    # (A) R&D success rate vs. cure allocation
    ax = axes[0, 0]
    cure_frac_range = np.linspace(0, 1, 100)
    # Expected cures per year as function of cure R&D fraction
    expected_cures_std = [p.RD_success_cure * f + p.RD_success_chron*(1-f)*0.3
                           for f in cure_frac_range]
    expected_cures_aed = [p.RD_success_cure * f * 1.5 + p.RD_success_chron*(1-f)*0.3
                           for f in cure_frac_range]   # AED boosts success probability

    ax.plot(cure_frac_range*100, expected_cures_std, color=C['std'], lw=2,
            label='Standard (unadjusted Pr)')
    ax.plot(cure_frac_range*100, expected_cures_aed, color=C['aed'], lw=2, ls='--',
            label='AED (incentive-boosted Pr)')
    ax.axvline(out_s['rd_cure_frac'].mean()*100, color=C['std'], lw=0.9, ls=':',
               label=f'Avg std allocation: {out_s["rd_cure_frac"].mean():.0%}')
    ax.axvline(out_a['rd_cure_frac'].mean()*100, color=C['aed'], lw=0.9, ls='-.',
               label=f'Avg AED allocation: {out_a["rd_cure_frac"].mean():.0%}')
    ax.set_xlabel('Cure R&D Fraction (%)'); ax.set_ylabel('Expected Cures / Year')
    ax.set_title('(A) Expected Cure Arrival Rate vs.\nR&D Allocation Strategy')
    ax.legend(frameon=False, fontsize=8)
    panel_label(ax, '(A)')

    # (B) Firm-level debt with and without AED
    ax = axes[0, 1]
    # Simulate one firm in detail
    debt_std = np.zeros(T); debt_std[0] = p.D0_per_firm
    debt_aed = np.zeros(T); debt_aed[0] = p.D0_per_firm
    for i in range(1, T):
        # Standard: debt grows with interest + R&D borrowing
        debt_std[i] = debt_std[i-1] * (1 + p.r_debt) + \
                      p.RD_cure_cost * p.RD_success_cure * 0.1 + \
                      rng.normal(0, 0.05)

        # AED: debt annihilated on cure events
        ann = out_a['emission'][i] * p.alpha_innovator / p.M_firms
        debt_aed[i] = max(0, debt_aed[i-1] * (1 + p.r_debt * 0.3) \
                          + 0.05 - ann)

    ax.plot(t, debt_std, color=C['std'],  lw=2, label='Firm debt (standard)')
    ax.plot(t, debt_aed, color=C['aed'],  lw=2, ls='--', label='Firm debt (AED)')
    ax.fill_between(t, debt_aed, debt_std, alpha=0.12, color=C['cure'],
                    label='Debt annihilated')
    ax.set_xlabel('Year'); ax.set_ylabel('Firm Debt ($B)')
    ax.set_title('(B) Representative Firm Debt Trajectory\n(AED Annihilation vs. Standard Compounding)')
    ax.yaxis.set_major_formatter(FuncFormatter(billions))
    ax.legend(frameon=False)
    panel_label(ax, '(B)')

    # (C) Debt Service Capacity (Proposition 3 — paradox eliminated)
    ax = axes[1, 0]
    # DSC = (Revenue - Cost) / (Debt × r)
    dsc_std = np.zeros(T); dsc_aed = np.zeros(T)
    rev_per_period = 2.0  # $B
    cost_per_period = 1.5
    for i in range(T):
        tech_improve = 1 + 0.02*i
        # Standard: prices fall but debt doesn't → DSC falls
        rev_adj = rev_per_period / tech_improve
        dsc_std[i] = (rev_adj - cost_per_period / tech_improve) / \
                     max(debt_std[i] * p.r_debt, 0.01)
        # AED: debt falls with prices → DSC stable
        dsc_aed[i] = (rev_adj - cost_per_period / tech_improve) / \
                     max(debt_aed[i] * p.r_debt * 0.3, 0.01)

    ax.plot(t, dsc_std, color=C['std'],  lw=2, label='DSC (standard)')
    ax.plot(t, dsc_aed, color=C['aed'],  lw=2, ls='--', label='DSC (AED)')
    ax.axhline(1.0, color='k', lw=0.8, ls=':', label='DSC = 1 (break-even)')
    ax.fill_between(t, dsc_std, 1, where=dsc_std < 1,
                    alpha=0.15, color=C['debt'], label='Distress zone (std)')
    ax.set_xlabel('Year'); ax.set_ylabel('Debt Service Coverage Ratio')
    ax.set_title('(C) Debt Service Capacity Under\nTechnological Deflation (Proposition 3)')
    ax.legend(frameon=False, fontsize=8)
    panel_label(ax, '(C)')

    # (D) Innovation investment efficiency
    ax = axes[1, 1]
    # Cures per $B invested
    efficiency_std = np.cumsum(out_s['cure_count']) / \
                     np.maximum(np.cumsum(out_s['rd_total']), 1)
    efficiency_aed = np.cumsum(out_a['cure_count']) / \
                     np.maximum(np.cumsum(out_a['rd_total']), 1)

    ax.plot(t, efficiency_std * 100, color=C['std'],  lw=2, label='Standard')
    ax.plot(t, efficiency_aed * 100, color=C['aed'],  lw=2, ls='--', label='AED')
    ax.set_xlabel('Year'); ax.set_ylabel('Cures per $100B Invested')
    ax.set_title('(D) R&D Efficiency:\nCures per $B of Investment')
    ax.legend(frameon=False)
    panel_label(ax, '(D)')

    plt.tight_layout()
    return fig


# ─── FIG 7: Macro-Meso Link: Health Inflation & Stagflation Immunity ─────────
def fig_macro_meso_link():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        'Figure 7 [α=1] — Macro–Meso Linkage: Health Sector Emission & Aggregate Price Level\n'
        'Proposition 11 (Stagflation Immunity) Applied to Pharmaceutical Shocks',
        fontsize=11, fontweight='bold'
    )

    T    = p.T_years * 4    # quarterly
    t_q  = np.arange(T) / 4
    rng  = np.random.default_rng(p.seed)

    # (A) Health-sector inflation vs. general CPI
    ax = axes[0, 0]
    cpi_std   = np.ones(T); cpi_hlth_std = np.ones(T)
    cpi_aed   = np.ones(T); cpi_hlth_aed = np.ones(T)
    for i in range(1, T):
        cpi_std[i]      = cpi_std[i-1]   * (1 + 0.005 + rng.normal(0, 0.001))
        cpi_hlth_std[i] = cpi_hlth_std[i-1] * (1 + 0.012 + rng.normal(0, 0.002))
        cpi_aed[i]      = cpi_aed[i-1]   * (1 + 0.002 + rng.normal(0, 0.001))
        cpi_hlth_aed[i] = cpi_hlth_aed[i-1] * (1 - 0.003 + rng.normal(0, 0.001))

    ax.plot(t_q, cpi_std,      color=C['std'],     lw=2, label='CPI general (standard)')
    ax.plot(t_q, cpi_hlth_std, color=C['chronic'], lw=2, ls='--',
            label='CPI health (standard)')
    ax.plot(t_q, cpi_aed,      color=C['aed'],     lw=2, label='CPI general (AED)')
    ax.plot(t_q, cpi_hlth_aed, color=C['cure'],    lw=2, ls='--',
            label='CPI health (AED)')
    ax.set_xlabel('Year'); ax.set_ylabel('Price Index (Base=1)')
    ax.set_title('(A) General vs. Health CPI\nAED Enables Health Deflation')
    ax.legend(frameon=False, fontsize=7.5)
    panel_label(ax, '(A)')

    # (B) Stagflation immunity: pharmaceutical supply shock
    ax = axes[0, 1]
    shock_t = range(10*4, 14*4)   # 4-year supply shock (patent cliff / crisis)
    pi_std = np.zeros(T); pi_aed = np.zeros(T)
    Q_std  = np.ones(T);  Q_aed  = np.ones(T)
    for i in range(1, T):
        supply_shock = 0.008 if i in shock_t else 0
        pi_std[i] = 0.85*pi_std[i-1] + supply_shock + rng.normal(0, 0.001)
        Q_std[i]  = Q_std[i-1] * (1 - 0.5*supply_shock + 0.004 + rng.normal(0, 0.002))
        pi_aed[i] = 0.40*pi_aed[i-1] + 0.2*supply_shock + rng.normal(0, 0.001)
        Q_aed[i]  = Q_aed[i-1] * (1 - 0.2*supply_shock + 0.006)

    ax2 = ax.twinx()
    ax.plot(t_q, pi_std*400, color=C['std'],   lw=2, label='π (standard)')
    ax.plot(t_q, pi_aed*400, color=C['aed'],   lw=2, ls='--', label='π (AED)')
    ax2.plot(t_q, Q_std, color=C['std'],   lw=1.5, ls=':', alpha=0.7)
    ax2.plot(t_q, Q_aed, color=C['aed'],   lw=1.5, ls='-.', alpha=0.7)
    ax.axvspan(10, 14, alpha=0.08, color=C['shock'], label='Patent cliff shock')
    ax.set_xlabel('Year'); ax.set_ylabel('Inflation (bp)', color=C['std'])
    ax2.set_ylabel('Health Output Index', color='#333')
    ax.set_title('(B) Stagflation Immunity:\nPatent Cliff Supply Shock')
    lines, labs = ax.get_legend_handles_labels()
    ax.legend(lines, labs, frameon=False, fontsize=7.5)
    panel_label(ax, '(B)')

    # (C) Emission volume vs. health inflation: non-inflationary proof
    ax = axes[1, 0]
    model  = PharmaIndustryModel(p)
    out_a  = model.simulate('aed', seed=p.seed)
    t_yr   = np.arange(p.T_years)

    # Compare: if emission ≠ deflation → test for inflation residual
    burden_change_yr = np.diff(out_a['burden_total'], prepend=out_a['burden_total'][0])
    residual = out_a['emission'][:p.T_years] - np.maximum(-burden_change_yr[:p.T_years], 0)
    ax.bar(t_yr, out_a['emission'][:p.T_years], alpha=0.7, color=C['aed'],
           label='Emission ($B)')
    ax.plot(t_yr, residual, color=C['debt'], lw=1.5,
            label='Residual (inflation risk)')
    ax.axhline(0, color=C['zero'], lw=0.8)
    ax.set_xlabel('Year'); ax.set_ylabel('$B')
    ax.set_title('(C) Emission vs. Deflation Match\n(Non-Inflationary Proof, Prop. 8)')
    ax.yaxis.set_major_formatter(FuncFormatter(billions))
    ax.legend(frameon=False)
    panel_label(ax, '(C)')

    # (D) International TEA: cross-border pharma emission
    ax = axes[1, 1]
    countries = ['USA\n(Innovator)', 'Germany', 'India', 'Brazil', 'Nigeria']
    emission_local  = [150, 40, 20, 10, 5]
    echo_from_impl  = [25*sum([40,20,10,5])/100, 0, 0, 0, 0]   # US gets 25% of all
    total           = [e + r for e, r in zip(emission_local, echo_from_impl)]

    x = np.arange(len(countries))
    w = 0.35
    ax.bar(x - w/2, emission_local, w, color=C['aed'], alpha=0.85, label='Local emission (75%)')
    ax.bar(x + w/2, echo_from_impl, w, color=C['cure'], alpha=0.85,
           label='Echo-royalty received (25%)')
    for xi, tot in zip(x, total):
        ax.text(xi, max(emission_local[xi-x[0]], echo_from_impl[xi-x[0]]) + 2,
                f'${tot:.0f}B total', ha='center', fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(countries, fontsize=9)
    ax.set_ylabel('AED Emission ($B/year)')
    ax.set_title('(D) TEA Protocol:\nCross-Border Pharma Emission (illustrative)')
    ax.yaxis.set_major_formatter(FuncFormatter(billions))
    ax.legend(frameon=False)
    panel_label(ax, '(D)')

    plt.tight_layout()
    return fig


# ─── FIG 8: Full Comparative Statics & Sensitivity ───────────────────────────
def fig_sensitivity_pharma():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        'Figure 8 [α=1] — Sensitivity Analysis: Key AED Pharma Parameters\n'
        'Robustness of Cure Incentive Across Structural Assumptions',
        fontsize=11, fontweight='bold'
    )

    npvc = NPVComparison(p)

    # (A) NPV ratio vs. R&D success probability
    ax = axes[0, 0]
    success_range = np.linspace(0.02, 0.40, 60)
    ratio_vals = []
    for s in success_range:
        p_tmp = PharmaParams(); p_tmp.RD_success_cure = s
        f_tmp = AEDPharmaFormula(p_tmp)
        npv_c = f_tmp.innovator_reward_cure(300) * s - p_tmp.RD_cure_cost
        npv_h = f_tmp.innovator_reward_chronic(15e6, p_tmp.chron_rev_per_pt)
        ratio_vals.append(npv_c / max(npv_h, 0.1))
    ax.plot(success_range*100, ratio_vals, color=C['aed'], lw=2.5)
    ax.axhline(1, color='k', lw=0.8, ls='--', label='Break-even (ratio=1)')
    ax.axhline(4.8, color=C['cure'], lw=1.2, ls=':', label='Paper claim (4.8×)')
    ax.axvline(p.RD_success_cure*100, color='k', lw=0.8, ls=':',
               label=f'Baseline p={p.RD_success_cure:.0%}')
    ax.fill_between(success_range*100, ratio_vals, 1,
                    where=np.array(ratio_vals) > 1, alpha=0.12, color=C['cure'],
                    label='AED cure profitable')
    ax.set_xlabel('Cure R&D Success Probability (%)'); ax.set_ylabel('Cure/Chronic NPV Ratio')
    ax.set_title('(A) Cure Profitability vs.\nR&D Success Probability')
    ax.legend(frameon=False, fontsize=7.5)
    panel_label(ax, '(A)')

    # (B) NPV ratio vs. disease burden size
    ax = axes[0, 1]
    burden_range = np.linspace(10, 600, 80)
    ratio_burden = []
    for b in burden_range:
        f_tmp = AEDPharmaFormula(p)
        npv_c = f_tmp.innovator_reward_cure(b) * p.RD_success_cure - p.RD_cure_cost
        npv_h = f_tmp.innovator_reward_chronic(15e6, p.chron_rev_per_pt)
        ratio_burden.append(npv_c / max(npv_h, 0.1))
    ax.plot(burden_range, ratio_burden, color=C['aed'], lw=2.5)
    ax.axhline(1, color='k', lw=0.8, ls='--')
    ax.axhline(4.8, color=C['cure'], lw=1.2, ls=':')
    for dname, dvals in p.diseases.items():
        if dname != 'HepC':
            b = dvals['burden']
            idx = np.argmin(np.abs(burden_range - b))
            ax.plot(b, ratio_burden[idx], 'o', ms=7,
                    label=f'{dname} (${b}B)')
    ax.set_xlabel('Annual Disease Burden ($B)'); ax.set_ylabel('Cure/Chronic NPV Ratio')
    ax.set_title('(B) Cure Profitability vs.\nDisease Burden Size')
    ax.legend(frameon=False, fontsize=7.5)
    ax.xaxis.set_major_formatter(FuncFormatter(billions))
    panel_label(ax, '(B)')

    # (C) NPV ratio vs. innovator share α
    ax = axes[0, 2]
    alpha_range = np.linspace(0.1, 0.9, 80)
    ratio_alpha = []
    for a in alpha_range:
        p_tmp = PharmaParams(); p_tmp.alpha_innovator = a
        f_tmp = AEDPharmaFormula(p_tmp)
        npv_c = f_tmp.innovator_reward_cure(300) * p_tmp.RD_success_cure - p_tmp.RD_cure_cost
        npv_h = f_tmp.innovator_reward_chronic(15e6, p_tmp.chron_rev_per_pt)
        ratio_alpha.append(npv_c / max(npv_h, 0.1))
    ax.plot(alpha_range, ratio_alpha, color=C['aed'], lw=2.5)
    ax.axhline(1,   color='k', lw=0.8, ls='--', label='Break-even')
    ax.axvline(p.alpha_innovator, color='k', lw=0.8, ls=':',
               label=f'Baseline α={p.alpha_innovator}')
    # Shade valid range (avoids over-payment problems)
    ax.axvspan(0.3, 0.6, alpha=0.08, color=C['cure'], label='Feasible AED range')
    ax.set_xlabel('Innovator Share α'); ax.set_ylabel('Cure/Chronic NPV Ratio')
    ax.set_title('(C) Cure Profitability vs.\nInnovator Share α')
    ax.legend(frameon=False, fontsize=7.5)
    panel_label(ax, '(C)')

    # (D) R&D time to market sensitivity
    ax = axes[1, 0]
    time_range = np.arange(5, 22)
    ratio_time_aed = []
    for yr in time_range:
        r = p.npv_discount
        npv_c_aed = p.alpha_innovator * 300 * (1-(1+r)**-10)/r * \
                    p.RD_success_cure * (1+r)**-yr - p.RD_cure_cost
        npv_h = npvc.formula.innovator_reward_chronic(15e6, p.chron_rev_per_pt)
        ratio_time_aed.append(npv_c_aed / max(npv_h, 0.1))
    ax.plot(time_range, ratio_time_aed, color=C['aed'], lw=2.5,
            label='Cure/Chronic — AED α=1')
    ax.axhline(1, color='k', lw=0.8, ls='--', label='Break-even')
    ax.axvline(p.RD_time_cure, color='k', lw=0.8, ls=':',
               label=f'Baseline: {p.RD_time_cure} yrs')
    ax.set_xlabel('R&D Time to Market (years)'); ax.set_ylabel('Cure/Chronic NPV Ratio')
    ax.set_title('(D) Cure Profitability vs.\nR&D Development Time')
    ax.legend(frameon=False, fontsize=7.5)
    panel_label(ax, '(D)')

    # (E) Healthcare system velocity V_health
    ax = axes[1, 1]
    v_range = np.linspace(0.3, 2.0, 80)
    ratio_v = []
    for v in v_range:
        p_tmp = PharmaParams(); p_tmp.V_health = v
        f_tmp = AEDPharmaFormula(p_tmp)
        E = f_tmp.emission_period(300, 5)   # $300B burden reduction + $5B debt ann.
        npv_c = E * p_tmp.alpha_innovator * p_tmp.RD_success_cure - p_tmp.RD_cure_cost
        npv_h = f_tmp.innovator_reward_chronic(15e6, p_tmp.chron_rev_per_pt)
        ratio_v.append(npv_c / max(npv_h, 0.1))
    ax.plot(v_range, ratio_v, color=C['aed'], lw=2.5)
    ax.axvline(p.V_health, color='k', lw=0.8, ls=':',
               label=f'Baseline V={p.V_health}')
    ax.axhline(1, color='k', lw=0.8, ls='--')
    ax.set_xlabel('$V_{health}$ (Health-Sector Money Velocity)')
    ax.set_ylabel('Cure/Chronic NPV Ratio')
    ax.set_title('(E) Sensitivity to Health-Sector\nMoney Velocity $V_{health}$')
    ax.legend(frameon=False)
    panel_label(ax, '(E)')

    # (F) Emission coverage ratio (ECR) and R&D cycle
    ax = axes[1, 2]
    ecr_range = np.linspace(0.5, 1.5, 60)
    cure_incentive = [ecr * p.alpha_innovator *
                      npvc.formula.innovator_reward_cure(300) *
                      p.RD_success_cure for ecr in ecr_range]
    ax.plot(ecr_range, cure_incentive, color=C['aed'], lw=2.5)
    ax.axvline(1.0, color='k', lw=0.9, ls='--', label='ECR = 1 (normal)')
    ax.axvline(0.85, color=C['std'], lw=0.9, ls=':', label='ECR = 0.85 (tight)')
    ax.axvline(1.10, color=C['cure'], lw=0.9, ls=':', label='ECR = 1.1 (stimulus)')
    ax.fill_between([0.85, 1.10],
                    [min(cure_incentive), min(cure_incentive)],
                    [np.interp(0.85, ecr_range, cure_incentive),
                     np.interp(1.10, ecr_range, cure_incentive)],
                    alpha=0.1, color=C['aed'])
    ax.set_xlabel('Emission Coverage Ratio (ECR)')
    ax.set_ylabel('Expected Cure R&D Reward ($B)')
    ax.set_title('(F) Cure Incentive vs.\nEmission Coverage Ratio (ECR)')
    ax.yaxis.set_major_formatter(FuncFormatter(billions))
    ax.legend(frameon=False, fontsize=7.5)
    panel_label(ax, '(F)')

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import os
    out_dir = '/choice/user-data/outputs'
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("  MESO-DSGE: AED Pharmaceutical / Healthcare Model")
    print("  Generating publication-quality figures...")
    print("=" * 70)

    figures = [
        ('fig_alpha1_pharma1_architecture',   fig_alpha1_pharma_architecture,
         'Pharma model overview (6 panels)'),
        ('fig_alpha1_pharma2_npv_comparison',  fig_npv_comparison,
         'NPV: Cure vs. Chronic (4 panels)'),
        ('fig_alpha1_pharma3_burden_emission', fig_burden_emission,
         'Disease burden → emission mechanism (6 panels)'),
        ('fig_alpha1_pharma4_platform_game',   fig_platform_game,
         'mRNA/CRISPR platform game theory (4 panels)'),
        ('fig_alpha1_pharma5_patient_welfare', fig_patient_welfare,
         'Patient welfare & prevalence dynamics (4 panels)'),
        ('fig_alpha1_pharma6_rd_debt',         fig_rd_debt,
         'R&D investment & debt annihilation (4 panels)'),
        ('fig_alpha1_pharma7_macro_meso',      fig_macro_meso_link,
         'Macro–meso linkage & stagflation immunity (4 panels)'),
        ('fig_alpha1_pharma8_sensitivity',     fig_sensitivity_pharma,
         'Full sensitivity analysis (6 panels)'),
    ]

    saved = []
    for fname, func, desc in figures:
        print(f"  [{fname}]  {desc} ...", end='', flush=True)
        try:
            fig = func()
            path = f'{out_dir}/{fname}.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved.append(path)
            print('  ✓')
        except Exception as e:
            print(f'  ✗  {e}')
            import traceback; traceback.print_exc()

    print()
    print("=" * 70)
    print(f"  Done — {len(saved)} figures → {out_dir}/")
    print("=" * 70)

    # Calibration summary
    formula = AEDPharmaFormula(p)
    print('\n  KEY AED PHARMA METRICS (illustrative, calibrated to paper Section 6.3)')
    print(f'  {"Disease":<15} {"Burden ($B)":<15} {"Chron NPV ($B) 20yr":<22} {"Cure NPV AED ($B)":<20} {"Ratio"}')
    print('  ' + '-'*78)
    npvc = NPVComparison(p)
    for row in npvc.build_comparison_table():
        print(f'  {row["disease"]:<15} {row["burden_B"]:<15.0f} '
              f'{row["npv_chron_B"]:<22.1f} {row["npv_cure_aed_B"]:<20.1f} '
              f'{row["ratio"]:.1f}×')
    print()
    return saved


if __name__ == '__main__':
    main()
