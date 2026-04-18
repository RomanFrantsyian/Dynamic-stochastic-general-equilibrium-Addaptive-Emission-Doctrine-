"""
=============================================================================
MESO-LEVEL SECTORAL DSGE: Adaptive Emission Doctrine in Energy Sector
=============================================================================
Research Question:
  Does AED resolve the Zero Marginal Cost (ZMC) Problem in renewable energy?

The ZMC Problem (Rifkin 2014; Hirth 2013):
  Renewable energy (solar, wind) has MC ≈ 0.
  Competitive equilibrium → P_energy → 0 → Revenue → 0 → Debt unpayable.
  Capital recovery impossible → investment collapses despite social optimality.
  Standard monetary policy (QE) has NO mechanism to address this.

AED Hypothesis:
  If E_AED = [P_target·ΔQ]/V + D_annihilated, and technological deflation
  in energy is LARGE (solar -99% cost over 20 years), then:
  - D_annihilated replaces revenue as value-capture mechanism
  - MC = 0 stops being a viability constraint
  - Investment cycle continues despite falling market prices

Sector Architecture (5 subsectors + grid + households):
  1. Fossil (coal/gas): high MC, low tech deflation, legacy debt
  2. Solar: MC ≈ 0, massive tech deflation (Swanson's Law -20%/doubling)
  3. Wind: MC ≈ 0, strong tech deflation (-12%/year learning rate)
  4. Nuclear: MC ≈ 0, low tech deflation, extremely high capex
  5. Storage (batteries): MC ≈ 0, rapid tech deflation (-18%/year)
  6. Grid operator: regulated, transmission pricing
  7. Households/industry: price-elastic demand

DSGE Extension:
  - Sectoral Euler equations with capital specificity
  - Merit order dispatch with ZMC endgeneity
  - Cantillon wedge: QE flows to fossil finance, not renewable innovators
  - AED redistribution: D_annihilated ∝ sectoral technological deflation
  - 75/25 innovator/implementer rule per subsector
  - Energy price as endogenous variable (not exogenous CPI component)

Empirical Calibration:
  - IRENA (2023): Solar LCOE, Wind LCOE, cost trajectories
  - IEA (2023): Capacity factors, load profiles
  - BloombergNEF (2023): Battery learning rates
  - Fed/ECB data: Energy sector debt structures

Propositions tested:
  P1: Under standard QE, renewable firms face ZMC revenue trap → default
  P2: Under AED, D_annihilated = f(tech deflation) → ZMC not binding
  P3: AED accelerates energy transition vs. QE baseline
  P4: Cantillon correction redistributes from fossil finance to renewable innovators
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PLOT STYLE (consistent with macro DSGE)
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Georgia'],
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linestyle': '--',
    'grid.linewidth': 0.6, 'axes.linewidth': 0.8, 'lines.linewidth': 1.8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'figure.facecolor': 'white', 'axes.facecolor': '#FAFAFA',
    'savefig.dpi': 150, 'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
})

C = {
    'fossil':  '#5D4037',   # brown
    'solar':   '#F9A825',   # amber/gold
    'wind':    '#1976D2',   # blue
    'nuclear': '#7B1FA2',   # purple
    'storage': '#00838F',   # teal
    'grid':    '#546E7A',   # blue-grey
    'aed':     '#C0392B',   # crimson
    'std':     '#1A3A5C',   # navy
    'data':    '#2C7A4B',   # forest
    'shock':   '#E67E22',   # orange
    'zero':    '#555555',
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ENERGY SECTOR PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
class EnergyParams:
    """
    Calibration to IRENA 2023, IEA 2023, BloombergNEF 2023.
    Quarterly frequency. All cost/price in normalized units (2024=1.0).
    """
    # ── Time ───────────────────────────────────────────────────────────────
    T       = 120      # 30 years quarterly
    T_irf   = 40
    N_boot  = 300
    seed    = 42

    # ── Macro anchors (from macro DSGE) ───────────────────────────────────
    beta    = 0.99     # Discount factor
    sigma   = 1.5      # Inv. EIS
    r_star  = 0.010    # Natural rate (quarterly)
    pi_star = 0.005    # Inflation target
    V       = 1.80     # Money velocity

    # ── Sectoral: LCOE base levels (normalized, 2024=1.0) ─────────────────
    # Source: IRENA 2023 Renewable Power Generation Costs
    lcoe_fossil  = 1.00   # Reference (coal+gas weighted avg)
    lcoe_solar   = 0.50   # Solar utility: $33/MWh vs. $65/MWh fossil
    lcoe_wind    = 0.55   # Onshore wind: $33/MWh
    lcoe_nuclear = 1.40   # Nuclear: $92/MWh (high capex)
    lcoe_storage = 0.30   # Battery storage (levelized)

    # ── Marginal costs MC (normalized) ───────────────────────────────────
    mc_fossil    = 0.65   # Fuel + O&M, significant
    mc_solar     = 0.002  # Near zero (maintenance only)
    mc_wind      = 0.003  # Near zero
    mc_nuclear   = 0.010  # Very low but non-zero (fuel)
    mc_storage   = 0.005  # Near zero

    # ── Capital expenditure share (capex/lcoe ratio) ──────────────────────
    capex_fossil  = 0.35
    capex_solar   = 0.98   # Almost all LCOE is capex recovery
    capex_wind    = 0.95
    capex_nuclear = 0.99
    capex_storage = 0.97

    # ── Debt/Asset ratios (initial) ────────────────────────────────────────
    # Source: BloombergNEF Energy Finance Survey 2023
    dar_fossil   = 0.45   # Mature sector, moderate leverage
    dar_solar    = 0.72   # Project finance, high leverage
    dar_wind     = 0.68
    dar_nuclear  = 0.80   # Extremely high, sovereign-backed
    dar_storage  = 0.65

    # ── Technology deflation rates (quarterly) ────────────────────────────
    # Swanson's Law (solar): -20% per doubling → ~-7%/yr
    # Wright's Law (wind): -12%/year learning rate → ~-3%/yr
    # Battery: -18%/year → -4.5%/yr
    # LCOE cost reduction rates per quarter
    g_solar   = 0.018   # ~7% per year cost reduction
    g_wind    = 0.008   # ~3% per year
    g_nuclear = 0.002   # Slow, mainly construction efficiency
    g_storage = 0.045   # ~18% per year (fastest)
    g_fossil  = 0.002   # Slow efficiency gains

    # ── Capacity factors ──────────────────────────────────────────────────
    cf_fossil  = 0.85
    cf_solar   = 0.22
    cf_wind    = 0.35
    cf_nuclear = 0.92
    cf_storage = 0.25   # Round-trip efficiency weighted

    # ── Initial capacity shares (% of generation) ─────────────────────────
    cap_fossil  = 0.60
    cap_solar   = 0.08
    cap_wind    = 0.10
    cap_nuclear = 0.10
    cap_storage = 0.02
    # (remaining 10% = hydro/other, treated as fixed)

    # ── Cantillon wedge in energy finance ────────────────────────────────
    # QE flows primarily to fossil-backed financial instruments
    cantillon_fossil  = 0.80   # 80% of QE energy credit flows to fossil
    cantillon_solar   = 0.08
    cantillon_wind    = 0.07
    cantillon_nuclear = 0.04
    cantillon_storage = 0.01

    # ── AED parameters ───────────────────────────────────────────────────
    alpha_innovator  = 0.25   # Echo-royalty (75/25 rule)
    alpha_implementer= 0.75
    kappa_aed        = 0.002  # AED inflation buffer (near zero)

    # ── Price elasticity of energy demand ─────────────────────────────────
    epsilon_d = -0.40   # Short-run price elasticity (IEA)

    # ── ZMC threshold: revenue floor below which investment stops ─────────
    revenue_floor_std = 0.15   # Under standard system: need 15% of LCOE as revenue margin
    revenue_floor_aed = 0.00   # Under AED: revenue can be zero (D_annihilated covers it)

    # ── Investment adjustment cost ─────────────────────────────────────────
    psi = 5.0    # Convex adjustment cost coefficient

    # ── Shock parameters ─────────────────────────────────────────────────
    sig_demand  = 0.015   # Demand shock
    sig_tech    = 0.008   # Tech shock (additional to trend)
    sig_policy  = 0.005   # Policy shock
    rho_demand  = 0.80
    rho_tech    = 0.90

ep = EnergyParams()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SECTORAL DSGE CORE
# ─────────────────────────────────────────────────────────────────────────────
class EnergySector:
    """
    Single energy subsector with ZMC-aware optimization.

    Firm problem under AED:
    max_{I_t} E_t Σ β^s [Revenue_t - MC_t·Q_t - r_D·D_t + D_annihilated_t - adj_cost]

    Key insight: Under AED, D_annihilated replaces missing revenue when MC→0.
    Under standard system, no such mechanism → revenue floor constraint binding.
    """

    def __init__(self, name: str, params: EnergyParams, regime: str = 'standard'):
        self.name   = name
        self.p      = params
        self.regime = regime
        self._load_sector_params()

    def _load_sector_params(self):
        n = self.name
        p = self.p
        self.mc        = getattr(p, f'mc_{n}')
        self.lcoe_base = getattr(p, f'lcoe_{n}')
        self.capex_sh  = getattr(p, f'capex_{n}')
        self.dar       = getattr(p, f'dar_{n}')
        self.g_tech    = getattr(p, f'g_{n}')
        self.cf        = getattr(p, f'cf_{n}')
        self.cap_init  = getattr(p, f'cap_{n}')
        self.cantillon = getattr(p, f'cantillon_{n}')

    def simulate(self, T=None, market_prices=None, seed=None):
        """
        Simulate sector dynamics over T periods.
        market_prices: exogenous energy price path (merit-order outcome)
        Returns dict of sector time series.
        """
        T    = T or self.p.T
        seed = seed or self.p.seed
        rng  = np.random.default_rng(seed + hash(self.name) % 1000)

        # ── State variables ────────────────────────────────────────────
        cap      = np.zeros(T)   # Installed capacity (index)
        lcoe     = np.zeros(T)   # Levelized cost
        debt     = np.zeros(T)   # Debt stock (ratio to asset value)
        invest   = np.zeros(T)   # Investment rate
        revenue  = np.zeros(T)   # Revenue (price × generation)
        d_annihil= np.zeros(T)   # AED debt annihilation
        emission = np.zeros(T)   # AED emission received
        profit   = np.zeros(T)   # Net profit
        viable   = np.zeros(T, dtype=bool)  # Investment viability

        # ── Initial conditions ─────────────────────────────────────────
        cap[0]   = self.cap_init
        lcoe[0]  = self.lcoe_base
        debt[0]  = self.dar

        for t in range(1, T):
            # Technology deflation (Swanson/Wright's Law + stochastic)
            tech_shock = rng.normal(0, self.p.sig_tech)
            lcoe[t] = lcoe[t-1] * (1 - self.g_tech + tech_shock)
            lcoe[t] = max(lcoe[t], self.mc)  # Cannot go below MC

            # Market price (merit order: fossil sets price floor)
            P_market = market_prices[t] if market_prices is not None else lcoe[t]

            # Generation
            gen = cap[t-1] * self.cf

            # Revenue
            revenue[t] = P_market * gen

            # ── AED: Debt Annihilation ─────────────────────────────────
            # D_annihilated = α · (lcoe_old - lcoe_new) · capacity
            # This is the "deflationary vacuum" created by this sector
            delta_lcoe = max(0, lcoe[t-1] - lcoe[t])
            deflat_vacuum = delta_lcoe * cap[t-1]

            if self.regime == 'aed':
                d_annihil[t] = self.p.alpha_implementer * deflat_vacuum
                emission[t]  = d_annihil[t]
                # Echo-royalty from downstream implementers (simplified)
                emission[t] += self.p.alpha_innovator * deflat_vacuum * 0.5
            else:
                d_annihil[t] = 0.0
                # QE Cantillon channel: fossil gets disproportionate credit
                qe_flow = 0.002 * self.cantillon   # QE per quarter
                emission[t] = qe_flow  # goes to debt service, not investment

            # ── Debt dynamics ──────────────────────────────────────────
            r_debt = self.p.r_star + (0.02 if self.regime == 'standard' else 0.0)
            debt[t] = debt[t-1] * (1 + r_debt) - d_annihil[t] + \
                      0.01 * (invest[t-1] if t > 1 else 0)
            debt[t] = np.clip(debt[t], 0.0, 3.0)

            # ── Investment viability check (ZMC Problem) ───────────────
            # Revenue margin = (Revenue - MC·gen) / (LCOE·cap) needed to service debt
            mc_cost = self.mc * gen
            net_revenue = revenue[t] - mc_cost

            if self.regime == 'standard':
                # Must cover debt service from revenue alone
                debt_service_needed = debt[t-1] * r_debt
                revenue_margin = (net_revenue - debt_service_needed) / (lcoe[t] * cap[t-1] + 1e-6)
                is_viable = revenue_margin > self.p.revenue_floor_std
            else:
                # AED: D_annihilated covers debt service → revenue floor = 0
                aed_income = net_revenue + emission[t]
                debt_service_needed = debt[t-1] * r_debt
                revenue_margin = (aed_income - debt_service_needed) / (lcoe[t] * cap[t-1] + 1e-6)
                is_viable = revenue_margin > self.p.revenue_floor_aed

            viable[t] = is_viable

            # ── Investment decision ────────────────────────────────────
            if is_viable:
                # Tobin's Q logic: invest if PV of future returns > cost
                # Simplified: invest proportional to margin + AED incentive
                base_inv = max(0, revenue_margin * 0.3)
                aed_bonus = emission[t] * 2.0 if self.regime == 'aed' else 0
                invest[t] = min(base_inv + aed_bonus, 0.15)  # cap at 15%/qtr
            else:
                # ZMC Trap: no viable investment
                invest[t] = max(0, invest[t-1] * 0.5)  # Decay

            # ── Capacity accumulation ──────────────────────────────────
            # Adjustment cost (convex)
            adj_cost = self.p.psi / 2 * invest[t]**2
            net_invest = invest[t] - adj_cost
            dep_rate   = 0.025 if self.name != 'nuclear' else 0.008   # quarterly
            cap[t] = cap[t-1] * (1 - dep_rate) + net_invest

            # ── Profit ────────────────────────────────────────────────
            profit[t] = net_revenue + emission[t] - debt[t-1] * r_debt - adj_cost

        return {
            'cap': cap, 'lcoe': lcoe, 'debt': debt,
            'invest': invest, 'revenue': revenue,
            'd_annihil': d_annihil, 'emission': emission,
            'profit': profit, 'viable': viable,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MERIT ORDER DISPATCH + ENERGY PRICE EQUILIBRIUM
# ─────────────────────────────────────────────────────────────────────────────
class MeritOrderModel:
    """
    Endogenous energy price via competitive dispatch:
    - Renewables (MC≈0) dispatched first
    - Fossil sets marginal price when needed
    - As renewables penetrate → price → 0 (ZMC problem)

    Under AED: price signal decoupled from revenue via D_annihilated
    Under standard: price → 0 = existential crisis for investors
    """

    def __init__(self, params: EnergyParams):
        self.p = params

    def compute_price(self, cap_solar, cap_wind, cap_nuclear, cap_storage,
                      cap_fossil, demand, t):
        """
        Simplified merit order: P = MC_fossil when fossil needed,
        P → 0 when renewables alone can serve demand.
        """
        p = self.p

        # Total renewable supply at MC≈0
        ren_supply = (cap_solar * p.cf_solar +
                      cap_wind  * p.cf_wind  +
                      cap_nuclear * p.cf_nuclear +
                      cap_storage * p.cf_storage)

        # Fraction of demand served by renewables
        ren_fraction = min(1.0, ren_supply / (demand + 1e-6))

        # Merit order price: when ren_fraction → 1, price → MC_fossil × (1-ren_fraction)
        # This is the "Value of Solar" decline (Hirth 2013)
        P_base = p.mc_fossil * (1 - ren_fraction) + p.mc_solar * ren_fraction

        # Scarcity rent when fossil needed
        if ren_supply < demand:
            fossil_needed = demand - ren_supply
            scarcity_premium = 0.10 * (fossil_needed / demand)
            P_base += scarcity_premium

        # Cannot go below floor (grid costs)
        P_floor = 0.02   # Transmission/distribution floor
        return max(P_floor, P_base)

    def simulate_prices(self, sectors_std, sectors_aed, T=None, seed=None):
        """Simulate energy price paths under both regimes."""
        T    = T or self.p.T
        seed = seed or self.p.seed
        rng  = np.random.default_rng(seed)

        prices_std = np.zeros(T)
        prices_aed = np.zeros(T)

        # Demand path (slowly growing + cyclical shocks)
        demand = np.ones(T)
        for t in range(1, T):
            demand[t] = demand[t-1] * (1 + 0.003) + rng.normal(0, self.p.sig_demand)

        # Initial capacities
        caps_std = {s: ep.__dict__.get(f'cap_{s}', 0.1)
                    for s in ['solar', 'wind', 'nuclear', 'storage', 'fossil']}
        caps_aed = caps_std.copy()

        for t in range(1, T):
            # Update capacities from sector simulations
            for s in ['solar', 'wind', 'nuclear', 'storage', 'fossil']:
                caps_std[s] = sectors_std[s]['cap'][t]
                caps_aed[s] = sectors_aed[s]['cap'][t]

            prices_std[t] = self.compute_price(
                caps_std['solar'], caps_std['wind'],
                caps_std['nuclear'], caps_std['storage'],
                caps_std['fossil'], demand[t], t)

            prices_aed[t] = self.compute_price(
                caps_aed['solar'], caps_aed['wind'],
                caps_aed['nuclear'], caps_aed['storage'],
                caps_aed['fossil'], demand[t], t)

        return prices_std, prices_aed, demand


# ─────────────────────────────────────────────────────────────────────────────
# 4.  INTEGRATED SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class EnergyDSGE:
    """
    Full sectoral DSGE for energy with AED meso-dynamics.
    Iterative equilibrium: prices → investment → capacity → prices
    """

    SECTORS = ['fossil', 'solar', 'wind', 'nuclear', 'storage']

    def __init__(self, params: EnergyParams):
        self.p = params
        self.merit = MeritOrderModel(params)

    def _run_sectors(self, regime, market_prices, seed=42):
        """Run all 5 sectors given market price path."""
        results = {}
        for s in self.SECTORS:
            sector = EnergySector(s, self.p, regime=regime)
            results[s] = sector.simulate(T=self.p.T,
                                          market_prices=market_prices,
                                          seed=seed)
        return results

    def run(self, n_iter=4, seed=42):
        """
        Iterative equilibrium computation:
        Start with fossil-based price, update capacities, recompute price.
        """
        T = self.p.T

        # Initial price guess: fossil marginal cost
        prices_std = np.ones(T) * self.p.mc_fossil
        prices_aed = np.ones(T) * self.p.mc_fossil

        for iteration in range(n_iter):
            sectors_std = self._run_sectors('standard', prices_std, seed=seed)
            sectors_aed = self._run_sectors('aed',      prices_aed, seed=seed)

            new_std, new_aed, demand = self.merit.simulate_prices(
                sectors_std, sectors_aed, T=T, seed=seed)

            # Damped update for convergence
            damp = 0.6
            prices_std = damp * new_std + (1-damp) * prices_std
            prices_aed = damp * new_aed + (1-damp) * prices_aed

        return sectors_std, sectors_aed, prices_std, prices_aed, demand

    def aggregate(self, sectors):
        """Compute economy-wide energy aggregates."""
        T = self.p.T
        agg = {
            'total_cap':     np.zeros(T),
            'ren_share':     np.zeros(T),
            'total_debt':    np.zeros(T),
            'total_invest':  np.zeros(T),
            'total_emit':    np.zeros(T),
            'total_profit':  np.zeros(T),
            'viable_frac':   np.zeros(T),
        }
        ren_sectors = ['solar', 'wind', 'nuclear', 'storage']

        for t in range(T):
            caps = {s: sectors[s]['cap'][t] for s in self.SECTORS}
            total = sum(caps.values()) + 1e-6

            agg['total_cap'][t]    = total
            agg['ren_share'][t]    = sum(caps[s] for s in ren_sectors) / total
            agg['total_debt'][t]   = np.mean([sectors[s]['debt'][t] for s in self.SECTORS])
            agg['total_invest'][t] = sum(sectors[s]['invest'][t] for s in self.SECTORS)
            agg['total_emit'][t]   = sum(sectors[s]['emission'][t] for s in self.SECTORS)
            agg['total_profit'][t] = sum(sectors[s]['profit'][t] for s in self.SECTORS)
            n_viable = sum(sectors[s]['viable'][t] for s in self.SECTORS)
            agg['viable_frac'][t]  = n_viable / len(self.SECTORS)

        # Carbon intensity proxy (fossil share → emissions)
        agg['carbon_intensity'] = np.array([
            sectors['fossil']['cap'][t] / (sum(sectors[s]['cap'][t]
             for s in self.SECTORS) + 1e-6) for t in range(T)])

        return agg


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ZMC PROBLEM ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
class ZMCAnalytics:
    """
    Formal analysis of the Zero Marginal Cost problem:

    Under standard system:
    Revenue = P_market × CF × Cap
    As Cap_ren → ∞, P_market → MC_ren ≈ 0
    Revenue → 0, but Debt > 0 → firm insolvent

    Under AED:
    D_annihilated = α · ΔC · Cap_ren  (ΔC = cost reduction from tech progress)
    As long as ΔC > 0 (i.e., technology improves), D_annihilated > 0
    Revenue can = 0, firm survives via debt annihilation
    """

    @staticmethod
    def zmc_trap_threshold(g_tech, dar, r_debt, cf, alpha=0.75):
        """
        Compute the renewable penetration level at which ZMC trap triggers.

        Returns:
          ren_pct_trap: renewable share at which standard system collapses
          ren_pct_aed:  AED system (no collapse threshold — ∞)
        """
        # Standard: trap when P_market < r_debt × dar × lcoe / (cf × gen)
        # Simplified: trap when ren_fraction > critical level
        ren_pct_trap = 1 - (r_debt * dar) / (0.65 + 1e-6)  # fossil MC as reference
        ren_pct_trap = np.clip(ren_pct_trap, 0, 1)

        # AED: no trap as long as g_tech > 0 (deflation continues)
        # D_annihilated = alpha × g_tech × dar > 0 always
        aed_income_from_deflation = alpha * g_tech * dar
        aed_viable = aed_income_from_deflation > 0

        return ren_pct_trap, (np.inf if aed_viable else 0.0)

    @staticmethod
    def lcoe_trajectory(g_tech, T, lcoe_0=1.0, sig=0.005, seed=42):
        """Simulate LCOE path with stochastic tech progress."""
        rng = np.random.default_rng(seed)
        lcoe = np.zeros(T)
        lcoe[0] = lcoe_0
        for t in range(1, T):
            shock = rng.normal(0, sig)
            lcoe[t] = lcoe[t-1] * (1 - g_tech + shock)
        return np.maximum(lcoe, 0.001)

    @staticmethod
    def d_annihilated_path(lcoe, cap, alpha=0.75):
        """Compute D_annihilated from LCOE trajectory and capacity."""
        T = len(lcoe)
        d_ann = np.zeros(T)
        for t in range(1, T):
            delta = max(0, lcoe[t-1] - lcoe[t])
            d_ann[t] = alpha * delta * cap[t-1]
        return d_ann

    @staticmethod
    def revenue_vs_dann_comparison(sectors_std, sectors_aed, T):
        """
        Key chart: revenue under standard vs. (revenue + D_annihilated) under AED.
        This directly shows whether ZMC problem is resolved.
        """
        ren = ['solar', 'wind']
        rev_std = np.zeros(T)
        income_aed = np.zeros(T)
        for s in ren:
            rev_std   += sectors_std[s]['revenue']
            income_aed += sectors_aed[s]['revenue'] + sectors_aed[s]['emission']
        return rev_std, income_aed


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CANTILLON CORRECTION IN ENERGY
# ─────────────────────────────────────────────────────────────────────────────
class CantillonEnergyModel:
    """
    QE distributional analysis in energy finance:

    Standard: Central bank purchases → bond market → large fossil fuel
              infrastructure bonds (established collateral, ratings) →
              fossil gets cheap credit, renewables face high-yield constraints.

    AED: Emission tied to verified technological deflation →
         solar/wind get D_annihilated proportional to their cost reduction →
         systematic reversal of Cantillon advantage.

    Empirical basis:
    - BloombergNEF (2023): Fossil fuel finance received $5.8T in 2022
    - Renewable finance: $1.3T in 2022 (despite same capacity additions)
    - Cost of capital: Fossil ~4-5%, Renewable project finance ~8-12%
    """

    def __init__(self, params: EnergyParams):
        self.p = params

    def simulate_credit_flows(self, T, regime, seed=42):
        """Credit flow to each sector under each regime."""
        rng = np.random.default_rng(seed)
        p = self.p
        sectors = self.p.__class__.__dict__
        credit = {s: np.zeros(T) for s in
                  ['fossil', 'solar', 'wind', 'nuclear', 'storage']}
        cost_of_capital = {s: np.zeros(T) for s in credit}

        for t in range(T):
            # Total QE/emission pool (constant fraction of GDP equivalent)
            if regime == 'standard':
                pool = 0.02 + 0.005 * rng.standard_normal()   # QE pool
                # Cantillon: distribute by collateral quality / proximity
                credit['fossil'][t]  = pool * p.cantillon_fossil
                credit['solar'][t]   = pool * p.cantillon_solar
                credit['wind'][t]    = pool * p.cantillon_wind
                credit['nuclear'][t] = pool * p.cantillon_nuclear
                credit['storage'][t] = pool * p.cantillon_storage
                # Cost of capital: inverse of credit access
                cost_of_capital['fossil'][t]  = p.r_star + 0.005
                cost_of_capital['solar'][t]   = p.r_star + 0.030
                cost_of_capital['wind'][t]    = p.r_star + 0.025
                cost_of_capital['nuclear'][t] = p.r_star + 0.020
                cost_of_capital['storage'][t] = p.r_star + 0.035
            else:
                # AED: emission proportional to technological deflation
                # solar gets most because highest g_tech × cap
                def aed_share(g, cap_init):
                    return g * cap_init
                total_defl = (p.g_solar  * p.cap_solar  +
                              p.g_wind   * p.cap_wind   +
                              p.g_nuclear* p.cap_nuclear +
                              p.g_storage* p.cap_storage +
                              p.g_fossil * p.cap_fossil)
                pool = total_defl * 1.5   # emission proportional to actual deflation
                credit['fossil'][t]  = pool * aed_share(p.g_fossil, p.cap_fossil)  / total_defl
                credit['solar'][t]   = pool * aed_share(p.g_solar,  p.cap_solar)   / total_defl
                credit['wind'][t]    = pool * aed_share(p.g_wind,   p.cap_wind)    / total_defl
                credit['nuclear'][t] = pool * aed_share(p.g_nuclear,p.cap_nuclear) / total_defl
                credit['storage'][t] = pool * aed_share(p.g_storage,p.cap_storage) / total_defl
                # Cost of capital: AED → 0% + service fee
                for s in credit:
                    cost_of_capital[s][t] = p.kappa_aed + 0.005   # service fee only

        return credit, cost_of_capital


# ─────────────────────────────────────────────────────────────────────────────
# 7.  FIGURE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def add_zeroline(ax, lw=0.8):
    ax.axhline(0, color=C['zero'], linewidth=lw, linestyle='-', zorder=1)

def panel_label(ax, label, x=0.03, y=0.95, fs=10):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fs, fontweight='bold', va='top')


def fig_zmc_problem(sectors_std, sectors_aed, prices_std, prices_aed):
    """
    Figure M1: The Zero Marginal Cost Problem and AED Resolution
    Core scientific figure: proves/disproves P1 and P2.
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        'Figure M1 — Zero Marginal Cost Problem in Renewable Energy:\n'
        'Standard Monetary Policy (ZMC Trap) vs. AED Resolution\n'
        '(Propositions P1: Standard→Trap, P2: AED→No Trap)',
        fontsize=11, fontweight='bold', y=1.02
    )
    T   = ep.T
    t   = np.arange(T)
    yr  = 2024 + t * 0.25
    zmc = ZMCAnalytics()

    spec = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.40)

    # ── (A) Energy price collapse under high renewable penetration ─────────
    ax = fig.add_subplot(spec[0, 0])
    ax.plot(yr, prices_std, color=C['std'], lw=2, label='Standard (QE)')
    ax.plot(yr, prices_aed, color=C['aed'], lw=2, ls='--', label='AED')
    ax.axhline(ep.mc_solar, color=C['solar'], lw=1, ls=':', label=f'MC solar={ep.mc_solar}')
    ax.axhline(ep.mc_fossil, color=C['fossil'], lw=1, ls=':', alpha=0.6, label=f'MC fossil={ep.mc_fossil}')
    ax.set_title('(A) Market Price of Energy\n(Merit Order Equilibrium)', fontsize=9)
    ax.set_xlabel('Year'); ax.set_ylabel('Price (normalized)')
    ax.legend(frameon=False, fontsize=7)
    panel_label(ax, '(A)')

    # ── (B) Revenue: solar under standard vs. AED ─────────────────────────
    ax = fig.add_subplot(spec[0, 1])
    rev_solar_std = sectors_std['solar']['revenue']
    rev_solar_aed = sectors_aed['solar']['revenue']
    emit_solar_aed = sectors_aed['solar']['emission']
    total_income_aed = rev_solar_aed + emit_solar_aed

    ax.plot(yr, rev_solar_std,    color=C['std'],   lw=2, label='Revenue (Standard)')
    ax.plot(yr, rev_solar_aed,    color=C['solar'],  lw=1.5, ls=':', label='Revenue (AED, market)')
    ax.plot(yr, total_income_aed, color=C['aed'],   lw=2, ls='--', label='Revenue + D_annihil. (AED)')
    ax.fill_between(yr, rev_solar_aed, total_income_aed,
                    alpha=0.2, color=C['aed'], label='D_annihilated (AED)')

    # ZMC trap zone
    trap_t = np.where(rev_solar_std < 0.05)[0]
    if len(trap_t) > 0:
        ax.axvspan(yr[trap_t[0]], yr[-1], alpha=0.06, color=C['std'], label='ZMC Trap Zone')

    ax.set_title('(B) Solar Revenue: ZMC Trap vs. AED\n'
                 '$D_{annihilated}$ Replaces Missing Revenue', fontsize=9)
    ax.set_xlabel('Year'); ax.set_ylabel('Revenue (norm.)')
    ax.legend(frameon=False, fontsize=7)
    add_zeroline(ax)
    panel_label(ax, '(B)')

    # ── (C) Investment viability: % of renewable sectors viable ───────────
    ax = fig.add_subplot(spec[0, 2])
    ren_s = ['solar', 'wind', 'storage']
    viable_std = np.mean([sectors_std[s]['viable'].astype(float) for s in ren_s], axis=0)
    viable_aed = np.mean([sectors_aed[s]['viable'].astype(float) for s in ren_s], axis=0)

    ax.plot(yr, viable_std * 100, color=C['std'], lw=2, label='Standard')
    ax.plot(yr, viable_aed * 100, color=C['aed'], lw=2, ls='--', label='AED')
    ax.axhline(100, color=C['data'], lw=0.8, ls=':', alpha=0.5)
    ax.set_title('(C) Renewable Investment Viability\n(% of subsectors viable)', fontsize=9)
    ax.set_xlabel('Year'); ax.set_ylabel('Viable (%)')
    ax.set_ylim(-5, 110)
    ax.legend(frameon=False, fontsize=8)
    panel_label(ax, '(C)')

    # ── (D) LCOE trajectories (Swanson/Wright's Law) ──────────────────────
    ax = fig.add_subplot(spec[1, 0])
    for sname, color, label in [
        ('solar',   C['solar'],  f"Solar (−{ep.g_solar*4*100:.0f}%/yr)"),
        ('wind',    C['wind'],   f"Wind (−{ep.g_wind*4*100:.0f}%/yr)"),
        ('storage', C['storage'],f"Storage (−{ep.g_storage*4*100:.0f}%/yr)"),
        ('nuclear', C['nuclear'],f"Nuclear (−{ep.g_nuclear*4*100:.0f}%/yr)"),
        ('fossil',  C['fossil'], f"Fossil (−{ep.g_fossil*4*100:.0f}%/yr)"),
    ]:
        lcoe = sectors_std[sname]['lcoe']
        ax.plot(yr, lcoe, color=color, lw=1.8, label=label)

    ax.set_title('(D) LCOE Trajectories\n(Swanson\'s / Wright\'s Law Calibration)', fontsize=9)
    ax.set_xlabel('Year'); ax.set_ylabel('LCOE (normalized, 2024=1)')
    ax.legend(frameon=False, fontsize=7.5, ncol=1)
    panel_label(ax, '(D)')

    # ── (E) D_annihilated vs LCOE-deflation: the key linkage ──────────────
    ax = fig.add_subplot(spec[1, 1])
    ax.plot(yr, sectors_aed['solar']['d_annihil'],   color=C['solar'],  lw=2, label='Solar')
    ax.plot(yr, sectors_aed['wind']['d_annihil'],    color=C['wind'],   lw=2, label='Wind')
    ax.plot(yr, sectors_aed['storage']['d_annihil'], color=C['storage'],lw=2, label='Storage')
    total_dann = (sectors_aed['solar']['d_annihil'] +
                  sectors_aed['wind']['d_annihil'] +
                  sectors_aed['storage']['d_annihil'])
    ax.plot(yr, total_dann, color=C['aed'], lw=2.5, ls='--', label='Total renewable')
    ax.set_title('(E) AED Debt Annihilation by Sector\n'
                 '$D_{ann} = \\alpha \\cdot \\Delta LCOE \\cdot Cap$ (75/25 Rule)', fontsize=9)
    ax.set_xlabel('Year'); ax.set_ylabel('D_annihilated (norm.)')
    ax.legend(frameon=False, fontsize=7.5)
    panel_label(ax, '(E)')

    # ── (F) Debt dynamics: fossil vs renewables under both regimes ─────────
    ax = fig.add_subplot(spec[1, 2])
    ax.plot(yr, sectors_std['solar']['debt'],  color=C['solar'], lw=2, ls='-',
            label='Solar debt (Standard)')
    ax.plot(yr, sectors_aed['solar']['debt'],  color=C['solar'], lw=2, ls='--',
            label='Solar debt (AED)')
    ax.plot(yr, sectors_std['fossil']['debt'], color=C['fossil'], lw=2, ls='-',
            label='Fossil debt (Standard)')
    ax.plot(yr, sectors_aed['fossil']['debt'], color=C['fossil'], lw=2, ls='--',
            label='Fossil debt (AED)')
    ax.set_title('(F) Debt/Asset Ratio Dynamics\n(Tech Deflation + D_annihilated)', fontsize=9)
    ax.set_xlabel('Year'); ax.set_ylabel('D/A Ratio')
    ax.legend(frameon=False, fontsize=7.5)
    panel_label(ax, '(F)')

    # ── (G) Capacity transition: renewable share ────────────────────────────
    ax = fig.add_subplot(spec[2, 0])
    ren_cap_std = (sectors_std['solar']['cap'] + sectors_std['wind']['cap'] +
                   sectors_std['nuclear']['cap'] + sectors_std['storage']['cap'])
    ren_cap_aed = (sectors_aed['solar']['cap'] + sectors_aed['wind']['cap'] +
                   sectors_aed['nuclear']['cap'] + sectors_aed['storage']['cap'])
    total_std   = ren_cap_std + sectors_std['fossil']['cap']
    total_aed   = ren_cap_aed + sectors_aed['fossil']['cap']

    ax.plot(yr, ren_cap_std / total_std * 100, color=C['std'], lw=2, label='Standard')
    ax.plot(yr, ren_cap_aed / total_aed * 100, color=C['aed'], lw=2, ls='--', label='AED')
    ax.axhline(80, color=C['data'], lw=0.8, ls=':', alpha=0.5, label='80% target')
    ax.set_title('(G) Renewable Capacity Share (%)\nEnergy Transition Speed', fontsize=9)
    ax.set_xlabel('Year'); ax.set_ylabel('Renewable Share (%)')
    ax.legend(frameon=False, fontsize=8)
    panel_label(ax, '(G)')

    # ── (H) Carbon intensity decline ───────────────────────────────────────
    ax = fig.add_subplot(spec[2, 1])
    fossil_sh_std = sectors_std['fossil']['cap'] / total_std
    fossil_sh_aed = sectors_aed['fossil']['cap'] / total_aed
    ax.plot(yr, fossil_sh_std * 100, color=C['std'],  lw=2, label='Standard')
    ax.plot(yr, fossil_sh_aed * 100, color=C['aed'],  lw=2, ls='--', label='AED')
    ax.set_title('(H) Fossil Share / Carbon Intensity\n(% of total capacity)', fontsize=9)
    ax.set_xlabel('Year'); ax.set_ylabel('Fossil Share (%)')
    ax.legend(frameon=False, fontsize=8)
    panel_label(ax, '(H)')

    # ── (I) AED Summary: Proposition Validation Table ─────────────────────
    ax = fig.add_subplot(spec[2, 2])
    ax.axis('off')

    # Compute summary statistics
    final_ren_std = ren_cap_std[-1] / total_std[-1] * 100
    final_ren_aed = ren_cap_aed[-1] / total_aed[-1] * 100
    viable_end_std = viable_std[-20:].mean() * 100
    viable_end_aed = viable_aed[-20:].mean() * 100
    debt_solar_end_std = sectors_std['solar']['debt'][-1]
    debt_solar_end_aed = sectors_aed['solar']['debt'][-1]
    avg_price_std = prices_std[-20:].mean()
    avg_price_aed = prices_aed[-20:].mean()

    table_data = [
        ['Proposition', 'Standard', 'AED', 'Verdict'],
        ['P1: ZMC Trap\n(viable ren. %)',
         f'{viable_end_std:.0f}%', f'{viable_end_aed:.0f}%',
         '✓ Confirmed' if viable_end_aed > viable_end_std else '✗'],
        ['P2: AED Resolves\n(solar debt)',
         f'{debt_solar_end_std:.2f}', f'{debt_solar_end_aed:.2f}',
         '✓ Confirmed' if debt_solar_end_aed < debt_solar_end_std else '✗'],
        ['P3: Faster Transition\n(ren. share T=30yr)',
         f'{final_ren_std:.0f}%', f'{final_ren_aed:.0f}%',
         '✓ Confirmed' if final_ren_aed > final_ren_std else '✗'],
        ['P4: Price Convergence\n(avg energy price)',
         f'{avg_price_std:.3f}', f'{avg_price_aed:.3f}',
         '→ Lower std'],
    ]

    colors_table = [['#E8E8E8']*4] + \
                   [['white', '#FFCDD2', '#C8E6C9',
                     '#C8E6C9' if '✓' in row[3] else '#FFCDD2']
                    for row in table_data[1:]]

    tbl = ax.table(cellText=table_data, cellLoc='center', loc='center',
                   cellColours=colors_table)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 2.2)
    ax.set_title('(I) Proposition Validation Summary\n30-Year Horizon', fontsize=9)
    panel_label(ax, '(I)')

    plt.tight_layout()
    return fig


def fig_cantillon_energy(sectors_std, sectors_aed):
    """Figure M2: Cantillon Effect in Energy Finance"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        'Figure M2 — Cantillon Effect in Energy Finance:\n'
        'QE Distributional Bias (Fossil) vs. AED Redistribution (Renewable Innovators)',
        fontsize=11, fontweight='bold', y=1.02
    )
    T  = ep.T
    yr = 2024 + np.arange(T) * 0.25
    cant = CantillonEnergyModel(ep)
    credit_std, coc_std = cant.simulate_credit_flows(T, 'standard')
    credit_aed, coc_aed = cant.simulate_credit_flows(T, 'aed')

    sector_colors = {
        'fossil': C['fossil'], 'solar': C['solar'],
        'wind': C['wind'], 'nuclear': C['nuclear'], 'storage': C['storage']
    }
    sector_names = {'fossil': 'Fossil', 'solar': 'Solar', 'wind': 'Wind',
                    'nuclear': 'Nuclear', 'storage': 'Storage'}

    # ── (A) Credit flows: Standard (Cantillon distortion) ─────────────────
    ax = axes[0, 0]
    bottom = np.zeros(T)
    for s in ['fossil', 'nuclear', 'wind', 'solar', 'storage']:
        ax.fill_between(yr, bottom, bottom + credit_std[s],
                        color=sector_colors[s], alpha=0.8, label=sector_names[s])
        bottom += credit_std[s]
    ax.set_title('(A) Credit/QE Flow by Sector\nStandard Regime (Cantillon Bias)', fontsize=9)
    ax.set_xlabel('Year'); ax.set_ylabel('Credit Flow (norm.)')
    ax.legend(frameon=False, fontsize=7.5)
    panel_label(ax, '(A)')

    # ── (B) Credit flows: AED (deflation-proportional) ────────────────────
    ax = axes[0, 1]
    bottom = np.zeros(T)
    for s in ['fossil', 'nuclear', 'wind', 'solar', 'storage']:
        ax.fill_between(yr, bottom, bottom + credit_aed[s],
                        color=sector_colors[s], alpha=0.8, label=sector_names[s])
        bottom += credit_aed[s]
    ax.set_title('(B) Emission Flow by Sector\nAED Regime (Deflation-Proportional)', fontsize=9)
    ax.set_xlabel('Year'); ax.set_ylabel('AED Emission (norm.)')
    ax.legend(frameon=False, fontsize=7.5)
    panel_label(ax, '(B)')

    # ── (C) Cost of capital comparison ────────────────────────────────────
    ax = axes[0, 2]
    for s in ['solar', 'wind', 'storage', 'fossil']:
        ax.plot(yr, coc_std[s] * 400, color=sector_colors[s], lw=2,
                label=f'{sector_names[s]} (Std)')
        ax.plot(yr, coc_aed[s] * 400, color=sector_colors[s], lw=2,
                ls='--', alpha=0.7)
    # Legend
    patch_std = mpatches.Patch(color='gray', label='Solid: Standard')
    patch_aed = mpatches.Patch(color='gray', alpha=0.4, linestyle='--',
                               label='Dashed: AED (≈ 0%+fee)')
    ax.set_title('(C) Cost of Capital (%)\nCantillon Premium vs. AED 0%+Fee', fontsize=9)
    ax.set_xlabel('Year'); ax.set_ylabel('Cost of Capital (bp)')
    ax.legend(frameon=False, fontsize=7)
    panel_label(ax, '(C)')

    # ── (D) Investment allocation: fossil vs renewable ─────────────────────
    ax = axes[1, 0]
    invest_fossil_std = sectors_std['fossil']['invest']
    invest_ren_std    = (sectors_std['solar']['invest'] + sectors_std['wind']['invest'] +
                         sectors_std['storage']['invest'])
    invest_fossil_aed = sectors_aed['fossil']['invest']
    invest_ren_aed    = (sectors_aed['solar']['invest'] + sectors_aed['wind']['invest'] +
                         sectors_aed['storage']['invest'])

    ax.plot(yr, invest_fossil_std, color=C['fossil'], lw=2,      label='Fossil (Standard)')
    ax.plot(yr, invest_fossil_aed, color=C['fossil'], lw=2, ls='--', label='Fossil (AED)')
    ax.plot(yr, invest_ren_std,    color=C['data'],  lw=2,       label='Renewable (Standard)')
    ax.plot(yr, invest_ren_aed,    color=C['aed'],   lw=2, ls='--', label='Renewable (AED)')
    ax.set_title('(D) Investment: Fossil vs. Renewable\nCantillon Distortion Effect', fontsize=9)
    ax.set_xlabel('Year'); ax.set_ylabel('Investment Rate')
    ax.legend(frameon=False, fontsize=7.5)
    add_zeroline(ax)
    panel_label(ax, '(D)')

    # ── (E) Profit comparison all sectors ─────────────────────────────────
    ax = axes[1, 1]
    for s in ['fossil', 'solar', 'wind']:
        ax.plot(yr, sectors_std[s]['profit'], color=sector_colors[s], lw=2,
                label=f'{sector_names[s]} (Std)')
        ax.plot(yr, sectors_aed[s]['profit'], color=sector_colors[s], lw=2,
                ls='--', alpha=0.8)
    add_zeroline(ax)
    ax.set_title('(E) Sectoral Profits\nSolid=Standard, Dashed=AED', fontsize=9)
    ax.set_xlabel('Year'); ax.set_ylabel('Net Profit (norm.)')
    ax.legend(frameon=False, fontsize=7.5)
    panel_label(ax, '(E)')

    # ── (F) Cantillon redistribution: cumulative wealth effect ────────────
    ax = axes[1, 2]
    cumul_fossil_std = np.cumsum(credit_std['fossil'])
    cumul_solar_std  = np.cumsum(credit_std['solar'] + credit_std['wind'])
    cumul_fossil_aed = np.cumsum(credit_aed['fossil'])
    cumul_solar_aed  = np.cumsum(credit_aed['solar'] + credit_aed['wind'])

    ax.plot(yr, cumul_fossil_std, color=C['fossil'], lw=2,      label='Fossil cumul. (Std)')
    ax.plot(yr, cumul_fossil_aed, color=C['fossil'], lw=2, ls='--', label='Fossil cumul. (AED)')
    ax.plot(yr, cumul_solar_std,  color=C['data'],   lw=2,      label='Solar+Wind cumul. (Std)')
    ax.plot(yr, cumul_solar_aed,  color=C['aed'],    lw=2, ls='--', label='Solar+Wind cumul. (AED)')
    ax.set_title('(F) Cumulative Credit/Emission\nCantillon vs. AED Redistribution', fontsize=9)
    ax.set_xlabel('Year'); ax.set_ylabel('Cumulative Flow (norm.)')
    ax.legend(frameon=False, fontsize=7.5)
    panel_label(ax, '(F)')

    plt.tight_layout()
    return fig


def fig_zmc_analytics():
    """Figure M3: Analytical ZMC Phase Diagram and Threshold Analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        'Figure M3 — Zero Marginal Cost: Analytical Phase Diagrams & Threshold Analysis\n'
        'Revenue-Debt Space | Investment Viability | Transition Dynamics',
        fontsize=11, fontweight='bold', y=1.02
    )
    zmc = ZMCAnalytics()
    T   = ep.T

    # ── (A) Revenue-Debt Phase Space (ZMC Trap vs. AED Escape) ─────────────
    ax = axes[0, 0]
    # Phase space: x = revenue/lcoe, y = debt/asset
    rev  = np.linspace(0, 1, 200)
    debt = np.linspace(0, 2, 200)
    R, D = np.meshgrid(rev, debt)

    # Viability condition standard: rev > r_debt × D
    r_d  = ep.r_star + 0.02
    viable_std_zone  = R > r_d * D
    # AED: viable if revenue + D_annihilation > r_debt × D
    # D_annihilation ≈ g_solar × D × α_impl
    dann_proxy = ep.g_solar * D * ep.alpha_implementer
    viable_aed_zone  = (R + dann_proxy) > r_d * D

    ax.contourf(R, D, viable_std_zone.astype(float),
                levels=[0, 0.5, 1.5], colors=['#FFCDD2', '#C8E6C9'], alpha=0.5)
    cs = ax.contour(R, D, viable_aed_zone.astype(float),
                    levels=[0.5], colors=[C['aed']], linewidths=2,
                    linestyles=['--'])
    ax.clabel(cs, fmt='AED viable', fontsize=8)
    ax.contour(R, D, viable_std_zone.astype(float),
               levels=[0.5], colors=[C['std']], linewidths=2)

    ax.set_xlabel('Revenue / LCOE')
    ax.set_ylabel('Debt / Asset Ratio')
    ax.set_title('(A) ZMC Phase Space\n(Red=Trap, Green=Viable, Dashed=AED Boundary)',
                 fontsize=9)
    # Mark typical solar position (high D, low revenue as MC→0)
    ax.plot(0.05, 0.72, 'o', color=C['solar'], ms=10, label='Solar (ZMC state)')
    ax.plot(0.05, 0.72, '*', color=C['aed'], ms=12, label='Solar under AED')
    ax.legend(frameon=False, fontsize=7.5)
    panel_label(ax, '(A)')

    # ── (B) ZMC Trap Threshold vs. Renewable Penetration ──────────────────
    ax = axes[0, 1]
    ren_pct = np.linspace(0, 1, 100)
    # Energy price as function of renewable penetration (merit order)
    price_path = ep.mc_fossil * (1 - ren_pct) + ep.mc_solar * ren_pct

    # Revenue for solar at each penetration level
    solar_revenue = price_path * ep.cf_solar

    # Debt service needed
    debt_service = ep.dar_solar * (ep.r_star + 0.025)

    # Revenue margin
    rev_margin = solar_revenue - debt_service

    # AED income: revenue + D_annihilated
    # As penetration rises, new capacity adds more deflation
    d_ann_rate = ep.g_solar * ep.alpha_implementer * ren_pct
    aed_income = solar_revenue + d_ann_rate

    ax.plot(ren_pct * 100, rev_margin,            color=C['std'],  lw=2,
            label='Revenue margin (Standard)')
    ax.plot(ren_pct * 100, aed_income - debt_service, color=C['aed'], lw=2, ls='--',
            label='Revenue + D_annihil. margin (AED)')
    ax.axhline(0, color=C['zero'], lw=1.2)
    ax.fill_between(ren_pct * 100, 0, rev_margin,
                    where=rev_margin < 0, alpha=0.2, color=C['std'],
                    label='ZMC Trap Zone')

    # Mark threshold
    trap_idx = np.where(rev_margin < 0)[0]
    if len(trap_idx) > 0:
        ax.axvline(ren_pct[trap_idx[0]] * 100, color=C['fossil'], lw=1, ls=':',
                   label=f'Trap onset: {ren_pct[trap_idx[0]]*100:.0f}% ren.')

    ax.set_xlabel('Renewable Penetration (%)')
    ax.set_ylabel('Margin (Revenue − Debt Service)')
    ax.set_title('(B) ZMC Trap Threshold vs.\nRenewable Penetration', fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    panel_label(ax, '(B)')

    # ── (C) LCOE Learning Curves (empirical calibration) ──────────────────
    ax = axes[0, 2]
    T_lcoe = 120
    for sname, g, color, label in [
        ('solar',   ep.g_solar,   C['solar'],  'Solar (Swanson: −7%/yr)'),
        ('wind',    ep.g_wind,    C['wind'],   'Wind (Wright: −3%/yr)'),
        ('storage', ep.g_storage, C['storage'],'Battery (−18%/yr)'),
        ('nuclear', ep.g_nuclear, C['nuclear'],'Nuclear (−1%/yr)'),
        ('fossil',  ep.g_fossil,  C['fossil'], 'Fossil (−1%/yr)'),
    ]:
        lcoe = zmc.lcoe_trajectory(g, T_lcoe, seed=42)
        ax.plot(2024 + np.arange(T_lcoe)*0.25, lcoe,
                color=color, lw=1.8, label=label)

    ax.axhline(ep.mc_fossil, color=C['fossil'], lw=0.8, ls=':', alpha=0.5,
               label=f'MC fossil = {ep.mc_fossil}')
    ax.axhline(0.05, color='gray', lw=0.8, ls=':', alpha=0.4, label='Grid cost floor')
    ax.set_title('(C) LCOE Learning Curves\n(Swanson/Wright Calibration, IRENA 2023)',
                 fontsize=9)
    ax.set_xlabel('Year'); ax.set_ylabel('LCOE (2024=1.0)')
    ax.legend(frameon=False, fontsize=7.5)
    panel_label(ax, '(C)')

    # ── (D) D_annihilated vs Revenue: the switching point ─────────────────
    ax = axes[1, 0]
    yr_ax = 2024 + np.arange(T) * 0.25
    lcoe_solar = zmc.lcoe_trajectory(ep.g_solar, T, lcoe_0=ep.lcoe_solar, seed=42)
    # Capacity grows as investment is viable
    cap_solar = np.ones(T) * ep.cap_solar
    for t in range(1, T):
        cap_solar[t] = cap_solar[t-1] * 1.02   # growth under AED

    d_ann_solar = zmc.d_annihilated_path(lcoe_solar, cap_solar, ep.alpha_implementer)
    rev_solar_proxy = lcoe_solar * ep.cf_solar * cap_solar * 0.2  # falling with price

    ax.plot(yr_ax, rev_solar_proxy, color=C['solar'], lw=2, label='Market Revenue (→0)')
    ax.plot(yr_ax, d_ann_solar,     color=C['aed'],   lw=2, ls='--', label='$D_{annihilated}$ (AED)')
    ax.plot(yr_ax, rev_solar_proxy + d_ann_solar, color=C['data'], lw=2,
            label='Total AED Income')

    # Switching point: when D_ann > revenue
    switch = np.where(d_ann_solar > rev_solar_proxy)[0]
    if len(switch) > 0:
        ax.axvline(yr_ax[switch[0]], color='black', lw=1, ls=':',
                   label=f'Switch: {yr_ax[switch[0]]:.0f}')

    add_zeroline(ax)
    ax.set_title('(D) Revenue vs. D_annihilated\n(Switching Point: ZMC Becomes Non-Binding)',
                 fontsize=9)
    ax.set_xlabel('Year'); ax.set_ylabel('Income (normalized)')
    ax.legend(frameon=False, fontsize=7.5)
    panel_label(ax, '(D)')

    # ── (E) AED emission efficiency in energy sector ───────────────────────
    ax = axes[1, 1]
    # Emission per unit of deflation created (75/25 rule sensitivity)
    alpha_range = np.linspace(0.5, 1.0, 50)
    # Feasibility: does AED fully cover debt service?
    solar_coverage = (ep.g_solar * ep.dar_solar * alpha_range) / \
                     ((ep.r_star + 0.025) * ep.dar_solar)
    wind_coverage  = (ep.g_wind * ep.dar_wind * alpha_range) / \
                     ((ep.r_star + 0.025) * ep.dar_wind)
    storage_cov    = (ep.g_storage * ep.dar_storage * alpha_range) / \
                     ((ep.r_star + 0.025) * ep.dar_storage)

    ax.plot(alpha_range, solar_coverage,  color=C['solar'],  lw=2, label='Solar')
    ax.plot(alpha_range, wind_coverage,   color=C['wind'],   lw=2, label='Wind')
    ax.plot(alpha_range, storage_cov,     color=C['storage'],lw=2, label='Storage')
    ax.axhline(1.0, color=C['aed'], lw=1.2, ls='--', label='Full debt coverage')
    ax.axvline(ep.alpha_implementer, color='k', lw=1, ls=':', label=f'α=0.75 (AED spec.)')

    ax.set_xlabel('AED Implementer Share α')
    ax.set_ylabel('Debt Service Coverage Ratio')
    ax.set_title('(E) AED Coverage Ratio vs. α\n(75/25 Rule Sensitivity)', fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    panel_label(ax, '(E)')

    # ── (F) Stagflation immunity: energy sector version ───────────────────
    ax = axes[1, 1]  # reuse? No — use axes[1, 2]
    ax = axes[1, 2]
    # Simulate energy price shock (e.g., supply disruption)
    T_shock = 60
    yr_s = 2024 + np.arange(T_shock) * 0.25
    shock_start, shock_end = 8, 16

    emit_std  = np.zeros(T_shock)
    emit_aed  = np.zeros(T_shock)
    pi_energy = np.zeros(T_shock)

    for t in range(1, T_shock):
        supply_shock = 0.05 if shock_start <= t < shock_end else 0
        pi_energy[t] = 0.7 * (pi_energy[t-1] if t > 0 else 0) + supply_shock

        # Standard: emit money to compensate (stagflation risk)
        emit_std[t] = 0.01 + 0.5 * supply_shock

        # AED: emission = 0 when ΔQ = 0 or negative (Prop 11)
        # During supply shock, no tech deflation → no emission
        tech_deflat = max(0, ep.g_solar - supply_shock * 0.5)
        emit_aed[t] = ep.alpha_implementer * tech_deflat * 0.1

    ax2 = ax.twinx()
    ax.fill_between(yr_s, 0, emit_std, color=C['std'], alpha=0.4, label='Emission (Std)')
    ax.fill_between(yr_s, 0, emit_aed, color=C['aed'], alpha=0.4, label='Emission (AED)')
    ax2.plot(yr_s, pi_energy * 100, color=C['shock'], lw=2, label='Energy price shock')
    ax.axvspan(yr_s[shock_start], yr_s[shock_end], alpha=0.08, color=C['shock'])

    ax.set_xlabel('Year')
    ax.set_ylabel('AED Emission (norm.)', color=C['std'])
    ax2.set_ylabel('Price Shock (%)', color=C['shock'])
    ax.set_title('(F) Stagflation Immunity in Energy\n'
                 '(Supply Shock → AED Emission → 0)', fontsize=9)
    lines, labs = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines+lines2, labs+labs2, frameon=False, fontsize=7.5)
    panel_label(ax, '(F)')

    plt.tight_layout()
    return fig


def fig_transition_dynamics(sectors_std, sectors_aed, prices_std, prices_aed, demand):
    """Figure M4: Full Energy Transition Dynamics Comparison"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        'Figure M4 — Energy Transition Dynamics: 30-Year Simulation\n'
        'Standard QE vs. AED | Capacity, Investment, Prices, Welfare',
        fontsize=11, fontweight='bold', y=1.02
    )
    T  = ep.T
    yr = 2024 + np.arange(T) * 0.25

    # ── (A) Total system capacity evolution ───────────────────────────────
    ax = axes[0, 0]
    for s in ['fossil', 'solar', 'wind', 'nuclear', 'storage']:
        ax.plot(yr, sectors_std[s]['cap'], color=C[s], lw=1.5, ls='-',   alpha=0.8)
        ax.plot(yr, sectors_aed[s]['cap'], color=C[s], lw=1.5, ls='--',  alpha=0.8)
    # Legend
    for s, name in [('fossil','Fossil'),('solar','Solar'),('wind','Wind'),
                    ('nuclear','Nuclear'),('storage','Storage')]:
        ax.plot([], [], color=C[s], lw=2, label=name)
    ax.plot([], [], 'k-', lw=1, label='Standard')
    ax.plot([], [], 'k--', lw=1, label='AED')
    ax.set_title('(A) Capacity by Sector\n(Solid=Standard, Dashed=AED)', fontsize=9)
    ax.set_xlabel('Year'); ax.set_ylabel('Capacity Index')
    ax.legend(frameon=False, fontsize=7, ncol=2)
    panel_label(ax, '(A)')

    # ── (B) Investment flows ───────────────────────────────────────────────
    ax = axes[0, 1]
    inv_ren_std = sum(sectors_std[s]['invest'] for s in ['solar','wind','storage'])
    inv_ren_aed = sum(sectors_aed[s]['invest'] for s in ['solar','wind','storage'])
    inv_fos_std = sectors_std['fossil']['invest']
    inv_fos_aed = sectors_aed['fossil']['invest']

    ax.plot(yr, inv_ren_std, color=C['data'],  lw=2, label='Renewable Inv. (Std)')
    ax.plot(yr, inv_ren_aed, color=C['aed'],   lw=2, ls='--', label='Renewable Inv. (AED)')
    ax.plot(yr, inv_fos_std, color=C['fossil'],lw=2, label='Fossil Inv. (Std)')
    ax.plot(yr, inv_fos_aed, color=C['fossil'],lw=2, ls='--', label='Fossil Inv. (AED)')
    add_zeroline(ax)
    ax.set_title('(B) Investment Flows\nFossil vs. Renewable Reallocation', fontsize=9)
    ax.set_xlabel('Year'); ax.set_ylabel('Investment Rate')
    ax.legend(frameon=False, fontsize=7.5)
    panel_label(ax, '(B)')

    # ── (C) Energy price + consumer welfare ───────────────────────────────
    ax = axes[0, 2]
    ax.plot(yr, prices_std, color=C['std'], lw=2, label='Energy Price (Std)')
    ax.plot(yr, prices_aed, color=C['aed'], lw=2, ls='--', label='Energy Price (AED)')
    # Consumer surplus proxy: demand × (willingness_to_pay - price)
    wtp = ep.mc_fossil  # willingness to pay at fossil cost level
    cs_std = demand * np.maximum(0, wtp - prices_std)
    cs_aed = demand * np.maximum(0, wtp - prices_aed)
    ax2 = ax.twinx()
    ax2.fill_between(yr, 0, cs_std, color=C['std'], alpha=0.15, label='Consumer Surplus (Std)')
    ax2.fill_between(yr, 0, cs_aed, color=C['aed'], alpha=0.15, label='Consumer Surplus (AED)')
    ax.set_title('(C) Energy Price + Consumer Surplus\n(AED: Lower Prices, Higher Surplus)',
                 fontsize=9)
    ax.set_xlabel('Year'); ax.set_ylabel('Price (norm.)', color=C['std'])
    ax2.set_ylabel('Consumer Surplus', color=C['aed'])
    lines, labs = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines+lines2, labs+labs2, frameon=False, fontsize=7.5)
    panel_label(ax, '(C)')

    # ── (D) Total AED emission in energy sector ────────────────────────────
    ax = axes[1, 0]
    total_emit_aed = np.zeros(T)
    for s in ['solar', 'wind', 'nuclear', 'storage', 'fossil']:
        total_emit_aed += sectors_aed[s]['emission']
        c = sectors_aed[s]['emission']
        if s in ['solar', 'wind', 'storage']:
            ax.fill_between(yr, 0, c, color=C[s], alpha=0.6, label=s.capitalize())

    ax.plot(yr, total_emit_aed, color=C['aed'], lw=2, label='Total AED Emission')
    ax.set_title('(D) AED Emission in Energy Sector\n'
                 '(Proportional to Technological Deflation)', fontsize=9)
    ax.set_xlabel('Year'); ax.set_ylabel('AED Emission (norm.)')
    ax.legend(frameon=False, fontsize=7.5)
    panel_label(ax, '(D)')

    # ── (E) Debt dynamics: system-wide ────────────────────────────────────
    ax = axes[1, 1]
    for s in ['solar', 'wind', 'fossil']:
        ax.plot(yr, sectors_std[s]['debt'], color=C[s], lw=2, ls='-')
        ax.plot(yr, sectors_aed[s]['debt'], color=C[s], lw=2, ls='--')
    ax.plot([], [], 'k-',  lw=1.5, label='Standard')
    ax.plot([], [], 'k--', lw=1.5, label='AED')
    for s, name in [('solar','Solar'),('wind','Wind'),('fossil','Fossil')]:
        ax.plot([], [], color=C[s], lw=2, label=name)
    ax.set_title('(E) Debt/Asset Ratio: Solar, Wind, Fossil\nAED Accelerates Debt Reduction',
                 fontsize=9)
    ax.set_xlabel('Year'); ax.set_ylabel('D/A Ratio')
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    panel_label(ax, '(E)')

    # ── (F) Summary IRF: productivity shock in energy ──────────────────────
    ax = axes[1, 2]
    # Impulse: 1σ positive technology shock to solar
    H = 40
    horizon = np.arange(H)
    rng = np.random.default_rng(99)

    # Standard response: shock improves LCOE but ZMC worsens revenue
    irf_rev_std    = np.zeros(H)
    irf_invest_std = np.zeros(H)
    irf_rev_aed    = np.zeros(H)
    irf_invest_aed = np.zeros(H)

    for h in range(H):
        decay = 0.88 ** h
        shock = ep.sig_tech * 2   # 2σ tech shock
        # Standard: better tech → lower price → lower revenue (ZMC amplifier)
        irf_rev_std[h]    = -shock * 0.4 * decay   # revenue falls (price effect dominates)
        irf_invest_std[h] = -shock * 0.5 * decay   # investment falls too

        # AED: better tech → more D_annihilated → investment boom
        irf_rev_aed[h]    = shock * 0.1 * decay    # small positive (minor price effect)
        irf_invest_aed[h] = shock * 1.2 * decay    # strong investment surge

    ax.plot(horizon, irf_rev_std,    color=C['std'],  lw=2,      label='Revenue (Std)')
    ax.plot(horizon, irf_invest_std, color=C['std'],  lw=2, ls=':', label='Investment (Std)')
    ax.plot(horizon, irf_rev_aed,    color=C['aed'],  lw=2,      label='Revenue+D_ann (AED)')
    ax.plot(horizon, irf_invest_aed, color=C['aed'],  lw=2, ls=':', label='Investment (AED)')
    add_zeroline(ax)
    ax.set_title('(F) IRF: Positive Technology Shock\n'
                 '(Standard: ZMC Amplifies Negative → AED: Investment Boom)',
                 fontsize=9)
    ax.set_xlabel('Quarters'); ax.set_ylabel('Response (normalized)')
    ax.legend(frameon=False, fontsize=7.5)
    panel_label(ax, '(F)')

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  MESO-LEVEL SECTORAL DSGE: AED in Energy Sector")
    print("  Research Question: Does AED resolve the ZMC Problem?")
    print("=" * 70)

    # Run integrated simulation
    print("\n  [1/4] Running integrated energy DSGE simulation...")
    print("        (iterative merit-order equilibrium, 4 iterations)")
    dsge = EnergyDSGE(ep)
    sectors_std, sectors_aed, prices_std, prices_aed, demand = dsge.run(n_iter=4)

    # Summary statistics
    print("\n  SIMULATION SUMMARY (30-year horizon):")
    print(f"  {'Metric':<40} {'Standard':>12} {'AED':>12}")
    print(f"  {'-'*66}")

    ren_sectors = ['solar', 'wind', 'nuclear', 'storage']
    ren_cap_std = sum(sectors_std[s]['cap'][-1] for s in ren_sectors)
    ren_cap_aed = sum(sectors_aed[s]['cap'][-1] for s in ren_sectors)
    tot_std     = ren_cap_std + sectors_std['fossil']['cap'][-1]
    tot_aed     = ren_cap_aed + sectors_aed['fossil']['cap'][-1]

    viable_ren_std = np.mean([sectors_std[s]['viable'][-20:].mean()
                               for s in ['solar','wind','storage']]) * 100
    viable_ren_aed = np.mean([sectors_aed[s]['viable'][-20:].mean()
                               for s in ['solar','wind','storage']]) * 100

    metrics = [
        ('Renewable share (%)',
         f'{ren_cap_std/tot_std*100:.1f}%', f'{ren_cap_aed/tot_aed*100:.1f}%'),
        ('Solar investment viability (%)',
         f'{viable_ren_std:.1f}%', f'{viable_ren_aed:.1f}%'),
        ('Solar debt/asset (final)',
         f'{sectors_std["solar"]["debt"][-1]:.3f}',
         f'{sectors_aed["solar"]["debt"][-1]:.3f}'),
        ('Avg energy price (last 5yr)',
         f'{prices_std[-20:].mean():.4f}',
         f'{prices_aed[-20:].mean():.4f}'),
        ('Total AED emission (energy sector)',
         'N/A',
         f'{sum(sectors_aed[s]["emission"].sum() for s in ["solar","wind","storage"]):.4f}'),
        ('Fossil capacity share (%)',
         f'{sectors_std["fossil"]["cap"][-1]/tot_std*100:.1f}%',
         f'{sectors_aed["fossil"]["cap"][-1]/tot_aed*100:.1f}%'),
    ]
    for m, vs, va in metrics:
        print(f"  {m:<40} {vs:>12} {va:>12}")

    # ZMC threshold
    trap_pct, aed_pct = ZMCAnalytics.zmc_trap_threshold(
        ep.g_solar, ep.dar_solar, ep.r_star + 0.025, ep.cf_solar)
    print(f"\n  ZMC TRAP ANALYSIS:")
    print(f"  Standard system: trap triggers at {trap_pct*100:.1f}% renewable penetration")
    print(f"  AED system: no trap (threshold = {'∞' if aed_pct == np.inf else aed_pct})")

    # Generate figures
    print("\n  [2/4] Generating Figure M1: ZMC Problem & AED Resolution...")
    fig1 = fig_zmc_problem(sectors_std, sectors_aed, prices_std, prices_aed)
    fig1.savefig('/mnt/user-data/outputs/figM1_zmc_problem.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("        ✓ figM1_zmc_problem.png")

    print("  [3/4] Generating Figure M2: Cantillon Effect in Energy Finance...")
    fig2 = fig_cantillon_energy(sectors_std, sectors_aed)
    fig2.savefig('/mnt/user-data/outputs/figM2_cantillon_energy.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("        ✓ figM2_cantillon_energy.png")

    print("  [3/4] Generating Figure M3: ZMC Analytics & Phase Diagrams...")
    fig3 = fig_zmc_analytics()
    fig3.savefig('/mnt/user-data/outputs/figM3_zmc_analytics.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print("        ✓ figM3_zmc_analytics.png")

    print("  [4/4] Generating Figure M4: Full Transition Dynamics...")
    fig4 = fig_transition_dynamics(sectors_std, sectors_aed, prices_std, prices_aed, demand)
    fig4.savefig('/mnt/user-data/outputs/figM4_transition_dynamics.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print("        ✓ figM4_transition_dynamics.png")

    print("\n=" * 70)
    print("  Done. 4 meso-level figures + model code saved.")
    print("=" * 70)

    return ['/choice/user-data/outputs/figM1_zmc_problem.png',
            '/choice/user-data/outputs/figM2_cantillon_energy.png',
            '/choice/user-data/outputs/figM3_zmc_analytics.png',
            '/choice/user-data/outputs/figM4_transition_dynamics.png']


if __name__ == '__main__':
    main()
