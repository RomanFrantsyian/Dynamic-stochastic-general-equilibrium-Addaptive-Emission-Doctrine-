"""
AED Model - Main simulation orchestrator using Mesa 3.5.0.
"""
import mesa
import yaml
import logging
from typing import List, Dict
from pathlib import Path

from agents.central_bank import CentralBank
from agents.commercial_bank import CommercialBank
from agents.innovator_firm import InnovatorFirm
from agents.implementer_firm import ImplementerFirm
from agents.household import Household
from agents.investor import Investor
from agents.government import Government

from utils.messaging import MessageQueue
from utils.market import OrderBook
from utils.agent_registry import AgentRegistry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('AEDModel')


class AEDModel(mesa.Model):
    """
    Adaptive Economic Doctrine Agent-Based Model.

    Orchestrates the simulation of a monetary economy with AED mechanisms.

    Mesa 3.5.0 Compliance:
    - No schedulers (manual agent iteration)
    - Automatic unique_id assignment
    - Built-in model.steps counter
    - Staged activation via 9 sub-rounds
    """

    def __init__(self, scenario: str = 'baseline', **kwargs):
        """
        Initialize the AED model.

        Args:
            scenario: Scenario name ('baseline', 'aed_pillar1', 'aed_full', 'aed_gradual')
            **kwargs: Override config parameters
        """
        # CRITICAL: Must call super().__init__()
        super().__init__()

        # Load configuration
        self.config = self._load_config(scenario)
        self.config.update(kwargs)

        # Initialize utility systems
        self.message_queue = MessageQueue()
        self.order_book = OrderBook()
        self.agent_registry = AgentRegistry()

        # Simulation state
        self.scenario_mode = self.config.get('scenario_mode', 'BASELINE')
        self.is_gradual_scenario = (self.scenario_mode == 'AED_GRADUAL')
        self.running = True

        # Agent collections (for iteration)
        self.central_bank = None
        self.government = None
        self.commercial_banks: List[CommercialBank] = []
        self.innovator_firms: List[InnovatorFirm] = []
        self.implementer_firms: List[ImplementerFirm] = []
        self.households: List[Household] = []
        self.investors: List[Investor] = []

        # Build agents
        self._build_agents()

        # Convenience properties
        self.all_firms = self.innovator_firms + self.implementer_firms

        # Setup data collection
        self._setup_datacollector()

    def _load_config(self, scenario: str) -> Dict:
        """Load configuration from YAML files."""
        config_dir = Path(__file__).parent / 'config'

        # Load default config
        with open(config_dir / 'default.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Load scenario overrides
        scenario_file = config_dir / f'{scenario}.yaml'
        if scenario_file.exists():
            with open(scenario_file, 'r') as f:
                scenario_config = yaml.safe_load(f)
                config.update(scenario_config)

        return config

    def _build_agents(self):
        """
        Build all agent instances.

        Mesa 3.5.0: No unique_id passed to agent constructors.
        """
        # Singleton agents
        self.central_bank = CentralBank(model=self)
        self.government = Government(model=self)

        # Commercial banks
        num_banks = self.config.get('num_commercial_banks', 5)
        for _ in range(num_banks):
            bank = CommercialBank(
                model=self,
                initial_reserves=self.config.get('bank_initial_reserves', 10_000_000)
            )
            self.commercial_banks.append(bank)

        # Innovator firms
        num_innovators = self.config.get('num_innovator_firms', 20)
        for _ in range(num_innovators):
            firm = InnovatorFirm(
                model=self,
                initial_debt=self.config.get('innovator_initial_debt', 2_000_000),
                initial_capital=self.config.get('innovator_initial_capital', 1_000_000)
            )
            self.innovator_firms.append(firm)

        # Implementer firms
        num_implementers = self.config.get('num_implementer_firms', 100)
        for _ in range(num_implementers):
            firm = ImplementerFirm(
                model=self,
                initial_debt=self.config.get('implementer_initial_debt', 500_000),
                initial_capital=self.config.get('implementer_initial_capital', 500_000)
            )
            self.implementer_firms.append(firm)

        # Households
        num_households = self.config.get('num_households', 500)
        for _ in range(num_households):
            household = Household(
                model=self,
                initial_savings=self.random.uniform(0, 50_000)
            )
            self.households.append(household)

        # Investors
        num_investors = self.config.get('num_investors', 20)
        for _ in range(num_investors):
            investor = Investor(
                model=self,
                initial_capital=self.random.uniform(100_000, 10_000_000)
            )
            self.investors.append(investor)

        # Assign employees to firms (distribute households across firms)
        all_firms_list = self.innovator_firms + self.implementer_firms
        if all_firms_list and self.households:
            employees_per_firm = len(self.households) // len(all_firms_list)
            remainder = len(self.households) % len(all_firms_list)
            for i, firm in enumerate(all_firms_list):
                firm.employees = employees_per_firm + (1 if i < remainder else 0)

        # Create initial loans in banks corresponding to firms' initial debt
        # Distribute firms evenly across banks
        if self.commercial_banks and all_firms_list:
            from data_structures.loan import LoanRecord
            for i, firm in enumerate(all_firms_list):
                if firm.debt > 0:
                    bank = self.commercial_banks[i % len(self.commercial_banks)]
                    loan = LoanRecord(
                        loan_id=len(bank.loans),
                        borrower_id=firm.unique_id,
                        borrower_type=firm.__class__.__name__.lower(),
                        principal=firm.debt,
                        interest_rate=bank.base_interest_rate,
                        is_performing=True
                    )
                    bank.loans.append(loan)
                    bank.total_loans += firm.debt

    def _setup_datacollector(self):
        """Setup Mesa DataCollector."""
        from analysis.metrics import compute_gini

        model_reporters = {
            # === Macroeconomic Aggregates ===
            'GDP': lambda m: sum(f.units_produced * f.market_price for f in m.all_firms),
            'TotalOutput': lambda m: sum(f.units_produced for f in m.all_firms),
            'AveragePriceLevel': lambda m: (
                sum(f.market_price for f in m.all_firms) / len(m.all_firms)
                if m.all_firms else 0
            ),
            'TotalRevenue': lambda m: sum(f.revenue for f in m.all_firms),
            'AggregateConsumption': lambda m: sum(
                h.consumption_budget for h in m.households
            ),

            # === Monetary Sector ===
            'MoneySupply': lambda m: m.central_bank.money_supply,
            'EmissionVolume': lambda m: m.central_bank.emission_volume,
            'PeriodEmission': lambda m: m.central_bank.period_emission,
            'Velocity': lambda m: getattr(m.central_bank, 'velocity', 0.0),
            'KeyInterestRate': lambda m: m.central_bank.key_interest_rate,

            # === Debt & Credit ===
            'TotalFirmDebt': lambda m: sum(f.debt for f in m.all_firms),
            'DebtAnnihilated': lambda m: m.central_bank.total_debt_annihilated,
            'DebtToGDP': lambda m: (
                sum(f.debt for f in m.all_firms) /
                max(sum(f.units_produced * f.market_price for f in m.all_firms), 1)
            ),
            'TotalBankCredit': lambda m: sum(b.total_loans for b in m.commercial_banks),
            'SystemNPLRatio': lambda m: (
                sum(b.npl_ratio for b in m.commercial_banks) / len(m.commercial_banks)
                if m.commercial_banks else 0
            ),

            # === Banking Sector ===
            'TotalBankReserves': lambda m: sum(b.reserves for b in m.commercial_banks),

            # === Household Sector ===
            'Gini': lambda m: compute_gini(m),
            'MedianWealth': lambda m: (
                sorted([h.wealth for h in m.households])[len(m.households) // 2]
                if m.households else 0
            ),
            'MeanWealth': lambda m: (
                sum(h.wealth for h in m.households) / len(m.households)
                if m.households else 0
            ),
            'MedianIncome': lambda m: (
                sorted([h.income for h in m.households])[len(m.households) // 2]
                if m.households else 0
            ),
            'TotalHouseholdMoney': lambda m: sum(h['money'] for h in m.households),

            # === Innovation ===
            'TotalPatents': lambda m: sum(len(f.patents) for f in m.innovator_firms),
            'AverageTechLevel': lambda m: (
                sum(f.technology_level for f in m.innovator_firms) / len(m.innovator_firms)
                if m.innovator_firms else 0
            ),

            # === Government ===
            'GovBudget': lambda m: m.government.budget,
            'GovRevenue': lambda m: m.government.cumulative_revenue,
            'GovSpending': lambda m: m.government.cumulative_spending,
        }

        agent_reporters = {
            'AgentType': lambda a: a.__class__.__name__,
            'Money': lambda a: a['money'],
        }

        self.datacollector = mesa.DataCollector(
            model_reporters=model_reporters,
            agent_reporters=agent_reporters
        )

    def step(self):
        """
        Execute one simulation step with 9 sub-rounds.

        Mesa 3.5.0: model.steps is automatically incremented before this method.
        """
        # === GRADUAL TRANSITION (if applicable) ===
        if self.is_gradual_scenario:
            self._apply_gradual_transition()

        # === SUB-ROUND 1: PRODUCTION ===
        for firm in self.all_firms:
            firm.produce()

        # === SUB-ROUND 2: INNOVATION & PATENTS ===
        for firm in self.innovator_firms:
            firm.innovate()
            firm.open_patents()

        for firm in self.implementer_firms:
            firm.adopt_technology()

        # === SUB-ROUND 3: PRICE SETTING ===
        for firm in self.all_firms:
            firm.set_prices()

        for household in self.households:
            household.compute_budget()

        # === SUB-ROUND 4: GOODS MARKET (TRADING) ===
        # Track pre-trading money for revenue computation
        pre_trade_money = {id(firm): firm['money'] for firm in self.all_firms}

        for firm in self.all_firms:
            firm.offer_goods()

        for household in self.households:
            household.buy_goods()

        self.government.invest_in_infrastructure()

        for household in self.households:
            household.consume_goods()

        self.order_book.clear()

        # Compute revenue from trading (money gained during goods market)
        for firm in self.all_firms:
            firm.revenue = firm['money'] - pre_trade_money[id(firm)]
            if firm.revenue < 0:
                firm.revenue = 0.0

        # === SUB-ROUND 5: WAGE PAYMENTS & TAX/FEES ===
        for firm in self.all_firms:
            firm.pay_wages()

        for bank in self.commercial_banks:
            bank.collect_interest()

        self.government.collect_taxes_or_seigniorage()

        # === SUB-ROUND 6: DEFLATION VERIFICATION ===
        for firm in self.all_firms:
            firm.submit_deflation_declaration()

        for bank in self.commercial_banks:
            bank.verify_deflation_claims()

        # === SUB-ROUND 7: DEBT RESTRUCTURING & EMISSION ===
        self.central_bank.collect_restructuring_reports()
        self.central_bank.compute_and_execute_emission()

        for bank in self.commercial_banks:
            bank.receive_emission_compensation()

        # AED: distribute emission 70% implementers, 20% innovators, 10% government
        self.central_bank.distribute_patent_emission()

        # === SUB-ROUND 8: INVESTMENT ===
        for investor in self.investors:
            investor.update_dar_info()

        for household in self.households:
            household.invest_savings()

        for investor in self.investors:
            investor.allocate_capital()

        for bank in self.commercial_banks:
            bank.process_loan_applications()

        # === SUB-ROUND 9: DATA COLLECTION ===
        self.central_bank.broadcast_dar_registry()
        self.central_bank.update_macro_aggregates()

        # Collect data (self.steps already incremented by Mesa)
        self.datacollector.collect(self)

        # Clear message queue
        self.message_queue.clear()
        self.message_queue.advance_step()

        # Log progress
        logger.info(f"Step {self.steps}: Emission={self.central_bank.period_emission:.0f}, "
                    f"NPLs={self.central_bank.period_npl_count}")

    def _apply_gradual_transition(self):
        """
        Apply gradual parameter changes for AED_GRADUAL scenario.

        Transitions through phases:
        - Steps 0-4: BASELINE
        - Steps 5-9: AED_PILLAR1
        - Steps 10-14: AED_FULL with 15% tax
        - Steps 15+: AED_FULL with 0% tax
        """
        from mechanisms.scenarios import apply_gradual_transition

        # Update config based on current step
        self.config = apply_gradual_transition(self.config, self.steps)

        # Update scenario mode if phase changed
        new_mode = self.config.get('scenario_mode', 'BASELINE')
        if new_mode != self.scenario_mode:
            print(f"Step {self.steps}: Transitioning from {self.scenario_mode} to {new_mode}")
            self.scenario_mode = new_mode

        # Update agent parameters
        self.central_bank.emission_coverage_ratio = self.config.get('emission_coverage_ratio', 0.0)
        self.central_bank.scenario_mode = self.scenario_mode
        self.government.tax_rate = self.config.get('tax_rate', 0.35)

    def run_for(self, num_steps: int):
        """
        Run simulation for specified number of steps.

        Args:
            num_steps: Number of steps to execute
        """
        for _ in range(num_steps):
            self.step()