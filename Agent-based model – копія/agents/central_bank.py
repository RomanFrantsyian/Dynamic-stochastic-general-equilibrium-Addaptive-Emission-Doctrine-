"""
Central Bank agent - manages monetary policy and AED mechanisms.
"""
from typing import Dict, List, TYPE_CHECKING
import mesa
from agents.base import AEDAgent

if TYPE_CHECKING:
    from model import AEDModel
from mechanisms.emission import compute_emission_volume, compute_deflationary_vacuum
from mechanisms.debt_restructuring import validate_restructuring_report


class CentralBank(AEDAgent):
    """
    Singleton central bank managing monetary policy and AED mechanisms.

    Responsibilities:
    - Monetary emission computation
    - Debt annihilation tracking
    - DAR (Debt Annihilation Reward) registry management
    - Interest rate setting
    - Deflationary vacuum calculation
    """

    def __init__(self, model: 'AEDModel'):
        super().__init__(model)

        # Monetary aggregates
        self.money_supply = model.config.get('initial_money_supply', 100_000_000)
        self.emission_volume = 0.0
        self.cumulative_emission = 0.0

        # AED parameters
        self.price_target = model.config.get('price_target', 1.0)
        self.inflation_buffer = model.config.get('inflation_buffer', 1.02)
        self.emission_coverage_ratio = model.config.get('emission_coverage_ratio', 0.75)
        self.state_seigniorage_share = model.config.get('state_seigniorage_share', 0.10)

        # AED 70/20/10 distribution shares
        self.implementer_share = model.config.get('implementer_emission_share', 0.70)
        self.innovator_share = model.config.get('innovator_emission_share', 0.20)
        # state_seigniorage_share = 0.10 (already defined above)

        # Policy rates
        self.key_interest_rate = model.config.get('key_interest_rate', 0.05)

        # DAR registry {firm_id: dar_score}
        self.dar_registry: Dict[int, float] = {}

        # Patent vacuums collected this period [{implementer_id, innovator_id, patent_id, price_reduction}]
        self.period_patent_vacuums: List[Dict] = []

        # Tracking variables
        self.deflationary_vacuum = 0.0
        self.total_debt_annihilated = 0.0
        self.fraud_detection_threshold = model.config.get('fraud_detection_threshold', 0.05)

        # Macro aggregates
        self.velocity = 0.0

        # Current period tracking
        self.period_emission = 0.0
        self.period_debt_annihilated = 0.0
        self.period_npl_count = 0

        # Scenario mode
        self.scenario_mode = model.config.get('scenario_mode', 'BASELINE')

    def collect_restructuring_reports(self):
        """
        Collect and validate restructuring reports from commercial banks.
        Also collects patent vacuum data for 70/20/10 distribution.

        Called in sub-round 7 (Debt Restructuring & Emission).
        """
        self.period_debt_annihilated = 0.0
        self.period_npl_count = 0
        self.period_patent_vacuums = []

        messages = self.model.message_queue.get_messages(
            recipient_type='centralbank',
            recipient_id=self.unique_id,
            topic='debt_restructuring'
        )

        for msg in messages:
            bank_id = msg['sender_id']
            content = msg['content']
            debt_annihilated = content['debt_annihilated']
            npl_count = content['npl_count']

            if validate_restructuring_report(content, self.fraud_detection_threshold):
                self.period_debt_annihilated += debt_annihilated
                self.period_npl_count += npl_count

                # Collect patent vacuum data
                for pv in content.get('patent_vacuums', []):
                    self.period_patent_vacuums.append(pv)

                bank = self.model.agent_registry.get_agent('commercialbank', bank_id)
                if bank:
                    bank.eligible_for_compensation = True

    def compute_and_execute_emission(self):
        """
        Compute emission volume and execute distribution.

        BASELINE: Classical QE - emission proportional to money supply (E = M * qe_rate)
        AED:      Emission tied to deflationary vacuum from debt annihilation (E = V * alpha)
                  Distribution: 70% implementers, 20% innovators, 10% government

        Called in sub-round 7 (Debt Restructuring & Emission).
        """
        if self.scenario_mode == 'BASELINE':
            # Classical QE: emission proportional to money supply
            self.period_emission = compute_emission_volume(
                deflationary_vacuum=self.deflationary_vacuum,
                emission_coverage_ratio=self.emission_coverage_ratio,
                money_supply=self.money_supply,
                qe_rate=self.model.config.get('qe_rate', 0.02),
                mode=self.scenario_mode
            )
        else:
            # AED: emission tied to deflationary vacuum from debt annihilation
            self.deflationary_vacuum = compute_deflationary_vacuum(
                debt_annihilated=self.period_debt_annihilated,
                price_target=self.price_target,
                inflation_buffer=self.inflation_buffer
            )
            self.period_emission = compute_emission_volume(
                deflationary_vacuum=self.deflationary_vacuum,
                emission_coverage_ratio=self.emission_coverage_ratio
            )

        if self.period_emission <= 0:
            return

        # Update emission tracking
        self.emission_volume += self.period_emission
        self.cumulative_emission += self.period_emission
        self.total_debt_annihilated += self.period_debt_annihilated

    def distribute_patent_emission(self):
        """
        Distribute AED emission per patent vacuum: 70% implementer, 20% innovator, 10% government.

        Each patent's contribution to total price_reduction determines its share of period_emission.

        Called in sub-round 7 (Debt Restructuring & Emission).
        """
        if self.scenario_mode == 'BASELINE' or self.period_emission <= 0:
            return

        if not self.period_patent_vacuums:
            return

        # Total price reduction across all patents this period
        total_price_reduction = sum(pv['price_reduction'] for pv in self.period_patent_vacuums)

        if total_price_reduction <= 0:
            return

        for pv in self.period_patent_vacuums:
            # Share of emission proportional to this patent's price reduction
            patent_share = pv['price_reduction'] / total_price_reduction
            patent_emission = self.period_emission * patent_share

            implementer_amount = patent_emission * self.implementer_share  # 70%
            innovator_amount = patent_emission * self.innovator_share       # 20%
            gov_amount = patent_emission * self.state_seigniorage_share     # 10%

            # Distribute to implementer
            implementer = self.model.agent_registry.get_agent('implementerfirm', pv['implementer_id'])
            if implementer:
                implementer.create('money', implementer_amount)

            # Distribute to innovator
            innovator = self.model.agent_registry.get_agent('innovatorfirm', pv['innovator_id'])
            if innovator:
                innovator.create('money', innovator_amount)
                innovator.echo_royalties_received += innovator_amount

            # Distribute to government
            self.model.government.create('money', gov_amount)
            self.model.government.budget += gov_amount
            self.model.government.cumulative_revenue += gov_amount

    def broadcast_dar_registry(self):
        """
        Broadcast updated DAR registry to all agents.

        Called in sub-round 9 (Data Collection).
        """
        self.model.message_queue.broadcast(
            sender_type='centralbank',
            sender_id=self.unique_id,
            topic='dar_registry',
            content={'dar_registry': self.dar_registry.copy()}
        )

    def update_macro_aggregates(self):
        """
        Update and broadcast macroeconomic aggregates.

        money_supply is computed as sum of all agents' money holdings
        to accurately reflect the real money stock in the economy.

        Called in sub-round 9 (Data Collection).
        """
        self.money_supply = (
            sum(h['money'] for h in self.model.households) +
            sum(f['money'] for f in self.model.all_firms) +
            sum(b['money'] for b in self.model.commercial_banks) +
            self.model.government['money']
        )

        total_transactions = sum(firm.revenue for firm in self.model.all_firms)
        self.velocity = total_transactions / self.money_supply if self.money_supply > 0 else 0.0

        self.model.message_queue.broadcast(
            sender_type='centralbank',
            sender_id=self.unique_id,
            topic='macro_aggregates',
            content={
                'money_supply': self.money_supply,
                'emission_volume': self.emission_volume,
                'velocity': self.velocity,
                'key_interest_rate': self.key_interest_rate
            }
        )