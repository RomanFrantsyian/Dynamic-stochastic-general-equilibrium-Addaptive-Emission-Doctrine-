"""
Innovator Firm agent - R&D, patents, production.
"""
from typing import List, TYPE_CHECKING
from dataclasses import dataclass
from agents.base import FirmAgent

if TYPE_CHECKING:
    from model import AEDModel
from data_structures.patent import Patent


class InnovatorFirm(FirmAgent):
    """
    Innovator firm conducting R&D and licensing technology.

    Sector A firms:
    - Invest in R&D
    - Create patents
    - License technology to implementers
    - Receive echo-royalties from emission
    - Produce goods
    """

    def __init__(self, model: 'AEDModel', initial_debt: float = 2_000_000, initial_capital: float = 1_000_000):
        super().__init__(model)

        # Innovation state
        self.rd_budget = 0.0
        self.patents: List[Patent] = []
        self.cumulative_cost_reduction = 0.0

        # Technology
        self.technology_level = 1.0
        self.base_production_cost = 100.0
        self.production_cost = self.base_production_cost

        # Financial
        self.debt = initial_debt
        self.capital = initial_capital
        self.create('money', initial_capital)

        # DAR tracking
        self.dar_score = 0.0
        self.echo_royalties_received = 0.0

        # Parameters
        self.rd_intensity = model.config.get('innovator_rd_intensity', 0.15)
        self.patent_success_rate = model.config.get('innovator_patent_success_rate', 0.30)

    def innovate(self):
        """
        Conduct R&D and attempt to create patents.

        Called in sub-round 2 (Innovation & Patents).
        """
        # Allocate R&D budget
        self.rd_budget = self.revenue * self.rd_intensity if self.revenue > 0 else 0.0

        if self.rd_budget <= 0:
            return

        # Attempt patent creation
        if self.random.random() < self.patent_success_rate:
            # Successful innovation
            tech_improvement = self.random.uniform(0.05, 0.15)
            cost_reduction = self.random.uniform(0.03, 0.10)

            patent = Patent(
                patent_id=len(self.patents),
                technology_level=self.technology_level * (1 + tech_improvement),
                cost_reduction_factor=cost_reduction,
                royalty_rate=0.05,  # 5% royalty
                is_open=False
            )
            self.patents.append(patent)

            # Update firm technology
            self.technology_level = patent.technology_level
            self.cumulative_cost_reduction += cost_reduction
            self.production_cost = self.base_production_cost * (1 - self.cumulative_cost_reduction)

    def open_patents(self):
        """
        Make patents available for licensing.

        Called in sub-round 2 (Innovation & Patents).
        """
        for patent in self.patents:
            if not patent.is_open:
                patent.is_open = True

                # Broadcast patent availability
                self.model.message_queue.broadcast(
                    sender_type='innovatorfirm',
                    sender_id=self.unique_id,
                    topic='patent_available',
                    content={
                        'patent_id': patent.patent_id,
                        'technology_level': patent.technology_level,
                        'cost_reduction_factor': patent.cost_reduction_factor,
                        'royalty_rate': patent.royalty_rate
                    }
                )

    def set_prices(self):
        """
        Set market price based on production costs and margin.

        Called in sub-round 3 (Price Setting).
        """
        margin = 0.20  # 20% markup
        self.market_price = self.production_cost * (1 + margin)

    def offer_goods(self):
        """
        Post goods for sale on OrderBook.

        Called in sub-round 4 (Goods Market).
        """
        if self['goods'] > 0:
            self.post_offer(
                good_type='goods',
                quantity=self['goods'],
                price=self.market_price
            )

    def pay_wages(self):
        """
        Pay wages to employees (via household pool).

        Called in sub-round 5 (Wage Payments & Taxes).
        """
        total_wage_bill = self.employees * self.model.config.get('base_wage', 1000.0)

        if self['money'] >= total_wage_bill:
            # Distribute to household pool
            wage_per_household = total_wage_bill / len(self.model.households) if self.model.households else 0.0

            for household in self.model.households:
                if self['money'] >= wage_per_household:
                    self.give(household, 'money', wage_per_household)
                    household.income += wage_per_household

    def submit_deflation_declaration(self):
        """
        Submit deflation declaration to lender bank.

        Called in sub-round 6 (Deflation Verification).
        """
        if self.model.scenario_mode == 'BASELINE':
            return

        # Calculate production cost reduction
        cost_reduction = self.base_production_cost - self.production_cost

        if cost_reduction > 0:
            # Find lender (simplified: use first commercial bank)
            if self.model.commercial_banks:
                bank = self.model.commercial_banks[0]

                self.model.message_queue.send(
                    sender_type='innovatorfirm',
                    sender_id=self.unique_id,
                    recipient_type='commercialbank',
                    recipient_id=bank.unique_id,
                    topic='deflation_declaration',
                    content={
                        'production_cost_reduction': cost_reduction,
                        'base_cost': self.base_production_cost,
                        'current_cost': self.production_cost
                    }
                )

    def receive_echo_emission(self):
        """
        Receive echo-royalties from central bank emission.

        Called in sub-round 7 (Debt Restructuring & Emission).
        """
        if self.model.scenario_mode == 'BASELINE':
            return

        # Echo-royalty allocation based on DAR score
        from mechanisms.echo_emission import compute_echo_royalty

        echo_amount = compute_echo_royalty(
            dar_score=self.dar_score,
            total_emission=self.model.central_bank.period_emission,
            dar_registry=self.model.central_bank.dar_registry
        )

        if echo_amount > 0:
            self.create('money', echo_amount)
            self.echo_royalties_received += echo_amount

    def update_dar_info(self):
        """
        Report DAR information to central bank.

        Called in sub-round 8 (Investment).
        """
        # Compute DAR score based on cost reduction
        self.dar_score = self.cumulative_cost_reduction

        # Report to central bank
        self.model.central_bank.dar_registry[self.unique_id] = self.dar_score
