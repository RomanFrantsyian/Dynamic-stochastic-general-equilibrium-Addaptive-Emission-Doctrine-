"""
Implementer Firm agent - technology adoption, production.
"""
from typing import TYPE_CHECKING
from agents.base import FirmAgent

if TYPE_CHECKING:
    from model import AEDModel


class ImplementerFirm(FirmAgent):
    """
    Implementer firm adopting technology from innovators.

    Sector B firms:
    - License technology from innovators
    - Pay royalties
    - Produce goods at lower cost
    """

    def __init__(self, model: 'AEDModel', initial_debt: float = 500_000, initial_capital: float = 500_000):
        super().__init__(model)

        # Technology
        self.technology_level = 1.0
        self.licensed_patents = []

        # Financial
        self.debt = initial_debt
        self.capital = initial_capital
        self.create('money', initial_capital)

        # Production
        self.base_production_cost = 100.0
        self.production_cost = self.base_production_cost

        # Price tracking for deflationary vacuum computation
        self.prev_market_price = None

    def adopt_technology(self):
        """
        License technology from innovators.

        Called in sub-round 2 (Innovation & Patents).
        """
        messages = self.model.message_queue.get_messages(
            topic='patent_available'
        )

        for msg in messages:
            innovator_id = msg['sender_id']
            patent_info = msg['content']

            if patent_info['technology_level'] > self.technology_level:
                self.technology_level = patent_info['technology_level']

                cost_reduction = patent_info['cost_reduction_factor']
                self.production_cost *= (1 - cost_reduction)

                self.licensed_patents.append({
                    'innovator_id': innovator_id,
                    'patent_id': patent_info['patent_id'],
                    'royalty_rate': patent_info['royalty_rate']
                })

                break  # Adopt only one patent per period

    def set_prices(self):
        """
        Set market price based on production costs and margin.
        Saves previous price for deflationary vacuum computation.

        Called in sub-round 3 (Price Setting).
        """
        # Save previous price before updating
        self.prev_market_price = self.market_price

        margin = 0.10  # 10% markup
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
        Pay wages to employees and royalties to innovators.

        Called in sub-round 5 (Wage Payments & Taxes).
        """
        total_wage_bill = self.employees * self.model.config.get('base_wage', 1000.0)

        if self['money'] >= total_wage_bill:
            wage_per_household = total_wage_bill / len(self.model.households) if self.model.households else 0.0

            for household in self.model.households:
                if self['money'] >= wage_per_household:
                    self.give(household, 'money', wage_per_household)
                    household.income += wage_per_household

        for patent_license in self.licensed_patents:
            innovator_id = patent_license['innovator_id']
            royalty_rate = patent_license['royalty_rate']
            royalty_amount = self.revenue * royalty_rate

            if self['money'] >= royalty_amount:
                innovator = self.model.agent_registry.get_agent('innovatorfirm', innovator_id)
                if innovator:
                    self.give(innovator, 'money', royalty_amount)

    def submit_deflation_declaration(self):
        """
        Submit deflation declaration to lender bank.
        Includes market price reduction and patent info for 70/20/10 distribution.

        Called in sub-round 6 (Deflation Verification).
        """
        if self.model.scenario_mode == 'BASELINE':
            return

        # Need previous price to compute vacuum
        if self.prev_market_price is None:
            return

        price_reduction = self.prev_market_price - self.market_price

        if price_reduction > 0 and self.licensed_patents and self.model.commercial_banks:
            bank = self.model.commercial_banks[0]

            # Use the most recently adopted patent
            latest_patent = self.licensed_patents[-1]

            self.model.message_queue.send(
                sender_type='implementerfirm',
                sender_id=self.unique_id,
                recipient_type='commercialbank',
                recipient_id=bank.unique_id,
                topic='deflation_declaration',
                content={
                    'prev_market_price': self.prev_market_price,
                    'current_market_price': self.market_price,
                    'price_reduction': price_reduction,
                    'patent_id': latest_patent['patent_id'],
                    'innovator_id': latest_patent['innovator_id'],
                    # Keep for backward compatibility
                    'production_cost_reduction': self.base_production_cost - self.production_cost,
                    'base_cost': self.base_production_cost,
                    'current_cost': self.production_cost
                }
            )