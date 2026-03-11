"""
Government agent - fiscal policy.
"""
from typing import TYPE_CHECKING
from agents.base import AEDAgent

if TYPE_CHECKING:
    from model import AEDModel


class Government(AEDAgent):
    """
    Government agent managing fiscal policy.

    Responsibilities:
    - Tax collection (BASELINE) or seigniorage (AED)
    - Social spending
    - Infrastructure investment
    """

    def __init__(self, model: 'AEDModel'):
        super().__init__(model)

        # Fiscal state
        self.budget = 0.0
        self.cumulative_revenue = 0.0
        self.cumulative_spending = 0.0

        # Parameters
        self.tax_rate = model.config.get('tax_rate', 0.35)
        self.social_spending_rate = model.config.get('social_spending_rate', 0.60)
        self.infrastructure_spending_rate = model.config.get('infrastructure_spending_rate', 0.40)

    def collect_taxes_or_seigniorage(self):
        """
        Collect taxes (BASELINE) or receive seigniorage (AED).

        Called in sub-round 5 (Wage Payments & Taxes).
        """
        if self.model.scenario_mode == 'BASELINE':
            # Traditional tax collection
            self._collect_corporate_taxes()
        else:
            # AED: Receive seigniorage from emission
            self._receive_seigniorage()

    def _collect_corporate_taxes(self):
        """Collect corporate income tax from firms."""
        for firm in self.model.all_firms:
            if firm.revenue > 0:
                tax = firm.revenue * self.tax_rate
                if firm['money'] >= tax:
                    firm.give(self, 'money', tax)
                    self.budget += tax
                    self.cumulative_revenue += tax

    def _receive_seigniorage(self):
        """Receive seigniorage share of emission."""
        emission = self.model.central_bank.period_emission
        seigniorage = emission * self.model.central_bank.state_seigniorage_share

        if seigniorage > 0:
            self.create('money', seigniorage)
            self.budget += seigniorage
            self.cumulative_revenue += seigniorage

    def distribute_social_emission(self):
        """
        Distribute social emission to households.

        Called in sub-round 7 (Debt Restructuring & Emission).
        """
       # if self.model.scenario_mode == 'BASELINE':
       #     return

        emission = self.model.central_bank.period_emission
        social_share = 1 - self.model.central_bank.state_seigniorage_share
        social_emission = emission * social_share

        if social_emission > 0 and self.model.households:
            per_household = social_emission / len(self.model.households)

            for household in self.model.households:
                self.model.message_queue.send(
                    sender_type='government',
                    sender_id=self.unique_id,
                    recipient_type='household',
                    recipient_id=household.unique_id,
                    topic='social_emission',
                    content={'amount': per_household}
                )

    def invest_in_infrastructure(self):
        """
        Invest in infrastructure (consumption).

        Called in sub-round 4 (Goods Market).
        """
        infrastructure_budget = self.budget * self.infrastructure_spending_rate

        if infrastructure_budget > 0:
            # Purchase goods from market
            offers = self.model.order_book.get_offers('goods')

            remaining_budget = infrastructure_budget

            for offer in offers:
                if remaining_budget <= 0:
                    break

                max_quantity = remaining_budget / offer['price']
                quantity = min(max_quantity, offer['quantity'])

                if quantity > 0:
                    # Accept offer (simplified: assume government is agent 0)
                    cost = quantity * offer['price']
                    remaining_budget -= cost
                    self.cumulative_spending += cost
