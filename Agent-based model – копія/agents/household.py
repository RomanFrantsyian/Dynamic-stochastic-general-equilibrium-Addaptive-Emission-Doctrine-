"""
Household agent - consumption, savings, labor.
"""
from typing import TYPE_CHECKING
from agents.base import HouseholdAgent

if TYPE_CHECKING:
    from model import AEDModel


class Household(HouseholdAgent):
    """
    Household agent consuming goods and supplying labor.

    Responsibilities:
    - Consume goods based on budget
    - Save excess income
    - Invest savings
    - Receive social emission
    """

    def __init__(self, model: 'AEDModel', initial_savings: float = 10_000):
        super().__init__(model)

        # Financial state
        self.savings = initial_savings
        self.create('money', initial_savings)

        # Parameters
        self.consumption_propensity = model.config.get('household_consumption_propensity', 0.8)
        self.savings_rate = 1 - self.consumption_propensity

    def compute_budget(self):
        """
        Compute consumption budget based on available money.

        Uses total available funds (money holdings) as budget base,
        ensuring the economy circulates even when wages arrive after
        the goods market sub-round.

        Called in sub-round 3 (Price Setting).
        """
        self.wealth = self['money']
        self.consumption_budget = self.wealth * self.consumption_propensity

    def buy_goods(self):
        """
        Purchase goods from firms via OrderBook.

        Called in sub-round 4 (Goods Market).
        """
        if self.consumption_budget <= 0:
            return

        # Get available offers from OrderBook
        offers = self.model.order_book.get_offers('goods')

        # Sort by price (buy cheapest first)
        offers_sorted = sorted(offers, key=lambda o: o['price'])

        remaining_budget = self.consumption_budget

        for offer in offers_sorted:
            if remaining_budget <= 0:
                break

            # Calculate quantity to buy
            max_quantity = remaining_budget / offer['price']
            quantity_to_buy = min(max_quantity, offer['quantity'])

            if quantity_to_buy > 0:
                # Accept offer
                success = self.accept_offer(offer['offer_id'], quantity_to_buy)

                if success:
                    cost = quantity_to_buy * offer['price']
                    remaining_budget -= cost

    def consume_goods(self):
        """
        Consume purchased goods.

        Called in sub-round 4 (Goods Market).
        """
        goods_owned = self['goods']
        if goods_owned > 0:
            self.consume('goods', goods_owned)

    def invest_savings(self):
        """
        Invest savings with investors.

        Called in sub-round 8 (Investment).
        """
        # Calculate savings from income
        savings_amount = self.income * self.savings_rate

        if savings_amount > 0 and self['money'] >= savings_amount:
            # Find an investor (simplified: use first investor)
            if self.model.investors:
                investor = self.model.investors[0]

                self.give(investor, 'money', savings_amount)
                self.savings += savings_amount

                # Send investment message
                self.model.message_queue.send(
                    sender_type='household',
                    sender_id=self.unique_id,
                    recipient_type='investor',
                    recipient_id=investor.unique_id,
                    topic='investment',
                    content={'amount': savings_amount}
                )

        # Reset income for next period
        self.income = 0.0

    def receive_social_emission(self):
        """
        Receive social emission distribution from government.

        Called in sub-round 7 (Debt Restructuring & Emission).
        """
        #if self.model.scenario_mode == 'BASELINE':
        #    return

        # Get social emission messages
        messages = self.model.message_queue.get_messages(
            recipient_type='household',
            recipient_id=self.unique_id,
            topic='social_emission'
        )

        for msg in messages:
            emission_amount = msg['content'].get('amount', 0.0)
            if emission_amount > 0:
                self.create('money', emission_amount)
                self.income += emission_amount
