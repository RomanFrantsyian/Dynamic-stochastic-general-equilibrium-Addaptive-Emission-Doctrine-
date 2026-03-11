"""
Investor agent - capital allocation.
"""
from typing import TYPE_CHECKING
from agents.base import AEDAgent

if TYPE_CHECKING:
    from model import AEDModel


class Investor(AEDAgent):
    """
    Investor agent managing capital allocation.

    Responsibilities:
    - Collect household savings
    - Allocate capital to firms
    - Track portfolio performance
    """

    def __init__(self, model: 'AEDModel', initial_capital: float = 1_000_000):
        super().__init__(model)

        # Portfolio
        self.capital = initial_capital
        self.create('money', initial_capital)
        self.portfolio_value = initial_capital

        # DAR registry (copied from central bank)
        self.dar_registry = {}

    def update_dar_info(self):
        """
        Update DAR registry from central bank.

        Called in sub-round 8 (Investment).
        """
        # Get DAR registry broadcast
        messages = self.model.message_queue.get_messages(
            topic='dar_registry'
        )

        for msg in messages:
            self.dar_registry = msg['content'].get('dar_registry', {})
            break

    def allocate_capital(self):
        """
        Allocate capital to firms based on DAR scores.

        Called in sub-round 8 (Investment).
        """
        if not self.dar_registry or self['money'] <= 0:
            return

        # Sort firms by DAR score
        sorted_firms = sorted(self.dar_registry.items(), key=lambda x: x[1], reverse=True)

        # Allocate to top firms
        allocation_per_firm = self['money'] / min(len(sorted_firms), 5)

        for firm_id, dar_score in sorted_firms[:5]:
            firm = self.model.agent_registry.get_agent('innovatorfirm', firm_id)

            if firm and allocation_per_firm > 0:
                self.give(firm, 'money', allocation_per_firm)
