"""
Base agent classes for AED model - Mesa 3.5.0 compliant.
"""
import mesa
from typing import TYPE_CHECKING
from utils.goods_system import GoodsHolder
from utils.messaging import MessagingMixin
from utils.market import TradingMixin

if TYPE_CHECKING:
    from model import AEDModel


class AEDAgent(mesa.Agent, GoodsHolder, MessagingMixin, TradingMixin):
    """
    Base class for all AED agents.

    Combines Mesa's Agent with custom utility systems for:
    - Inventory management (GoodsHolder)
    - Inter-agent messaging (MessagingMixin)
    - Market trading (TradingMixin)

    CRITICAL: Mesa 3.5.0 automatically assigns unique_id.
    Do NOT pass unique_id to __init__ or super().__init__.
    """

    def __init__(self, model: 'AEDModel'):
        """
        Initialize base agent.

        Args:
            model: The AEDModel instance (unique_id assigned automatically)
        """
        # Initialize Mesa Agent (automatic unique_id assignment)
        mesa.Agent.__init__(self, model)

        # Initialize utility mixins
        GoodsHolder.__init__(self)

        # Register with model for agent lookup
        agent_type = self.__class__.__name__.lower()
        model.agent_registry.register(agent_type, self.unique_id, self)


class FirmAgent(AEDAgent):
    """
    Base class for firm agents (Innovator and Implementer).

    Provides common firm functionality:
    - Production based on employees and technology
    - Cost management and pricing
    - Revenue tracking
    """

    def __init__(self, model: 'AEDModel'):
        super().__init__(model)

        # Production state
        self.employees = 0
        self.technology_level = 1.0
        self.productivity_factor = 1.0

        # Financial state
        self.production_cost = 100.0
        self.market_price = 120.0
        self.revenue = 0.0
        self.debt = 0.0

        # Production output
        self.units_produced = 0
        self.units_sold = 0

    def produce(self) -> int:
        """
        Execute production function.

        Production: Q = employees * technology_level * productivity_factor

        Returns:
            Number of units produced
        """
        self.units_produced = int(
            self.employees * self.technology_level * self.productivity_factor
        )
        self.create('goods', self.units_produced)
        return self.units_produced

    def compute_unit_cost(self) -> float:
        """Compute per-unit production cost."""
        if self.units_produced > 0:
            return self.production_cost / self.units_produced
        return 0.0


class HouseholdAgent(AEDAgent):
    """
    Base class for household agents.

    Provides consumption and utility functions.
    """

    def __init__(self, model: 'AEDModel'):
        super().__init__(model)

        # Economic state
        self.income = 0.0
        self.savings = 0.0
        self.wealth = 0.0
        self.consumption_budget = 0.0

        # Employment
        self.employer = None
        self.wage = 0.0

    def consume(self, good_type: str, quantity: float) -> float:
        """
        Consume goods.

        Args:
            good_type: Type of good to consume
            quantity: Amount to consume

        Returns:
            Actual quantity consumed
        """
        if self.has_goods(good_type, quantity):
            self.destroy(good_type, quantity)
            return quantity
        return 0.0
