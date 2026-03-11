"""
Goods inventory management system.
"""
from typing import Dict


class GoodsHolder:
    """
    Mixin providing inventory management.

    Replaces ABCeconomics' built-in goods system with custom implementation.
    """

    def __init__(self):
        """Initialize empty inventory."""
        self.goods: Dict[str, float] = {}

    def create(self, good_type: str, quantity: float):
        """
        Create goods (add to inventory).

        Args:
            good_type: Type of good ('money', 'goods', etc.)
            quantity: Amount to create
        """
        if good_type not in self.goods:
            self.goods[good_type] = 0.0
        self.goods[good_type] += quantity

    def destroy(self, good_type: str, quantity: float):
        """
        Destroy goods (remove from inventory).

        Args:
            good_type: Type of good
            quantity: Amount to destroy

        Raises:
            ValueError: If insufficient goods
        """
        if not self.has_goods(good_type, quantity):
            raise ValueError(f"Insufficient {good_type}: have {self.goods.get(good_type, 0)}, need {quantity}")
        self.goods[good_type] -= quantity

    def give(self, recipient: 'GoodsHolder', good_type: str, quantity: float):
        """
        Transfer goods to another agent.

        Args:
            recipient: Recipient agent
            good_type: Type of good
            quantity: Amount to transfer
        """
        self.destroy(good_type, quantity)
        recipient.create(good_type, quantity)

    def has_goods(self, good_type: str, quantity: float) -> bool:
        """
        Check if agent has sufficient goods.

        Args:
            good_type: Type of good
            quantity: Required amount

        Returns:
            True if sufficient goods available
        """
        return self.goods.get(good_type, 0.0) >= quantity

    def __getitem__(self, good_type: str) -> float:
        """
        Access goods via bracket notation.

        Example: agent['money']
        """
        return self.goods.get(good_type, 0.0)

    def __setitem__(self, good_type: str, quantity: float):
        """Set goods quantity directly."""
        self.goods[good_type] = quantity
