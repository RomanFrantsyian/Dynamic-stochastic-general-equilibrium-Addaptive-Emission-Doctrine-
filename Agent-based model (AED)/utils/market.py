"""
Market trading system with order book.
"""
from typing import List, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from model import AEDModel
    from agents.base import AEDAgent


@dataclass
class Offer:
    """Market offer structure."""
    offer_id: int
    seller_id: int
    seller_type: str
    good_type: str
    quantity: float
    price: float


class OrderBook:
    """
    Centralized order book for market trading.

    Cleared after trading sub-round.
    """

    def __init__(self):
        """Initialize empty order book."""
        self.offers: List[Offer] = []
        self.next_offer_id = 0

    def post_offer(self, seller_id: int, seller_type: str,
                   good_type: str, quantity: float, price: float) -> int:
        """
        Post sell offer.

        Args:
            seller_id: Seller unique_id
            seller_type: Seller agent type
            good_type: Type of good
            quantity: Quantity offered
            price: Price per unit

        Returns:
            Offer ID
        """
        offer = Offer(
            offer_id=self.next_offer_id,
            seller_id=seller_id,
            seller_type=seller_type,
            good_type=good_type,
            quantity=quantity,
            price=price
        )
        self.offers.append(offer)
        self.next_offer_id += 1
        return offer.offer_id

    def get_offers(self, good_type: str) -> List[Dict]:
        """
        Get all offers for a specific good type.

        Args:
            good_type: Type of good

        Returns:
            List of offer dictionaries
        """
        return [
            {
                'offer_id': o.offer_id,
                'seller_id': o.seller_id,
                'seller_type': o.seller_type,
                'good_type': o.good_type,
                'quantity': o.quantity,
                'price': o.price
            }
            for o in self.offers if o.good_type == good_type
        ]

    def accept_offer(self, model: 'AEDModel', buyer: 'AEDAgent',
                     offer_id: int, quantity: float) -> bool:
        """
        Accept an offer and execute trade.

        Args:
            model: The AEDModel instance
            buyer: Buyer agent
            offer_id: ID of offer to accept
            quantity: Quantity to purchase

        Returns:
            True if trade successful
        """
        # Find offer
        offer = next((o for o in self.offers if o.offer_id == offer_id), None)
        if not offer:
            return False

        # Check quantity
        if quantity > offer.quantity:
            return False

        # Calculate cost
        cost = quantity * offer.price

        # Check buyer has funds
        if not buyer.has_goods('money', cost):
            return False

        # Get seller
        seller = model.agent_registry.get_agent(offer.seller_type, offer.seller_id)
        if not seller or not seller.has_goods(offer.good_type, quantity):
            return False

        # Execute trade
        buyer.give(seller, 'money', cost)
        seller.give(buyer, offer.good_type, quantity)

        # Update offer
        offer.quantity -= quantity
        if offer.quantity <= 0:
            self.offers.remove(offer)

        return True

    def clear(self):
        """Clear all offers (called after trading sub-round)."""
        self.offers = []
        self.next_offer_id = 0


class TradingMixin:
    """
    Mixin adding trading convenience methods to agents.
    """

    def post_offer(self, good_type: str, quantity: float, price: float) -> int:
        """Post sell offer on order book."""
        return self.model.order_book.post_offer(
            seller_id=self.unique_id,
            seller_type=self.__class__.__name__.lower(),
            good_type=good_type,
            quantity=quantity,
            price=price
        )

    def accept_offer(self, offer_id: int, quantity: float) -> bool:
        """Accept an offer from order book."""
        return self.model.order_book.accept_offer(
            model=self.model,
            buyer=self,
            offer_id=offer_id,
            quantity=quantity
        )
