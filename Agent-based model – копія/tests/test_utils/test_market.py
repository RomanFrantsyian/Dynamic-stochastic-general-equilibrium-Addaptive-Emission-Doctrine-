"""
Test order book - tests/test_utils/test_market.py
"""
from model import AEDModel
from utils.market import OrderBook


def test_post_and_retrieve_offers():
    """Test posting and retrieving offers."""
    book = OrderBook()

    offer_id = book.post_offer(
        seller_id=1,
        seller_type='firm',
        good_type='goods',
        quantity=100,
        price=120.0
    )

    offers = book.get_offers('goods')
    assert len(offers) == 1
    assert offers[0]['quantity'] == 100


def test_accept_offer():
    """Test offer acceptance and trade execution."""
    model = AEDModel(scenario='baseline', seed=42)
    model.step()  # Initialize agents

    # Find a firm and household
    firm = model.innovator_firms[0]
    household = model.households[0]

    # Record pre-trade state
    pre_trade_household_money = household['money']
    pre_trade_household_goods = household['goods']

    # Firm posts offer
    firm.create('goods', 100)
    offer_id = firm.post_offer('goods', 100, 120.0)

    # Household accepts
    household.create('money', 15000)
    pre_buy_money = household['money']
    success = household.accept_offer(offer_id, 10)

    assert success
    assert household['goods'] == pre_trade_household_goods + 10
    assert household['money'] == pre_buy_money - (10 * 120)
