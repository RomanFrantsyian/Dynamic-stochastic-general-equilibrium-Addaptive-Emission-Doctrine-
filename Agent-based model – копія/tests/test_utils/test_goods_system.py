"""
Test goods system - tests/test_utils/test_goods_system.py
"""
import pytest
from utils.goods_system import GoodsHolder


def test_create_and_access():
    """Test goods creation and access."""
    holder = GoodsHolder()
    holder.create('money', 1000)

    assert holder['money'] == 1000
    assert holder.has_goods('money', 500)
    assert not holder.has_goods('money', 1500)


def test_transfer():
    """Test goods transfer."""
    agent1 = GoodsHolder()
    agent2 = GoodsHolder()

    agent1.create('money', 1000)
    agent1.give(agent2, 'money', 400)

    assert agent1['money'] == 600
    assert agent2['money'] == 400


def test_insufficient_goods():
    """Test error on insufficient goods."""
    holder = GoodsHolder()
    holder.create('money', 100)

    with pytest.raises(ValueError):
        holder.destroy('money', 200)
