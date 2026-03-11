"""
Test agent registry - tests/test_utils/test_agent_registry.py
"""
from utils.agent_registry import AgentRegistry


class MockAgent:
    """Mock agent for testing."""
    def __init__(self, unique_id):
        self.unique_id = unique_id


def test_register_and_lookup():
    """Test agent registration and lookup."""
    registry = AgentRegistry()
    agent = MockAgent(unique_id=1)

    registry.register('firm', 1, agent)

    result = registry.get_agent('firm', 1)
    assert result is agent


def test_get_agents_by_type():
    """Test getting all agents of a type."""
    registry = AgentRegistry()
    agent1 = MockAgent(unique_id=1)
    agent2 = MockAgent(unique_id=2)
    agent3 = MockAgent(unique_id=3)

    registry.register('firm', 1, agent1)
    registry.register('firm', 2, agent2)
    registry.register('bank', 3, agent3)

    firms = registry.get_agents_by_type('firm')
    assert len(firms) == 2

    banks = registry.get_agents_by_type('bank')
    assert len(banks) == 1


def test_lookup_nonexistent():
    """Test looking up non-existent agent returns None."""
    registry = AgentRegistry()

    result = registry.get_agent('firm', 999)
    assert result is None
