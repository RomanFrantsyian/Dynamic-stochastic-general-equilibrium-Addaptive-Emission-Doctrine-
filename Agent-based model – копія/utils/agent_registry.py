"""
Agent registry for cross-agent lookups.
"""
from typing import Optional, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.base import AEDAgent


class AgentRegistry:
    """
    Registry for looking up agents by (agent_type, agent_id) tuples.

    Enables ABCeconomics-style agent referencing.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._registry: Dict[tuple, 'AEDAgent'] = {}

    def register(self, agent_type: str, agent_id: int, agent: 'AEDAgent'):
        """
        Register an agent.

        Args:
            agent_type: Agent type (class name lowercase)
            agent_id: Agent unique_id
            agent: Agent instance
        """
        key = (agent_type, agent_id)
        self._registry[key] = agent

    def get_agent(self, agent_type: str, agent_id: int) -> Optional['AEDAgent']:
        """
        Look up agent by (type, id).

        Args:
            agent_type: Agent type
            agent_id: Agent unique_id

        Returns:
            Agent instance or None
        """
        key = (agent_type, agent_id)
        return self._registry.get(key)

    def get_agents_by_type(self, agent_type: str) -> List['AEDAgent']:
        """
        Get all agents of a specific type.

        Args:
            agent_type: Agent type

        Returns:
            List of agent instances
        """
        return [
            agent for (atype, aid), agent in self._registry.items()
            if atype == agent_type
        ]
