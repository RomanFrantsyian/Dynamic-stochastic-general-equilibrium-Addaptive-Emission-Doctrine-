"""
Inter-agent messaging system.
"""
from typing import List, Dict, Optional


class MessageQueue:
    """
    Centralized message queue for agent communication.

    Messages are step-scoped: available only during current step.
    """

    def __init__(self):
        """Initialize empty message queue."""
        self.messages: List[Dict] = []
        self.current_step = 0

    def send(self, sender_type: str, sender_id: int,
             recipient_type: str, recipient_id: int,
             topic: str, content: Dict):
        """
        Send targeted message to specific recipient.

        Args:
            sender_type: Sender agent type
            sender_id: Sender unique_id
            recipient_type: Recipient agent type
            recipient_id: Recipient unique_id
            topic: Message topic/category
            content: Message payload
        """
        message = {
            'step': self.current_step,
            'sender_type': sender_type,
            'sender_id': sender_id,
            'recipient_type': recipient_type,
            'recipient_id': recipient_id,
            'topic': topic,
            'content': content
        }
        self.messages.append(message)

    def broadcast(self, sender_type: str, sender_id: int,
                  topic: str, content: Dict):
        """
        Broadcast message to all agents.

        Args:
            sender_type: Sender agent type
            sender_id: Sender unique_id
            topic: Message topic/category
            content: Message payload
        """
        message = {
            'step': self.current_step,
            'sender_type': sender_type,
            'sender_id': sender_id,
            'recipient_type': 'broadcast',
            'recipient_id': None,
            'topic': topic,
            'content': content
        }
        self.messages.append(message)

    def get_messages(self, recipient_type: Optional[str] = None,
                     recipient_id: Optional[int] = None,
                     topic: Optional[str] = None) -> List[Dict]:
        """
        Retrieve messages matching filters.

        Args:
            recipient_type: Filter by recipient type
            recipient_id: Filter by recipient ID
            topic: Filter by topic

        Returns:
            List of matching messages
        """
        filtered = []

        for msg in self.messages:
            # Check current step
            if msg['step'] != self.current_step:
                continue

            # Check recipient type
            if recipient_type and msg['recipient_type'] not in (recipient_type, 'broadcast'):
                continue

            # Check recipient ID
            if recipient_id is not None and msg['recipient_id'] not in (recipient_id, None):
                continue

            # Check topic
            if topic and msg['topic'] != topic:
                continue

            filtered.append(msg)

        return filtered

    def clear(self):
        """Clear all messages (called at end of step)."""
        self.messages = []

    def advance_step(self):
        """Increment step counter."""
        self.current_step += 1


class MessagingMixin:
    """
    Mixin adding messaging convenience methods to agents.
    """

    def send_message(self, recipient_type: str, recipient_id: int,
                     topic: str, content: Dict):
        """Send message to specific recipient."""
        self.model.message_queue.send(
            sender_type=self.__class__.__name__.lower(),
            sender_id=self.unique_id,
            recipient_type=recipient_type,
            recipient_id=recipient_id,
            topic=topic,
            content=content
        )

    def broadcast_message(self, topic: str, content: Dict):
        """Broadcast message to all agents."""
        self.model.message_queue.broadcast(
            sender_type=self.__class__.__name__.lower(),
            sender_id=self.unique_id,
            topic=topic,
            content=content
        )

    def get_my_messages(self, topic: Optional[str] = None) -> List[Dict]:
        """Get messages addressed to this agent."""
        return self.model.message_queue.get_messages(
            recipient_type=self.__class__.__name__.lower(),
            recipient_id=self.unique_id,
            topic=topic
        )
