"""
Test messaging - tests/test_utils/test_messaging.py
"""
from utils.messaging import MessageQueue


def test_send_and_retrieve():
    """Test message sending and retrieval."""
    queue = MessageQueue()

    queue.send(
        sender_type='firm',
        sender_id=1,
        recipient_type='bank',
        recipient_id=5,
        topic='loan_request',
        content={'amount': 10000}
    )

    messages = queue.get_messages(
        recipient_type='bank',
        recipient_id=5,
        topic='loan_request'
    )

    assert len(messages) == 1
    assert messages[0]['content']['amount'] == 10000


def test_broadcast():
    """Test broadcast functionality."""
    queue = MessageQueue()

    queue.broadcast(
        sender_type='centralbank',
        sender_id=0,
        topic='interest_rate',
        content={'rate': 0.05}
    )

    # Anyone can retrieve broadcast
    messages = queue.get_messages(topic='interest_rate')
    assert len(messages) == 1


def test_clear():
    """Test message clearing."""
    queue = MessageQueue()
    queue.send('firm', 1, 'bank', 5, 'test', {})

    queue.clear()
    messages = queue.get_messages()
    assert len(messages) == 0
