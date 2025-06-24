# modules/rl_utils.py
import random
from collections import namedtuple, deque
from typing import Any, List, Optional, Tuple # For type hints

# Define the structure of an experience tuple
Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ExperienceReplayBuffer:
    """
    A simple FIFO experience replay buffer for RL agents.
    Stores (state, action, reward, next_state, done) transitions.
    """
    def __init__(self, capacity: int):
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("Replay buffer capacity must be a positive integer.")
        self.memory: deque[Experience] = deque([], maxlen=capacity)
        self.capacity: int = capacity

    def push(self, state: Any, action: Any, reward: float, next_state: Any | None, done: bool):
        """ Saves an experience tuple to the buffer. """
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size: int) -> Optional[List[Experience]]:
        """ Randomly samples a batch of experiences from memory. """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
        if len(self.memory) < batch_size:
            return None
        return random.sample(list(self.memory), batch_size)

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.memory)

if __name__ == '__main__':
    print("--- ExperienceReplayBuffer Test ---")
    buffer_capacity = 100
    replay_buffer = ExperienceReplayBuffer(buffer_capacity)
    print(f"Initialized buffer with capacity: {buffer_capacity}, current size: {len(replay_buffer)}")
    # ... (rest of the test from previous successful implementation) ...
    for i in range(5):
        replay_buffer.push([float(i)], i % 2, float(i/10), [float(i+1)] if i<4 else None, i==4)
    print(f"Buffer size after 5 pushes: {len(replay_buffer)}")
    sample = replay_buffer.sample(3)
    print(f"Sampled batch of 3: {sample is not None and len(sample) == 3}")
    for i in range(buffer_capacity): replay_buffer.push([float(i+5)],0,0,None,True)
    print(f"Buffer size after filling to capacity: {len(replay_buffer)}")
    assert len(replay_buffer) == buffer_capacity
    print("--- ExperienceReplayBuffer Test Complete ---")
