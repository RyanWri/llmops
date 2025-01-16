from src.poc.replay_buffer import ReplayBuffer
import numpy as np


def test_replay_buffer():
    # Initialize the replay buffer
    buffer = ReplayBuffer(capacity=100)

    # Create a dummy experience
    state = np.zeros((4, 84, 84))  # Example state (4 stacked frames of 84x84)
    action = 2  # Example action
    reward = 1.0  # Example reward
    next_state = np.ones((4, 84, 84))  # Example next state
    done = False  # Example done flag

    # Store the experience in the buffer
    buffer.store(state, action, reward, next_state, done)

    # Print the buffer size
    print(f"Replay Buffer Size: {len(buffer)}")

    # Print the types of the experience tuple
    buffer.print_experience_types()

    # Sample from the buffer
    batch_size = 1
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    # Check sampled batch
    print("Sampled Batch:")
    print(f"States Shape: {states.shape}")
    print(f"Actions Shape: {actions.shape}")
    print(f"Rewards Shape: {rewards.shape}")
    print(f"Next States Shape: {next_states.shape}")
    print(f"Dones Shape: {dones.shape}")


if __name__ == "__main__":
    test_replay_buffer()
