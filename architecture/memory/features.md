# Data Table Structure

| Feature                   | Description                                                      | Type      |
|---------------------------|------------------------------------------------------------------|-----------|
| Timestamp                 | The time when the data was recorded                             | DateTime  |
| Game Complexity           | Complexity level or ID representing the complexity of the game   | Int/String|
| Number of Game States     | Approximate number of distinct states in the game                | Int       |
| Input Dimensions          | Dimensions of the input (e.g., frame width x height x channels)  | String    |
| Network Architecture      | Description or identifier of the neural network architecture     | String    |
| Number of Parameters      | Total number of trainable parameters in the network              | Int       |
| Batch Size                | Batch size used during training                                  | Int       |
| Learning Rate             | Learning rate used in the optimizer                              | Float     |
| Buffer Size               | Size of the replay buffer                                        | Int       |
| Sample Size               | Number of samples taken from the replay buffer per update        | Int       |
| Number of Episodes        | Total number of episodes trained on during the sampling interval | Int       |
| Episode Length            | Average length of episodes                                       | Float     |
| Exploration Rate          | Exploration rate used in the epsilon-greedy policy               | Float     |
| Sampling Interval         | Interval at which GPU memory is sampled                          | Float     |
| Time Steps                | Total number of time steps during the interval                   | Int       |
| GPU Memory Usage          | GPU memory consumption at the time of sampling                   | Float     |


# Episode Features

### Constant Features: These are features that do not change throughout the duration of an experiment or series of training episodes. 
    They include:
        1. Game ID: Identifier for the Atari game being played.
        2. Network Architecture: The configuration of the neural network used (e.g., number of layers, types of layers).
        3. Buffer Size: Capacity of the replay buffer.
        4. Input Dimensions: Fixed dimensions of the input that the network accepts.
        
### Dynamic Features: These are features that can change from one episode to the next or are specific to particular episodes. 
    They include:
        1. Episode Reward: Total reward accumulated during an episode.
        2. Number of Steps: The total number of steps taken in an episode.
        3. Episode Length: Duration of the episode in terms of time or total frames.
        4. Exploration Rate: Can vary if using a strategy like epsilon decay.
        5. Entropy of State Visitation: This could change each episode based on the diversity of states visited.
    
### Target Variable: 
        Memory Usage: Actual GPU memory usage, which is your target variable.