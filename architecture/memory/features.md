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
