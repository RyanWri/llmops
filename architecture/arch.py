from diagrams import Diagram, Cluster
from diagrams.aws.compute import EC2
from diagrams.aws.ml import Sagemaker
from diagrams.generic.storage import Storage
from diagrams.onprem.compute import Server
from diagrams.elastic.agent import Integrations
from diagrams.onprem.client import User


def system_components(filename):
    # Create the high-level architecture diagram
    with Diagram(filename, direction="TB", show=False):
        # Clusters for logical grouping
        environment = Server("Environment (e.g., CartPole)")
        replay_buffer = Storage("Replay Buffer")
        agent = Integrations("Q-Network")  # Represents the neural network
        training = EC2("Training Pipeline")
        forecasting = Sagemaker("Memory Forecasting")


def data_flow(filename):
    # Create the high-level architecture diagram
    with Diagram(filename, show=False):
        user = User("User")

        # Environment
        environment = Server("Environment (e.g., CartPole)")

        # Replay Buffer
        replay_buffer = Storage("Replay Buffer")

        with Cluster("Computation Layer"):
            # Neural Network
            agent = Integrations("Q-Network")
            # Training Process
            training = EC2("Training Pipeline")

        with Cluster("Forecast Layer"):
            # Forecasting Module
            forecasting = Sagemaker("Memory Forecasting")

        # Data Flow Connections
        environment >> replay_buffer
        training >> agent
        agent >> replay_buffer
        replay_buffer >> agent
        training >> forecasting
        forecasting >> user
        user >> forecasting


if __name__ == "__main__":
    system_components(filename="architecture/images/system_components")
    data_flow(filename="architecture/images/data_flow")
