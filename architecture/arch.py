
# uml code for architecture 
"""
@startuml
hide class circle

class Agent {
}

class Experience {
    +state
    +action
    +next_state
    +reward
}

class ReplayBuffer {
    +experiences : List<Experience>
}

class GPU {
}

class ExperienceMemoryEstimator {
}

Agent --> ReplayBuffer : interacts
ReplayBuffer --> GPU : interacts
GPU --> ReplayBuffer : interacts
ExperienceMemoryEstimator --> GPU : interacts
ExperienceMemoryEstimator --> ReplayBuffer : interacts
ExperienceMemoryEstimator --> Agent : interacts

ReplayBuffer "1" *-- "many" Experience : holds

@enduml


"""