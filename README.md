# Autonomous-Driving-via-RL
Applying RL algorithm methods for autonomous driving in Carla simulator.
####  Sensors used - Camera and Lidar

    Requirements
    1) gym
    1) Carla 0.9.6 from https://github.com/carla-simulator/carla/releases/tag/0.9.6
    2) Install gym style wrapper from https://github.com/cjy1992/gym-carla
    3) PyTorch
    
    Training & Testing (for DQN) -
    1) Clone
    git clone https://github.com/akjayant/Autonomous-Driving-via-RL/
    2) Go to CARLA_0.9.6 directory & run the simulator in non-display mode.
    /CARLA_0.9.6$ DISPLAY= ./CarlaUE4.sh -opengl -carla-port=2000
    3) Train from repo root directory
    cd DQN_discrete_drive
    python train.py
    4) Test your agent
    python test.py
  

### 1) DQN Model built on Discrete Actions and primitive reward function - [DQN_Discrete_drive](https://github.com/akjayant/Autonomous-Driving-via-RL/tree/main/DQN_Discrete_drive)
    ver 2.0
    multipath - Treats CAMERA and LiDAR frames seperately, extracts features from respective CNN and then concat.
    concat - Concat CAMERA and LiDAR frames and extracts features froom it.
    
      - Does okay on straight roads & slightly curve roads and follows lane on roundabouts as well occasionally.
      - Fails on sharp turns.
      - Avoids collisions after training but during explorations,it does collide.

### 2) DDPG Agent built on contionous actions and primitve reward function -[DQN_Continuous_drive](https://github.com/akjayant/Autonomous-Driving-via-RL/tree/main/DDPG_Continuous_drive) 
    Doesn't work quite well as of now!
 
