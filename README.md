# Autonomous-Driving-via-RL
Applying RL algorithm methods for autonomous driving in Carla simulator

    Requirements
    1) gym
    1) Carla 0.9.6 from https://github.com/carla-simulator/carla/releases/tag/0.9.6
    2) Install gym style wrapper from https://github.com/cjy1992/gym-carla
    3) Pytorch
    
    Training & Testing -
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
    1) CNN extracts features from Camera frames and Lidar frames & then passes onto Q-network which is then trained via experience replay.
    2) Does okay on straight roads & slightly curve roads, breaks often (as to avoid collisions), follows lane,fails terribly on sharp turns & roundabouts.

##### Reward plot
   ![p](https://github.com/akjayant/Autonomous-Driving-via-RL/blob/main/DQN_Discrete_drive/training_plot.jpg)
      
##### Video  -
  ![p](https://github.com/akjayant/Autonomous-Driving-via-RL/blob/main/DQN_Discrete_drive/runs/video.gif)
 
### 2) DDPG Agent built on contionous actions and primitve reward function - WIP
 *(Timelapse made by Kapwing app)
