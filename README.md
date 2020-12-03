# Autonomous-Driving-via-RL
Applying RL algorithm methods for autonomous driving in Carla simulator

    Requirements
    1) Carla 0.9.6
    2) Install gym style wrapper from https://github.com/cjy1992/gym-carla

### 1) DQN Model built on Discrete Actions and primitive reward function - DQN-Discrete-drive
Does okay on straight roads & slightly curve roads, breaks often (as to avoid collisions), follows lane,fails terribly on sharp turns & roundabouts.

##### Reward plot
   ![p](https://github.com/akjayant/Autonomous-Driving-via-RL/blob/main/DQN_Discrete_drive/training_plot.jpg)
      
##### Video  -
  ![p](https://github.com/akjayant/Autonomous-Driving-via-RL/blob/main/DQN_Discrete_drive/runs/video.gif)
 
### 2) DDPG Agent built on contionous actions and primitve reward function - WIP
 *(Timelapse made by Kapwing app)
