## rl-market-imitation

Project to simulate the stock market on RL agents.

### About project

The purpose of this project is to create a multi–agent RL-environment, where agents will trade and their behavior will simulate a real market. Discriminator network is going to assess the similarity of the created simulation with the real market. 

### Project assumptions
+ Time is discrete, agents perform actions at the same time;
+ The market works without interruptions;
+ There is only one asset on the market;
+ The market can accept only one request from each agent.

### Simulation mechanics

Each agent submits an application to buy or sell a certain amount of an asset at a fixed price. 
If there are matching orders on the market then they are executing. Agent can cancel a previously submitted request.

### Realization

The environment accepts a list of each agent's actions as input and returns the status, rewards, and episode completion flags for each of the agents. 
Agents are training with the PPO algorithm. The initial capital of each agent is determined randomly. The episode for each agent ends if both assets and cash are equal to zero.


### File description:
+ coding - project implementation;
+ reports - reports of project in Russian.
