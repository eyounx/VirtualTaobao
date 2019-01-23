### Installation

```bash
pip install -e .
```

### Usage

```python
import gym
import virtualTB

env = gym.make('VirtualTB-v0')
print(env.action_space)
print(env.observation_space)
print(env.observation_space.low)
print(env.observation_space.high)
state = env.reset()
while True:
    env.render()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    
    if done: break
env.render()
```

### Description

This environment is a simulator of an online shopping platform which challenges the agent with the item pushing problem. It mainly focus on pushing items efficiently according to user's query and interests. The goal of agent is to maximize the IPV(item-detail page view). In one word, the more user clicks, the more rewards agent gets.


**What will happen when user raises a query in this online shopping platform?**
1. Several items related to the query were callbacked to form an itemset.
2. The agent assigns a weight to each attribute of the item.
3. Platform calculate item values according to the weight vector and select 10 items with the highest value.
4. Those items are pushed to the user, and user may click on some items(reward++), browse next page (so the agent need to decide new weight vector) or leave the platform.

### TODO
- Add DDPG/SL baseline
- Add new environment of 1-vs-n item pushing tasks
- Re-train the user-action model