# VirtualTaobao

This project provides VirtualTaobao simulators trained from the real-data of [Taobao](http://taobao.com), one of the largest online retail platforms. In Taobao, when a customer entered some query, the recommondation system returns a list of items according to the query and the customer profile. The system is expected to return a good list such that customers will have high chances of clicking the items. 

Using VirtualTaobao simulator, one can access a "live" environment just like the real Taobao environment. Virtual customers will be generated once at a time, the virtual customer starts a query, and the recommendation system needs to return a list of items. The virtual customer will decide if it would like to click the items in the list, similar to a real customer.

How VirtualTaobao was trained is described in 
> Jing-Cheng Shi, Yang Yu, Qing Da, Shi-Yong Chen, and An-Xiang Zeng. [Virtual-Taobao: Virtualizing real-world online retail environment for reinforcement learning](https://arxiv.org/abs/1805.10000). In: Proceedings of the 33rd AAAI Conference on Artificial Intelligence (AAAI’19), Honolulu, HI, 2019. 

We release in this repository a VirtualTaobao model for both the recommondation system research and the reinforcement learning research (see the supervised learning and reinforcement learning use cases below). Anyone can this simulator freely, but should give credit to the above reference.

Currently, VirtualTaobao V0 model (VirtualTB-v0) is provided, which was trained from a middle-scaled anonymized Taobao dataset. More larger models will be released soon.

### Installation

```bash
pip install -e .
```

### What will happen when a user raises a query?
1. Several items related to the query were callbacked to form an itemset. Each item in the itemset have a 27-dimesional attributes indicate the price, sales volume, CTR, etc. (For confidence issue, the itemset will not be exposed.)
2. The agent assigns a weight to each attribute of the item according to user's feature. The user's feature consists of 13-dimensional static attributes(one-hot encoding) and 3-dimensional dynamic attributes, including user's age, gender, browsing history and so on. 
3. Platform calculate the product of weight vector and the item attributes, and select 10 items with the highest value.
4. Those items are pushed to the user, and user may click on some items(reward++), browse next page (so the agent need to decide new weight vector) or leave the platform.

### Usage for Supervised Learning


### Usage for Reinforcement Learning

Here is a simplest example of using VirtualTaobao as an environment for reinforcement learning. A random action is sampled every step to do the recommendation.

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

As a more complete example using DDPG reinforcement learning algorithm is placed in 
```
virtualTB/ReinforcementLearning/main.py
```

### Acknowledgement

This project is an outcome of a joint work of Nanjing University and Alibaba Group, Inc.
