# VirtualTaobao

This project provides VirtualTaobao simulators trained from the real-data of [Taobao](http://taobao.com), one of the largest online retail platforms. In Taobao, when a customer entered some query, the recommondation system returns a list of items according to the query and the customer profile. The system is expected to return a good list such that customers will have high chances of clicking the items. 

Using VirtualTaobao simulator, one can access a "live" environment just like the real Taobao environment. Virtual customers will be generated once at a time, the virtual customer starts a query, and the recommendation system needs to return a list of items. The virtual customer will decide if it would like to click the items in the list, similar to a real customer.

How VirtualTaobao was trained is described in 
> Jing-Cheng Shi, Yang Yu, Qing Da, Shi-Yong Chen, and An-Xiang Zeng. [Virtual-Taobao: Virtualizing real-world online retail environment for reinforcement learning](https://arxiv.org/abs/1805.10000). In: Proceedings of the 33rd AAAI Conference on Artificial Intelligence (AAAI’19), Honolulu, HI, 2019. 

We release in this repository a VirtualTaobao model for both the recommondation system research and the reinforcement learning research (see the supervised learning and reinforcement learning use cases below). Anyone can use this simulator freely, but should give proper credit to the above reference.

Currently, VirtualTaobao V0 model (VirtualTB-v0) is provided, which was trained from a middle-scaled anonymized Taobao dataset. More larger models will be released soon.

### Installation

```bash
pip install -e .
```

### Simulated Environment
Virtual Taobao simulates the customers, items, and recommendation system. 
* A customer is associated with 13 static attributes that has been one-hot encoded in to 88 binary dimensions, and 3-dimensional dynamic attributes. Here, static/dynmaic means whether the features will change during an interactive process. The attributes information about involve customer age, customer gender, customer browsing history, etc.
* An item is associated with 27-dimensional attributes indicating the price, sales volume, CTR, etc. (For confidence issue, the itemset content is not exposed.)

An interactive process between the system and a customer is as follows
1. Virtual Taobao samples a feature vector of the customer, including both the customer's description and customer's query.
2. The system retrives a set of related items according to the query form the whole itemset.
3. The system uses a model to assign a weight vector corresponding to the item attributes.
4. The system calculates the product between the weight vector and the item attributes for each item, and select the top 10 items with the highest values.
5. The selected 10 items are published to the customer. Then the customer will choose to click on some items (CTR++), browse the next page (start over from step 2 with changed customer features), or leave the platform.

In the above process, the model in step 3, is to be trained. The model inputs the features of the customer, and outputs a 27-dimensional weight vector.

### Usage for Supervised Learning

A data set is prepared at

```
virtualTB/SupervisedLearning/dataset.txt
```

Each line of the data set consists an instance of features, labels, and the number of clicks, separated by tab.

To train a model from the data set, the following codes give an demonstration using PyTorch

```
virtualTB/SupervisedLearning/main.py
```

which contains a full process from data set loading, model training, and model test.

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

As a more complete example using DDPG reinforcement learning algorithm and PyTorch is placed in 
```
virtualTB/ReinforcementLearning/main.py
```

### Acknowledgement

This project is an outcome of a joint work of Nanjing University and Alibaba Group, Inc.
