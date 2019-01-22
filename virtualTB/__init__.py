from gym.envs.registration import register

register(
    id='VirtualTB-v0',
    entry_point='virtualTB.envs:VirtualTB',
)