from gym.envs.registration import register

register(
        id='myenv-v0',
        entry_point='myenv.env:MyEnv'
)

register(
        id='myenv-v1',
        entry_point='myenv.env_dynamic:MyEnv'
)

register(
        id='myenv-v2',
        entry_point='myenv.env_ex:MyEnv'
)

register(
        id='myenv-v3',
        entry_point='myenv.env_sfm:MyEnv'
)
