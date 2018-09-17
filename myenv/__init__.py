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

register(
        id='myenv-v4',
        entry_point='myenv.env_sfm2:MyEnv'
)

register(
        id='myenv-v5',
        entry_point='myenv.env_conv:MyEnv'
)

register(
        id='myenv-v6',
        entry_point='myenv.env_ex2:MyEnv'
)

register(
        id='myenv-v7',
        entry_point='myenv.env_ex_test:MyEnv'
)
