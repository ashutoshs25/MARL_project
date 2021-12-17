import ray
import sys
from ray import tune
from ray.rllib.models import ModelV2,ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from ray.rllib.utils import try_import_tf
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv,PettingZooEnv
from pettingzoo.butterfly import cooperative_pong_v3
import tensorflow as tf
import supersuit as ss
import torch
from torch import nn

ray.init(num_cpus=17, num_gpus=1, ignore_reinit_error=True)




class MLPModelV2(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name="my_model"):
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        # Simplified to one layer.
        input = tf.keras.layers.Input(obs_space.shape, dtype=obs_space.dtype)
        output = tf.keras.layers.Dense(num_outputs, activation=None)
        self.base_model = tf.keras.models.Sequential([input, output])
        #self.register_variables(self.base_model.variables)
    def forward(self, input_dict, state, seq_lens):
        return self.base_model(input_dict["obs"]), []



class CNNModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, model_config, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, model_config, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(
                4,
                32,
                [8, 8],
                stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(
                32,
                64,
                [4, 4],
                stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(
                64,
                64,
                [3, 3],
                stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(3136,512)),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()


def env_creator(args):
    #env = pistonball_v4.parallel_env(n_pistons=20, time_penalty=-0.1, continuous=True, random_drop=True, random_rotate=True, ball_mass=0.75, ball_friction=0.3, ball_elasticity=1.5, max_cycles=125)
    env = cooperative_pong_v3.parallel_env(ball_speed=9, left_paddle_speed=12,right_paddle_speed=12, cake_paddle=False, max_cycles=100, bounce_randomness=False)
    #env = ss.resize_v0(env,x_size=32,y_size=32,linear_interp=True)
    env = ss.color_reduction_v0(env, mode='B')
    #env = ss.pad_action_space_v0(env)
    #env = ss.pad_observations_v0(env)
    env = ss.dtype_v0(env, 'float32')
    env = ss.resize_v0(env, x_size=84, y_size=84)
    #env = ss.resize_v0(env, x_size=84, y_size=84)
    env = ss.frame_skip_v0(env, 4)
    env = ss.frame_stack_v1(env, 4)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    #env = ss.flatten_v0(env)
    return env


if __name__ == "__main__":
    env_name = "cooperative_pong_v3"

    method = sys.argv[1]

    seed = sys.argv[2]

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

    test_env = ParallelPettingZooEnv(env_creator({}))
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    #ModelCatalog.register_custom_model("MLPModel", MLPModel)
    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)
    ModelCatalog.register_custom_model("MLPModelV2", MLPModelV2)

    def gen_policy(i):

    
        config = {
            "model": {
                "custom_model": "CNNModelV2",
            },
            "gamma": 0.99,
        }
        return (None, obs_space, act_space, config)

    policies = {"policy_0": gen_policy(0)}

    policy_ids = list(policies.keys())

    '''
    num_agents = len(test_env.agents)
    #print(num_agents)
    policies = {"policy_{}".format(i): gen_policy(i) for i in range(num_agents)}
    policy_ids = list(policies.keys())
    '''

    
    
    if method == "PPO":
        tune.run(
            "PPO",
            name="PPO-PS",
            stop={"timesteps_total": 1000000},
            checkpoint_freq=10,
            local_dir="cooppong_results_final/"+env_name,
            config={
                # Environment specific
                "env": env_name,
                # General
                "log_level": "ERROR",
                "framework": "torch",
                "seed" : int(seed),
                "num_gpus": 1,
                "num_workers": 4,
                "num_envs_per_worker": 1,
                "compress_observations": False,
                "batch_mode": 'truncate_episodes',
            
            
                # 'use_critic': True,
                'use_gae': True,
                "lambda": 0.95,

                "gamma": .99,

                # "kl_coeff": 0.001,
                # "kl_target": 1000.,
                "kl_coeff": 0.5,
                "clip_param": 0.3,
                'grad_clip': None,
                "entropy_coeff": 0.01,
                'vf_loss_coeff': 0.25,

                "sgd_minibatch_size": 64,
                "num_sgd_iter": 10, # epoc
                'rollout_fragment_length': 512,
                "train_batch_size": 512*4,
                'lr': 2e-05,
                "clip_actions": True,
            

                # Method specific
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
            },
        )

    elif method == "ADQN": 
        tune.run(
                "APEX",
                name="ADQN-PS",
                stop={"timesteps_total": 1000000},
                checkpoint_freq=10,
                local_dir="cooppong_results_final/"+env_name,
                config={

                    "env": env_name,
                    "framework": "torch",
                    "seed" : int(seed),
                    "double_q": True,
                    "dueling": True,
                    "num_atoms": 1,
                    "noisy": False,
                    "n_step": 3,
                    "lr": 5e-5,
                    "adam_epsilon": 1.5e-4,
                    "buffer_size": int(4e5),
                    "exploration_config": {
                        "final_epsilon": 0.01,
                        "epsilon_timesteps": 200000,
                    },
                    "prioritized_replay": True,
                    "prioritized_replay_alpha": 0.5,
                    "prioritized_replay_beta": 0.4,
                    "final_prioritized_replay_beta": 1.0,
                    "prioritized_replay_beta_annealing_timesteps": 2000000,

                    "num_gpus": 1,

                    "log_level": "ERROR",
                    "num_workers": 4,
                    "num_envs_per_worker": 1,
                    "rollout_fragment_length": 512,
                    "train_batch_size": 512*4,
                    "target_network_update_freq": 10000,
                    "timesteps_per_iteration": 15000,
                    "learning_starts": 10000,
                    "compress_observations": False,
                    "gamma": 0.99,
                    # Method specific
                    "multiagent": {
                        "policies": policies,
                        "policy_mapping_fn": (
                            lambda agent_id: policy_ids[0]),
                     },
                    },
                )
            
    





