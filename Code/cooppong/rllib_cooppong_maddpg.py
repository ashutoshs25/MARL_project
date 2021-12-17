import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from ray.rllib.utils import try_import_tf
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv,ParallelPettingZooEnv
from pettingzoo.butterfly import cooperative_pong_v3,prison_v3
import tensorflow as tf
import supersuit as ss
import torch
from torch import nn
from supersuit.lambda_wrappers.action_lambda import aec_action_lambda
from gym.spaces import Box, Discrete, Tuple
from ray.rllib.policy.policy import PolicySpec
import numpy as np

ray.init(num_cpus=17, num_gpus=0, ignore_reinit_error=True)

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
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
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


_LOGIT_BOUND = 1e6

def sample_softmax(sm):
    sm = sm - np.max(sm)
    probs = np.exp(sm)
    probs = probs/np.sum(probs)
    try:
        return np.random.choice(a = np.arange(len(sm)), p = probs)
    except ValueError as e:
        logging.warning("softmax failure")
        return 0

def env_creator(args):

    
    env = cooperative_pong_v3.env(ball_speed=9, left_paddle_speed=12,right_paddle_speed=12, cake_paddle=False, max_cycles=100, bounce_randomness=False)

    
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.pad_action_space_v0(env)
    env = ss.pad_observations_v0(env)
    env = ss.dtype_v0(env, 'float32')
    env = ss.resize_v0(env, x_size=84, y_size=84)
    
    env = ss.frame_skip_v0(env, 4)
    env = ss.frame_stack_v1(env, 4)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    env = ss.flatten_v0(env)

    
    
    #env = PettingZooEnv(env)
    env = aec_action_lambda(env,
          lambda action, action_space: sample_softmax(action),
          lambda action_space: Box(shape=(action_space.n,), low=-_LOGIT_BOUND, high=_LOGIT_BOUND))

    env = PettingZooEnv(env)

    return env


if __name__ == "__main__":
    env_name = "cooperative_pong_v3"

    register_env(env_name, lambda config: env_creator({}))

    test_env = env_creator({})
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    
    
    def gen_policy(i):
        
        return (
            None,
            test_env.observation_space,
            test_env.action_space,
            {
                "agent_id": i,
                "use_local_critic": False,
                "obs_space": test_env.observation_space,
                "act_space": test_env.action_space,
            }
        )
    
    policies = {"policy_%d" %i: gen_policy(i) for i in range(2)}
    policy_ids = list(policies.keys())

    

    tune.run(
        "contrib/MADDPG",
        name="MADDPG",
        stop={"timesteps_total": 5000000},
        checkpoint_freq=10,
        local_dir="ray_results/"+env_name,
        config={
            # Environment specific
            "env": env_name,
            # General
            "log_level": "ERROR",
            "framework": "tf",
            "num_gpus": 0,
            "num_workers": 1,
            "compress_observations": False,
            "rollout_fragment_length": 512,
            "train_batch_size": 512,
            "use_local_critic": False,

            "use_state_preprocessor": False,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": (
                    lambda agent_id: policy_ids[int(agent_id[-1])]),
                },
            
        },
    )

    



