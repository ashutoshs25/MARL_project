from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import argparse
import gym
import os
import random
import ray
import numpy as np
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv,ParallelPettingZooEnv
from pettingzoo.sisl import waterworld_v3
from pettingzoo.atari import quadrapong_v3, pong_v2, foozpong_v2
import copy
import supersuit as ss
import torch
from torch import nn

M = 5  # Menagerie size


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



class MyCallbacks(DefaultCallbacks):
    def __init__(self):
        super(MyCallbacks, self).__init__()
        self.men = []

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        print("trainer.train() result: {} -> {} episodes".format(
            trainer, result["episodes_this_iter"]))
        i = result['training_iteration']    # starts from 1
        # the "shared_policy_1" is the only agent being trained
        print("training iteration:", i)

        if i <= M:
            # menagerie initialisation
            tmp = copy.deepcopy(trainer.get_policy("shared_policy_1").get_weights())
            self.men.append(tmp)

            #tmp3 = copy.deepcopy(trainer.get_policy("shared_policy_3").get_weights())
            #self.men.append(tmp3)

            filename1 = 'file_init_' + str(i) + '.txt'
            textfile1 = open(filename1, "w")
            for element1 in self.men:
                textfile1.write("############# menagerie entries ##################" + "\n")
                textfile1.write(str(element1) + "\n")
            textfile1.close()

        else:
            # the first policy added is erased
            self.men.pop(0)
            # add current training policy in the last position of the menagerie
            w = copy.deepcopy(trainer.get_policy("shared_policy_1").get_weights())
            self.men.append(w)
            # select one policy randomly
            sel = random.randint(0, M-1)

            trainer.set_weights({"shared_policy_2": self.men[sel]})

            weights = ray.put(trainer.get_weights("shared_policy_2"))
            trainer.workers.foreach_worker(
                lambda w: w.set_weights(ray.get(weights))
            )

        filename = 'file' + str(i) + '.txt'
        textfile = open(filename, "w")
        for element in self.men:
            textfile.write("############# menagerie entries ##################" + "\n")
            textfile.write(str(element) + "\n")

        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True


if __name__ == "__main__":

    ray.init(num_cpus=9, num_gpus=1, ignore_reinit_error=True)

    def env_creator(args):

            env = foozpong_v2.parallel_env()
            #env = ss.resize_v0(env,x_size=32,y_size=32,linear_interp=True)
            env = ss.max_observation_v0(env, 2)
            env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)
            env = ss.color_reduction_v0(env, mode='B')
            #env = ss.pad_action_space_v0(env)
            #env = ss.pad_observations_v0(env)
            env = ss.dtype_v0(env, 'float32')
            env = ss.resize_v0(env, x_size=84, y_size=84)
            #env = ss.resize_v0(env, x_size=84, y_size=84)
            env = ss.frame_skip_v0(env, 4)
            env = ss.frame_stack_v1(env, 4)
            #env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
            env = ParallelPettingZooEnv(env)
            #env = ss.pettingzoo_env_to_vec_env_v1(env)
            #env = ss.flatten_v0(env)
            return env

    env = env_creator({})
    register_env("foozpong", env_creator)
    obs_space = env.observation_space
    act_spc = env.action_space

    policies = {"shared_policy_1": (None, obs_space, act_spc, {}),
                "shared_policy_2": (None, obs_space, act_spc, {}),
                #"shared_policy_3": (None, obs_space, act_spc, {}),
                #"shared_policy_4": (None, obs_space, act_spc, {}),
                }

    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)


    
    def policy_mapping_fn(agent_id):
        if agent_id == "first_0":
            return "shared_policy_1"
        elif agent_id == "second_0":
            return "shared_policy_2"
        elif agent_id == "third_0":
            return "shared_policy_1"
        else:
            return "shared_policy_2"
        

    tune.run(
        "PPO",
        name="PPO_self_play_v0",
        stop={"training_iteration":10000}, #{"episodes_total": 50000},
        checkpoint_freq=10,
        local_dir="foozpong_results_final/",
        config={
            # Enviroment specific
            "env": "foozpong",
            "callbacks": MyCallbacks,
            # General
            "framework": "torch",
            "num_gpus": 1,
            "num_workers": 8,
            "num_envs_per_worker": 4,
            "model": {"custom_model": "CNNModelV2"},
            # Method specific
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["shared_policy_1"],
            },
        },
    )
