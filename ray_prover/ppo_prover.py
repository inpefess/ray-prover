#   Copyright 2023 Boris Shminke
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# noqa: D205, D400
"""
PPO example
============
"""
from typing import Any, Dict, List, Optional

import gymnasium as gym
from gym_saturation.wrappers.ast2vec_wrapper import AST2VecWrapper
from gym_saturation.wrappers.duplicate_key_obs import DuplicateKeyObsWrapper
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.parametric_actions_cartpole import (
    TorchParametricActionsModel,
)
from ray.rllib.examples.random_parametric_agent import (
    RandomParametricAlgorithm,
)
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from ray_prover.thompson_sampling import parse_args

EMBEDDING_DIM = 256


def env_creator(env_config: Dict[str, Any]) -> gym.Env:
    """
    Return a prover with AST2Vec state representation.

    :param env_config: an environment config
    :returns: an environment
    """
    config_copy = env_config.copy()
    problem_filename = config_copy.pop("problem_filename")
    env = DuplicateKeyObsWrapper(
        AST2VecWrapper(
            gym.make(**config_copy).unwrapped,
            features_num=EMBEDDING_DIM,
        ),
        # ``ParametricActionsModel`` expects a key 'cart' (from the
        # CartPole environment) to be present in the observation
        # dictionary. We add such a key and use 'avail_actions' as its
        # value, since in case of the given clause algorithm, the clauses
        # to choose from are both actions and observations.
        new_key="cart",
        key_to_duplicate="avail_actions",
    )
    env.set_task(problem_filename)
    return env


def train_ppo(
    arguments_to_parse: Optional[List[str]] = None,
) -> None:
    """
    Train PPO.

    >>> test_arguments = ["--prover", "Vampire", "--max_clauses", "1",
    ...     "--num_iter", "1"]
    >>> train_ppo(test_arguments + ["--random_baseline"])

    :param arguments_to_parse: command line arguments (or explicitly set ones)
    """
    parsed_arguments = parse_args(arguments_to_parse)
    register_env("PPOProver", env_creator)
    ModelCatalog.register_custom_model("pa_model", TorchParametricActionsModel)
    if parsed_arguments.random_baseline:
        config = AlgorithmConfig(RandomParametricAlgorithm)
    else:
        config = PPOConfig()
    algo = (
        config.environment(
            "PPOProver",
            env_config={
                "id": f"{parsed_arguments.prover}-v0",
                "max_clauses": parsed_arguments.max_clauses,
                "problem_filename": parsed_arguments.problem_filename,
            },
            # https://github.com/ray-project/ray/issues/23925
            disable_env_checking=True,
        )
        .framework("torch")
        .training(
            model={
                "custom_model": "pa_model",
                # we pass relevant parameters to ``ParametricActionsModel``
                "custom_model_config": {
                    "true_obs_shape": (
                        EMBEDDING_DIM * parsed_arguments.max_clauses,
                    ),
                    "action_embed_size": EMBEDDING_DIM,
                },
            }
        )
        .build()
    )
    for _ in range(parsed_arguments.num_iter):
        algo.train()


if __name__ == "__main__":
    train_ppo()  # pragma: no cover
