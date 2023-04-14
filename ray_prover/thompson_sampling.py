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
An examples of Thompson sampling
=================================
"""
import argparse
import os
from typing import Any, Dict, List, Optional

import gymnasium as gym
from gym_saturation.wrappers.age_weight_bandit import AgeWeightBandit
from gym_saturation.wrappers.constant_parametric_actions import (
    ConstantParametricActionsWrapper,
)
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.bandit import BanditLinTSConfig
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.tune.registry import register_env


def env_creator(env_config: Dict[str, Any]) -> gym.Env:
    """
    Return a multi-armed-bandit version of a saturation prover.

    :param env_config: an environment config
    :returns: an environment
    """
    env = ConstantParametricActionsWrapper(
        AgeWeightBandit(gym.make(**env_config)),
        avail_actions_key="item",
    )
    problem_filename = os.path.join(
        os.environ["WORK"],
        "data",
        "TPTP-v8.1.2",
        "Problems",
        "SET",
        "SET001-1.p",
    )
    env.set_task(problem_filename)
    return env


# pylint: disable=abstract-method
class PatchedRandomPolicy(RandomPolicy):
    """RandomPolicy from Ray examples misses a couple of methods."""

    # pylint: disable=unused-argument, missing-param-doc
    def load_batch_into_buffer(
        self, batch: SampleBatch, buffer_index: int = 0
    ) -> int:
        """Don't load anything anywhere."""
        return 0

    # pylint: disable=unused-argument
    def learn_on_loaded_batch(
        self, offset: int = 0, buffer_index: int = 0
    ) -> dict:
        """Don't learn anything and return empty results."""
        return {}


# pylint: disable=too-few-public-methods, abstract-method
class RandomAlgorithm(Algorithm):
    """Algorithm taking random actions and not learning anything."""

    # pylint: disable=unused-argument, missing-param-doc
    @classmethod
    def get_default_policy_class(cls, config: AlgorithmConfig) -> RandomPolicy:
        """We created PatchedRandomPolicy exactly for this algorithm."""
        return PatchedRandomPolicy  # type: ignore


def parse_args(
    arguments_to_parse: Optional[List[str]] = None,
) -> argparse.Namespace:
    """
    Parse command line arguments.

    :param arguments_to_parse: command line arguments (or explicitly set ones)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--random_baseline",
        action="store_true",
        help="Run random baseline instead of Thompson sampling",
    )
    parser.add_argument(
        "--prover",
        choices=["Vampire", "iProver"],
        required=True,
        help="Which prover to guide: Vampire or iProver",
    )
    parser.add_argument(
        "--max_clauses",
        type=int,
        required=True,
        help="Maximal number of clauses in the proof state",
    )
    parser.add_argument(
        "--num_iter",
        type=int,
        required=True,
        help="Number of training iterations",
    )
    return parser.parse_args(arguments_to_parse)


def train_thompson_sampling(
    arguments_to_parse: Optional[List[str]] = None,
) -> None:
    """
    Train Thompson sampling.

    >>> test_arguments = ["--prover", "Vampire", "--max_clauses", "1",
    ...     "--num_iter", "1"]
    >>> train_thompson_sampling(test_arguments)
    >>> train_thompson_sampling(test_arguments + ["--random_baseline"])

    :param arguments_to_parse: command line arguments (or explicitly set ones)
    """
    parsed_arguments = parse_args(arguments_to_parse)
    register_env("ProverBandit", env_creator)
    if parsed_arguments.random_baseline:
        config = AlgorithmConfig(RandomAlgorithm).framework("torch")
    else:
        config = BanditLinTSConfig()
    algo = config.environment(
        "ProverBandit",
        env_config={
            "id": f"{parsed_arguments.prover}-v0",
            "max_clauses": parsed_arguments.max_clauses,
        },
    ).build()
    for _ in range(parsed_arguments.num_iter):
        algo.train()


if __name__ == "__main__":
    train_thompson_sampling()  # pragma: no cover
