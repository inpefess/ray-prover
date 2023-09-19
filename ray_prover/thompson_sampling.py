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
Example Thompson sampling training
===================================
"""
from typing import Any, Dict

import gymnasium
from gym_saturation.wrappers.age_weight_bandit import AgeWeightBandit
from gym_saturation.wrappers.constant_parametric_actions import (
    ConstantParametricActionsWrapper,
)
from gymnasium import Env
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.bandit import BanditLinTSConfig

from ray_prover.random_algorithm import RandomAlgorithm
from ray_prover.training_helper import TrainingHelper


class ThompsonSampling(TrainingHelper):
    """
    Thompson sampling experiments helper.

    >>> storage_path = getfixture("tmp_path")
    >>> from gym_saturation.constants import MOCK_TPTP_PROBLEM
    >>> test_arguments = ["--prover", "Vampire", "--max_clauses", "1",
    ...     "--problem_filename", MOCK_TPTP_PROBLEM]
    >>> ThompsonSampling(True, storage_path).train_algorithm(
    ...     test_arguments + ["--random_baseline"])
    ╭─...
    ...
    ╭─...─╮
    │ Trial RandomAlgorithm_VampireTST001-... result │
    ├──...─┤
    │ episodes_total 1 │
    │ num_env_steps_sampled 1 │
    │ num_env_steps_trained 1 │
    │ sampler_results/episode_len_mean 1 │
    │ sampler_results/episode_reward_mean 0 │
    ╰─...─╯
    ...
    """

    def env_creator(
        self, env_config: Dict[str, Any]
    ) -> Env:  # pragma: no cover
        """
        Return a configured environment.

        :param env_config: an environment config
        :returns: an environment
        """
        config_copy = env_config.copy()
        problem_filename = config_copy.pop("problem_filename")
        env = ConstantParametricActionsWrapper(
            AgeWeightBandit(gymnasium.make(**config_copy)),
            avail_actions_key="item",
        )
        env.set_task(problem_filename)
        return env

    def get_algorithm_config(self) -> AlgorithmConfig:
        """
        Return an algorithm config.

        :returns: algorithm config
        """
        if self.parsed_arguments.random_baseline:
            config = (
                AlgorithmConfig(RandomAlgorithm)
                .framework("torch")
                .rollouts(rollout_fragment_length=1 if self.test_run else 100)
                .training(train_batch_size=1)
            )
        else:
            config = BanditLinTSConfig().reporting(
                min_sample_timesteps_per_iteration=0 if self.test_run else 100
            )
        return config


if __name__ == "__main__":
    ThompsonSampling().train_algorithm()  # pragma: no cover
