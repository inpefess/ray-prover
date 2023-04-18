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
import os
from typing import Any, Dict, Optional

import gymnasium
from gym_saturation.wrappers.ast2vec_wrapper import AST2VecWrapper
from gym_saturation.wrappers.duplicate_key_obs import DuplicateKeyObsWrapper
from gymnasium import Env
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.parametric_actions_cartpole import (
    TorchParametricActionsModel,
)
from ray.rllib.examples.random_parametric_agent import (
    RandomParametricAlgorithm,
)
from ray.rllib.models import ModelCatalog

from ray_prover.training_helper import TrainingHelper

EMBEDDING_DIM = 256


class PPOProver(TrainingHelper):
    """
    PPO-based prover experiments helper.

    >>> local_dir = getfixture("tmp_path")
    >>> from gym_saturation.constants import MOCK_TPTP_PROBLEM
    >>> test_arguments = ["--prover", "Vampire", "--max_clauses", "1",
    ...     "--problem_filename", MOCK_TPTP_PROBLEM]
    >>> PPOProver(True, local_dir).train_algorithm(
    ...     test_arguments + ["--random_baseline"])
    == Status ==
     ...
        hist_stats:
          episode_lengths:
          - 1
     ...
    <BLANKLINE>
    >>> PPOProver(True, local_dir).train_algorithm(test_arguments)
    == Status ==
     ...
        hist_stats:
          episode_lengths:
          - 1
     ...
    <BLANKLINE>

    :param arguments_to_parse: command line arguments (or explicitly set ones)
    :param test_run: we use light parameters for testing
    """

    def __init__(
        self,
        test_run: bool = False,
        local_dir: Optional[str] = None,
    ):
        """
        Initialise all.

        :param test_run: we use light parameters for testing
        :param local_dir: local directory to save training results to.
            If ``None`` then Ray default is used
        """
        super().__init__(test_run, local_dir)
        ModelCatalog.register_custom_model(
            "pa_model", TorchParametricActionsModel
        )

    def env_creator(
        self, env_config: Dict[str, Any]
    ) -> Env:  # pragma: no cover
        """
        Return a prover with AST2Vec state representation.

        :param env_config: an environment config
        :returns: an environment
        """
        config_copy = env_config.copy()
        problem_filename = config_copy.pop("problem_filename")
        env = DuplicateKeyObsWrapper(
            AST2VecWrapper(
                gymnasium.make(**config_copy).unwrapped,
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

    def get_algorithm_config(self) -> AlgorithmConfig:
        """
        Return an algorithm config.

        :returns: algorithm config
        """
        if self.parsed_arguments.random_baseline:
            config = AlgorithmConfig(RandomParametricAlgorithm).rollouts(
                rollout_fragment_length=1 if self.test_run else 200
            )
        else:
            config = PPOConfig().training(
                sgd_minibatch_size=1 if self.test_run else 128,
                num_sgd_iter=1 if self.test_run else 30,
            )
        return (
            config.environment(
                self.parsed_arguments.prover
                + os.path.basename(self.parsed_arguments.problem_filename),
                env_config={
                    "id": f"{self.parsed_arguments.prover}-v0",
                    "max_clauses": self.parsed_arguments.max_clauses,
                    "problem_filename": self.parsed_arguments.problem_filename,
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
                            EMBEDDING_DIM * self.parsed_arguments.max_clauses,
                        ),
                        "action_embed_size": EMBEDDING_DIM,
                    },
                },
                train_batch_size=2 if self.test_run else 4000,
            )
        )


if __name__ == "__main__":
    PPOProver().train_algorithm()  # pragma: no cover
