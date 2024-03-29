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
from functools import partial
from typing import Any, Dict, Optional

import gymnasium
from gym_saturation.wrappers import AST2VecWrapper
from gym_saturation.wrappers.llmwrapper import LLMWrapper
from gymnasium import Env
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog

from ray_prover.constants import PROBLEM_FILENAME
from ray_prover.curriculum import curriculum_fn
from ray_prover.parametric_actions_model import ParametricActionsModel
from ray_prover.random_algorithm import RandomAlgorithm
from ray_prover.training_helper import ClauseRepresentation, TrainingHelper


class PPOProver(TrainingHelper):
    """
    PPO-based prover experiments helper.

    >>> storage_path = getfixture("tmp_path")
    >>> from gym_saturation.constants import MOCK_TPTP_PROBLEM
    >>> test_arguments = ["--prover", "Vampire", "--max_clauses", "1",
    ...     "--problem_filename", MOCK_TPTP_PROBLEM, "--clause_representation",
    ...     "CODEBERT"]
    >>> PPOProver(True, storage_path).train_algorithm(
    ...     test_arguments + ["--random_baseline"])
    ╭─...
    ...
    ╭─...─╮
    │ Trial RandomAlgorithm_VampireTST001-...result │
    ├─...─┤
    │ episodes_total 2 │
    │ num_env_steps_sampled 2 │
    │ num_env_steps_trained 2 │
    │ sampler_results/episode_len_mean 1 │
    │ sampler_results/episode_reward_mean 0 │
    ╰─...─╯
    ...
    >>> PPOProver(True, storage_path).train_algorithm(test_arguments)
    ╭─...
    ...
    ╭─...─╮
    │ Trial PPO_VampireTST001-... result │
    ├─...─┤
    │ episodes_total 2 │
    │ num_env_steps_sampled 2 │
    │ num_env_steps_trained 2 │
    │ sampler_results/episode_len_mean 1 │
    │ sampler_results/episode_reward_mean 0 │
    ╰─...─╯
    ...

    :param arguments_to_parse: command line arguments (or explicitly set ones)
    :param test_run: we use light parameters for testing
    """

    def __init__(
        self,
        test_run: bool = False,
        storage_path: Optional[str] = None,
    ):
        """
        Initialise all.

        :param test_run: we use light parameters for testing
        :param storage_path: local directory to save training results to.
            If ``None`` then Ray default is used
        """
        super().__init__(test_run, storage_path)
        ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)

    def env_creator(
        self,
        env_config: Dict[str, Any],
    ) -> Env:  # pragma: no cover
        """
        Return a prover with state representation.

        :param env_config: an environment config
        :returns: an environment
        """
        config_copy = env_config.copy()
        problem_filename = config_copy.pop(PROBLEM_FILENAME)
        embedding_dim = (
            256
            if self.parsed_arguments.clause_representation
            == ClauseRepresentation.AST2VEC
            else 768
        )
        wrapper_class = (
            AST2VecWrapper
            if self.parsed_arguments.clause_representation
            == ClauseRepresentation.AST2VEC
            else LLMWrapper
        )
        env = wrapper_class(
            gymnasium.make(**config_copy).unwrapped,
            features_num=embedding_dim,
        )
        env.set_task(problem_filename)
        return env

    def get_algorithm_config(self) -> AlgorithmConfig:
        """
        Return an algorithm config.

        :returns: algorithm config
        """
        if self.parsed_arguments.random_baseline:
            config = AlgorithmConfig(RandomAlgorithm).rollouts(
                rollout_fragment_length=1 if self.test_run else 4000
            )
        else:
            config = PPOConfig().training(
                sgd_minibatch_size=1 if self.test_run else 128,
                num_sgd_iter=1 if self.test_run else 30,
            )
        embedding_dim = (
            256
            if self.parsed_arguments.clause_representation
            == ClauseRepresentation.AST2VEC
            else 768
        )
        return (
            config.environment(
                self.env_id,
                env_config={
                    "id": f"{self.parsed_arguments.prover}-v0",
                    "max_clauses": self.parsed_arguments.max_clauses,
                    "problem_filename": self.parsed_arguments.problem_filename,
                },
                # https://github.com/ray-project/ray/issues/23925
                disable_env_checking=True,
                env_task_fn=partial(
                    curriculum_fn,
                    self.parsed_arguments.second_problem_filename,
                ),
            )
            .framework("torch")
            .training(
                model={
                    "custom_model": "pa_model",
                    # we pass relevant parameters to ``ParametricActionsModel``
                    "custom_model_config": {
                        "true_obs_shape": (
                            embedding_dim * self.parsed_arguments.max_clauses,
                        ),
                        "action_embed_size": embedding_dim,
                    },
                },
                train_batch_size=2 if self.test_run else 4000,
                _enable_learner_api=False,
            )
            .rollouts(num_rollout_workers=0)
            .rl_module(_enable_rl_module_api=False)
        )


if __name__ == "__main__":
    PPOProver().train_algorithm()  # pragma: no cover
