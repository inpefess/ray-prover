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
Training Helper
================
"""
import argparse
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import ray
from gymnasium import Env
from ray import air, tune
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.callbacks import MultiCallbacks
from ray.tune.registry import register_env

from ray_prover.clauses_metrics import ClausesMetrics


class TrainingHelper(ABC):
    """Training helper abstract base class."""

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
        self.parsed_arguments = argparse.Namespace()
        self.test_run = test_run
        self.local_dir = local_dir
        self._env_id: Optional[str] = None

    @abstractmethod
    def env_creator(self, env_config: Dict[str, Any]) -> Env:
        """
        Return a configured environment.

        :param env_config: an environment config
        """

    def parse_args(
        self,
        arguments_to_parse: Optional[List[str]] = None,
    ) -> None:
        """
        Parse command line arguments.

        :param arguments_to_parse: command line arguments
            (or explicitly set ones)
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--random_baseline",
            action="store_true",
            help="Run random baseline instead of training an algorithm",
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
            "--problem_filename",
            type=str,
            required=False,
            help="TPTP problem file name",
        )
        self.parsed_arguments = parser.parse_args(arguments_to_parse)

    def train_algorithm(
        self,
        arguments_to_parse: Optional[List[str]] = None,
    ) -> None:
        """
        Train a reinforcement learning algorithm.

        :param arguments_to_parse: command line arguments
            (or explicitly set ones)
        """
        self.parse_args(arguments_to_parse)
        self._env_id = (
            self.parsed_arguments.prover
            + os.path.basename(self.parsed_arguments.problem_filename)[:-2]
        )
        register_env(self.env_id, self.env_creator)
        stop_conditions: Dict[str, Any] = (
            {"timesteps_total": 1, "episodes_total": 1}
            if self.test_run
            else (
                {"episodes_total": 100}
                if self.parsed_arguments.random_baseline
                else {"episode_reward_mean": 0.99}
            )
        )
        config = (
            self.get_algorithm_config()
            .environment(
                self.env_id,
                env_config={
                    "id": f"{self. parsed_arguments.prover}-v0",
                    "max_clauses": self.parsed_arguments.max_clauses,
                    "problem_filename": self.parsed_arguments.problem_filename,
                },
            )
            .callbacks(MultiCallbacks([ClausesMetrics]))
        )
        ray.init()
        tuner = tune.Tuner(
            config.algo_class,
            run_config=air.RunConfig(
                stop=stop_conditions, local_dir=self.local_dir
            ),
            param_space=config.to_dict(),
        )
        tuner.fit()
        ray.shutdown()

    @abstractmethod
    def get_algorithm_config(self) -> AlgorithmConfig:
        """Return an algorithm config."""

    @property
    def env_id(self) -> str:
        """
        Environment label.

        :returns: environment name set previously
        :raises ValueError: if not set previously
        """
        if self._env_id:
            return self._env_id
        raise ValueError("``env_id`` not set!")
