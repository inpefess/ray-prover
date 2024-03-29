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
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import ray
from gymnasium import Env
from ray import air, tune
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.callbacks import make_multi_callbacks
from ray.tune.registry import register_env
from ray.tune.stopper import Stopper

from ray_prover.clauses_metrics import ClauseMetrics
from ray_prover.custom_stopper import CustomStopper


class ClauseRepresentation(Enum):
    """Clause representation service."""

    AST2VEC = 0
    CODEBERT = 1


class TrainingHelper(ABC):
    """Training helper abstract base class."""

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
        self.parsed_arguments = argparse.Namespace()
        self.test_run = test_run
        self.storage_path = storage_path
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
            required=True,
            help="TPTP problem file name",
        )
        parser.add_argument(
            "--second_problem_filename",
            type=str,
            required=False,
            help="a harder problem to solve next",
        )
        parser.add_argument(
            "--clause_representation",
            choices=[
                ClauseRepresentation.AST2VEC.name,
                ClauseRepresentation.CODEBERT.name,
            ],
            required=False,
            help="clause representation service",
        )
        self.parsed_arguments = parser.parse_args(arguments_to_parse)

    def _get_stop_conditions(self) -> Union[Dict[str, Any], Stopper]:
        last_task = os.path.splitext(
            os.path.basename(
                self.parsed_arguments.second_problem_filename
                if self.parsed_arguments.second_problem_filename
                else self.parsed_arguments.problem_filename
            )
        )[0]
        return (
            {"timesteps_total": 1, "episodes_total": 1}
            if self.test_run
            else (
                {"timesteps_total": 44000}
                if self.parsed_arguments.random_baseline
                else CustomStopper(last_task)
            )
        )

    def _ray_loop(self) -> None:
        env_config = {
            "id": f"{self. parsed_arguments.prover}-v0",
            "max_clauses": self.parsed_arguments.max_clauses,
            "problem_filename": self.parsed_arguments.problem_filename,
        }
        config = (
            self.get_algorithm_config()
            .environment(
                self.env_id,
                env_config=env_config,
            )
            .callbacks(make_multi_callbacks([ClauseMetrics]))
        )
        ray.init()
        tuner = tune.Tuner(
            config.algo_class,
            run_config=air.RunConfig(
                stop=self._get_stop_conditions(),
                storage_path=self.storage_path,
            ),
            param_space=config.to_dict(),
        )
        tuner.fit()
        ray.shutdown()

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
        if self.parsed_arguments.clause_representation is not None:
            self.parsed_arguments.clause_representation = ClauseRepresentation[
                self.parsed_arguments.clause_representation
            ]
        self._env_id = (
            self.parsed_arguments.prover
            + os.path.basename(self.parsed_arguments.problem_filename)[:-2]
        )
        register_env(self.env_id, self.env_creator)
        self._ray_loop()

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
