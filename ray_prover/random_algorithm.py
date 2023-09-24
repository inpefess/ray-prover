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
Random Policy and Algorithm
============================
"""
from typing import List, Optional, Tuple, Union

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorStructType, TensorType


# pylint: disable=abstract-method
class PatchedRandomPolicy(RandomPolicy):
    """RandomPolicy from Ray examples misses a couple of methods."""

    # pylint: disable=unused-argument, missing-param-doc
    def load_batch_into_buffer(
        self, batch: SampleBatch, buffer_index: int = 0
    ) -> int:
        """
        Don't load anything anywhere.

        :returns: always zero (no samples loaded)
        """
        return 0  # pragma: no cover

    # pylint: disable=unused-argument
    def learn_on_loaded_batch(
        self, offset: int = 0, buffer_index: int = 0
    ) -> dict:
        """
        Don't learn anything and return empty results.

        :returns: empty dictionary (no metrics computed)
        """
        return {}  # pragma: no cover

    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[
            List[TensorStructType], TensorStructType
        ] = None,
        prev_reward_batch: Union[
            List[TensorStructType], TensorStructType
        ] = None,
        **kwargs,
    ) -> Tuple[list, list, dict]:
        """
        Compute actions for the current policy.

        :param obs_batch: Batch of observations.
        :param state_batches: List of RNN state input batches, if any.
        :param prev_action_batch: Batch of previous action values.
        :param prev_reward_batch: Batch of previous rewards.
        :param kwargs: Forward compatibility placeholder

        :returns:
            actions: Batch of output actions, with shape like
                [BATCH_SIZE, ACTION_SHAPE].
            state_outs (List[TensorType]): List of RNN state output
                batches, if any, each with shape [BATCH_SIZE, STATE_SIZE].
            info (List[dict]): Dictionary of extra feature batches, if any,
                with shape like
                {"f1": [BATCH_SIZE, ...], "f2": [BATCH_SIZE, ...]}.
        """
        return (
            [
                self.action_space_for_sampling.sample()
                for i in range(len(obs_batch))
            ],
            [],
            {},
        )


# pylint: disable=too-few-public-methods, abstract-method
class RandomAlgorithm(Algorithm):
    """Algorithm taking random actions and not learning anything."""

    # pylint: disable=unused-argument, missing-param-doc
    @classmethod
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> RandomPolicy:  # pragma: no cover
        """
        We created PatchedRandomPolicy exactly for this algorithm.

        :returns: patched random policy
        """
        return PatchedRandomPolicy  # type: ignore
