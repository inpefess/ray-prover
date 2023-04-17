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
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.sample_batch import SampleBatch


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
