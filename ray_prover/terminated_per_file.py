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
Terminated per file Callback
=============================
"""
import os
from typing import Dict, Optional

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy

from ray_prover.constants import PROBLEM_FILENAME


class TerminatedPerFile(DefaultCallbacks):
    """Terminated per file callback."""

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: Optional[int],
        **kwargs,
    ) -> None:
        """
        Run when an episode is done.

        :param worker: Reference to the current roll-out worker.
        :param base_env: BaseEnv running the episode. The underlying
            sub environment objects can be retrieved by calling
            `base_env.get_sub_environments()`.
        :param policies: Mapping of policy id to policy
            objects. In single agent mode there will only be a single
            "default_policy".
        :param episode: Episode object which contains episode
            state. You can use the `episode.user_data` dict to store
            temporary data, and `episode.custom_metrics` to store custom
            metrics for the episode.
            In case of environment failures, episode may also be an Exception
            that gets thrown from the environment before the episode finishes.
            Users of this callback may then handle these error cases properly
            with their custom logic.
        :param env_index: The index of the sub-environment that ended the
            episode (within the vector of sub-environments of the BaseEnv).
        :param kwargs: Forward compatibility placeholder.
        """
        agent_id = episode.get_agents()[0]
        problem_filename = os.path.splitext(
            # pylint: disable=protected-access
            os.path.basename(episode._last_infos[agent_id][PROBLEM_FILENAME])
        )[0]
        episode.custom_metrics[f"terminated/{problem_filename}"] = (
            1.0 if episode.is_terminated(agent_id) else 0.0
        )
