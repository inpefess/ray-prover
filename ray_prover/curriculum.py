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
Curriculum learning function
=============================
"""
from typing import Any, Dict

from gym_saturation.envs import SaturationEnv
from ray.rllib.env.env_context import EnvContext


# pylint: disable=unused-argument
def curriculum_fn(
    next_task: str,
    train_results: Dict[str, Any],
    task_settable_env: SaturationEnv,
    env_ctx: EnvContext,
) -> str:
    """
    Organise two-task curriculum.

    :param next_task: next task to take if the current one is solved
    :param train_results: training metrics
    :param task_settable_env: the environment
    :param env_ctx: left for compatibility
    :returns: a current task of next task
    """
    if train_results["episode_reward_mean"] >= 0.99:
        return next_task
    return task_settable_env.get_task()
