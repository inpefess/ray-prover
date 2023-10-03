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
Custom stopper
===============
"""
from ray.tune.stopper import Stopper


class CustomStopper(Stopper):
    """Custom stopper for curriculum learning."""

    def __init__(self, last_task: str):
        """
        Set the last task.

        :param last_task: a task to monitor
        """
        self.last_task = last_task

    def __call__(self, trial_id, result) -> bool:
        """
        Check whether to stop.

        :returns: if the trial should be terminated given the result
        """
        custom_metric = f"{self.last_task}/terminated_mean"
        if custom_metric in result["custom_metrics"]:
            return result["custom_metrics"][custom_metric] >= 0.99
        return False

    def stop_all(self) -> bool:
        """
        Define for compatibility.

        :returns: always ``False`` for compatibility
        """
        return False
