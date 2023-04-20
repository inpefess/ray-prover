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
Problem filename wrapper
=========================
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import Wrapper

from ray_prover.constants import PROBLEM_FILENAME


class ProblemFilenameWrapper(Wrapper):
    """Add problem filename to info."""

    # pylint: disable=arguments-differ
    def reset(  # type: ignore
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment.

        :param seed: seed for compatibility
        :param options: options for compatibility
        :returns: observation and changed info
        """
        observation, info = super().reset(seed=seed, options=options)
        info[PROBLEM_FILENAME] = self.get_task()
        return observation, info

    # pylint: disable=arguments-differ
    def step(
        self, action: np.int64
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Step.

        :param action: action
        :returns: usual ``step`` returned value with changed info
        """
        observation, reward, terminated, truncated, info = super().step(action)
        info[PROBLEM_FILENAME] = self.get_task()
        return observation, reward, terminated, truncated, info
