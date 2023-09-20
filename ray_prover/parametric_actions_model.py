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
Parametric Actions Model
========================
"""
from typing import Any, Dict, Tuple

import torch
from ray.rllib.examples.parametric_actions_cartpole import (
    TorchParametricActionsModel,
)


# pylint: disable=abstract-method
class ParametricActionsModel(TorchParametricActionsModel):
    """Parametric actions model without action mask."""

    def forward(
        self, input_dict: Dict[str, Any], state: Any, seq_lens: Any
    ) -> Tuple[torch.Tensor, Any]:
        """
        Compute action logits.

        :param input_dict: a dictionary of observations
        :param state: kept for compatibility
        :param seq_lens: kept for compatibility
        :return: action logits and unmodified input ``state``
        """
        avail_actions = input_dict["obs"]["avail_actions"]
        action_embed, _ = self.action_embed_model({"obs": avail_actions})
        intent_vector = torch.unsqueeze(action_embed, 1)
        action_logits = torch.sum(avail_actions * intent_vector, dim=2)
        return action_logits, state
