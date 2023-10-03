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
Two Headed Model
=================
"""
from typing import Any, Dict, Sequence, Tuple

import gymnasium as gym
import torch
from ray.rllib.examples.parametric_actions_cartpole import (
    TorchParametricActionsModel,
)
from ray.rllib.utils.typing import ModelConfigDict


# pylint: disable=abstract-method
class TwoHeadedModel(TorchParametricActionsModel):
    """
    Two-headed model for selecting a pair of clauses.

    >>> import gymnasium as gym
    >>> from gym_saturation.wrappers import Md2DWrapper, LLMWrapper
    >>> max_clauses = 7
    >>> embedding_dim = 768
    >>> prover = Md2DWrapper(LLMWrapper(
    ...     gym.make("Vampair-v0", max_clauses=max_clauses),
    ...     features_num=embedding_dim
    ... ))
    >>> model = TwoHeadedModel(
    ...     obs_space=prover.observation_space,
    ...     action_space=prover.action_space,
    ...     num_outputs=max_clauses,
    ...     model_config={},
    ...     name="test",
    ...     true_obs_shape=(max_clauses, embedding_dim),
    ...     action_embed_size=256,
    ... )
    >>> batch_size = 2
    >>> result_shape = model(
    ...     {"obs": {"clause_embeddings":
    ...         torch.zeros((batch_size, max_clauses, embedding_dim))
    ...     }},
    ...     None, None
    ... )[0].detach().shape
    >>> result_shape[0] == batch_size
    True
    >>> result_shape[1] == max_clauses * max_clauses
    True
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        true_obs_shape: Sequence[int] = (4,),
        action_embed_size: int = 2,
        **kwargs
    ):
        """Initialise heads of the policy network."""
        super().__init__(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            true_obs_shape,
            action_embed_size,
            **kwargs
        )
        self.first_clause_embedding = torch.nn.Sequential(
            torch.nn.Linear(action_embed_size, action_embed_size),
            torch.nn.ReLU(),
            torch.nn.Linear(action_embed_size, true_obs_shape[1]),
            torch.nn.ReLU(),
        )
        self.second_clause_embedding = torch.nn.Sequential(
            torch.nn.Linear(
                true_obs_shape[1] + action_embed_size, action_embed_size
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(action_embed_size, true_obs_shape[1]),
            torch.nn.ReLU(),
        )

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
        clause_embeddings = input_dict["obs"]["clause_embeddings"]
        action_embed, _ = self.action_embed_model({"obs": clause_embeddings})
        first_clause_embedding = self.first_clause_embedding(action_embed)
        second_clause_embedding = self.second_clause_embedding(
            torch.cat([action_embed, first_clause_embedding], dim=1)
        )
        first_clause_logits = torch.sum(
            clause_embeddings * torch.unsqueeze(first_clause_embedding, 1),
            dim=2,
        )
        second_clause_logits = torch.sum(
            clause_embeddings * torch.unsqueeze(second_clause_embedding, 1),
            dim=2,
        )
        return (
            torch.unsqueeze(first_clause_logits, 2)
            + torch.unsqueeze(second_clause_logits, 1)
        ).flatten(1, 2), state
