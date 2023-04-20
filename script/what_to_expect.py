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
Script for problem evaluation
==============================
"""
import os

import gymnasium
from gym_saturation.agent_testing import AgeWeightAgent, episode
from gym_saturation.envs.saturation_env import SaturationEnv
from gym_saturation.utils import reduce_to_proof


def process_problem(problem: str) -> None:
    """
    Process problem.

    :param problem: problem number
    """
    env: SaturationEnv = gymnasium.make(
        "Vampire-v0", max_clauses=15000
    )  # type: ignore
    env.set_task(
        os.path.join(
            os.environ["HOME"],
            "data",
            "TPTP-v8.1.2",
            "Problems",
            "SET",
            f"SET0{problem}-1.p",
        )
    )
    _, _, steps = episode(env, AgeWeightAgent(1, 5))
    proof = {
        clause["label"] for clause in reduce_to_proof(tuple(env.state.clauses))
    }
    clauses_in_proof = [
        len(clause["literals"])
        for clause in env.state.clauses
        if set(clause["inference_parents"]).difference(proof) == set()
    ]
    print(
        os.path.basename(env.get_task())[:-2],
        "&",
        steps + 1,
        "&",
        len(proof),
        "&",
        len(env.state.clauses),
        "&",
        len(clauses_in_proof),
        "&",
        sum(len(clause["literals"]) for clause in env.state.clauses),
        "&",
        sum(clauses_in_proof),
        "\\\\",
    )
    print("\\hline")


if __name__ == "__main__":
    for problem_code in [f"0{i}" for i in range(1, 10)] + ["10", "11"]:
        process_problem(problem_code)
