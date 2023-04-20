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
Common constants
=================
"""
import os

PROBLEM_FILENAME = "problem_filename"
SET_PROBLEMS = [
    os.path.join(
        os.environ["WORK"],
        "data",
        "TPTP-v8.1.2",
        "Problems",
        "SET",
        f"SET0{num}-1.p",
    )
    for num in [f"0{i}" for i in range(1, 10)] + ["10", "11"]
]
