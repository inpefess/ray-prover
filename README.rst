..
  Copyright 2023 Boris Shminke

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

|CircleCI|\ |Documentation Status|\ |codecov|\

***********
ray-prover
***********

Generic saturation prover using Ray RLlib.

How to install
***************

``ray-prover`` works only with Python 3.8, 3.9 or 3.10

.. code:: sh

   pip install git+https://github.com/inpefess/ray-prover.git

How to use
***********

There are two scripts: for PPO and Thompson sampling.

``--prover`` can be ``Vampire`` or ``iProver``

Add ``--random_baseline`` for not learning anything.

.. code:: sh
	  
   python ray_prover/thompson_sampling.py --prover Vampire --max_clause 20 --problem_filename ~/data/TPTP-v8.1.2/Problems/SET/SET001-1.p --random_baseline
   python ray_prover/ppo_prover.py --prover iProver --max_clause 15 --problem_filename ~/data/TPTP-v8.1.2/Problems/SET/SET001-1.p

.. |CircleCI| image:: https://circleci.com/gh/inpefess/ray-prover.svg?style=svg
   :target: https://circleci.com/gh/inpefess/ray-prover
.. |Documentation Status| image:: https://readthedocs.org/projects/ray-prover/badge/?version=latest
   :target: https://ray-prover.readthedocs.io/en/latest/?badge=latest
.. |codecov| image:: https://codecov.io/gh/inpefess/ray-prover/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/inpefess/ray-prover
