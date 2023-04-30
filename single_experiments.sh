#!/bin/bash

for part in 01 03 04 06 02 09 08 05 11 07 10
do python ray_prover/ppo_prover.py --prover Vampire --max_clause 500 --problem_filename ~/data/TPTP-v8.1.2/Problems/SET/SET0$part-1.p
done
