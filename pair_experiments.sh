#!/bin/bash

PARTS=("01 03 04 06 02 09 08")
PREFIX=~/data/TPTP-v8.1.2/Problems/SET/SET0

for PART_ONE in $PARTS
do
    for PART_TWO in $PARTS
    do
	if [[ "$PART_ONE" != "$PART_TWO" ]]
	then
	    python ray_prover/ppo_prover.py --prover Vampire --max_clause 500 --problem_filename  $PREFIX$PART_ONE-1.p --second_problem_filename $PREFIX$PART_TWO-1.p
	fi
    done
done
