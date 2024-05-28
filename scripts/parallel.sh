#!/bin/bash
for i in $(seq $@)
do
 sbatch cluster_several_batch_scenario_looped.sh $i
done
