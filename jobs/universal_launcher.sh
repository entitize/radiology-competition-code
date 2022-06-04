#!/bin/bash
# Run `sh universal_launcher.sh FILENAME`
# to requeue a job, run `scontrol requeue JOB_ID`
echo ========[FILE: $1]======== |& tee -a job_history.out
date +%Y-%m-%d-%H:%M:%S |& tee -a job_history.out
for i in {0..13}
do
  echo "Launching: $1_$i" |& tee -a job_history.out
  sbatch --job-name=$1_$i --export=ALL,DISEASE=$i --output=$1_$i.out --requeue $1 |& tee -a job_history.out
done
echo ==================== |& tee -a job_history.out
