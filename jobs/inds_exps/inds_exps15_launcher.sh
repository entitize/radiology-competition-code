for i in {0..13}
do
  sbatch --job-name=inds_exps15_$i --export=ALL,DISEASE=$i inds_exps15.sh
done