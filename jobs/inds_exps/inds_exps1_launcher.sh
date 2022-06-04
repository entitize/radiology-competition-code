for i in {0..13}
do
  sbatch --job-name=inds_exps1_$i --export=ALL,DISEASE=$i inds_exps1.sh
done