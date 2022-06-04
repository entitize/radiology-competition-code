for i in {0..13}
do
  sbatch --job-name=inds_exps10_$i --export=ALL,DISEASE=$i inds_exps10.sh
done