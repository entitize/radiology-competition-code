for i in {0..13}
do
  sbatch --job-name=inds_exps8_$i --export=ALL,DISEASE=$i inds_exps8.sh
done