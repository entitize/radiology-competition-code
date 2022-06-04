for i in {0..13}
do
  sbatch --job-name=inds_exps5_$i --export=ALL,DISEASE=$i inds_exps5.sh
done