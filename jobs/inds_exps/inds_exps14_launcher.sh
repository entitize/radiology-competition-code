for i in {0..13}
do
  sbatch --job-name=inds_exps14_$i --export=ALL,DISEASE=$i inds_exps14.sh
done