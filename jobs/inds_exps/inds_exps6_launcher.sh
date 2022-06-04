for i in {0..13}
do
  sbatch --job-name=inds_exps6_$i --export=ALL,DISEASE=$i inds_exps6.sh
done