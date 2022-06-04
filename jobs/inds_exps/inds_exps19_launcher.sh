for i in {0..13}
do
  sbatch --job-name=inds_exps19_$i --export=ALL,DISEASE=$i inds_exps19.sh
done