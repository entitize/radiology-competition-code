for i in {0..13}
do
  sbatch --job-name=inds_exps17_$i --export=ALL,DISEASE=$i inds_exps17.sh
done