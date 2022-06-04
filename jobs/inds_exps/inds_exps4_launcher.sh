for i in {0..13}
do
  sbatch --job-name=inds_exps4_$i --export=ALL,DISEASE=$i inds_exps4.sh
done