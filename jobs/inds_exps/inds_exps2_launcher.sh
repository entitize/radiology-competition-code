for i in {0..13}
do
  sbatch --job-name=inds_exps2_$i --export=ALL,DISEASE=$i inds_exps2.sh
done