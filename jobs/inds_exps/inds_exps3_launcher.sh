for i in {0..13}
do
  sbatch --job-name=inds_exps3_$i --export=ALL,DISEASE=$i inds_exps3.sh
done