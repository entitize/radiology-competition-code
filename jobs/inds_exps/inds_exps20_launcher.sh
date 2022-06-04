for i in {0..13}
do
  sbatch --job-name=inds_exps20_$i --export=ALL,DISEASE=$i inds_exps20.sh
done