for i in {0..13}
do
  sbatch --job-name=inds_exps16_$i --export=ALL,DISEASE=$i inds_exps16.sh
done