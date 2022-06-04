for i in {0..13}
do
  sbatch --job-name=inds_exps18_$i --export=ALL,DISEASE=$i inds_exps18.sh
done