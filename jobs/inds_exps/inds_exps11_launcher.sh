for i in {0..13}
do
  sbatch --job-name=inds_exps11_$i --export=ALL,DISEASE=$i inds_exps11.sh
done