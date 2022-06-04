for i in {0..13}
do
  sbatch --job-name=inds_exps13_$i --export=ALL,DISEASE=$i inds_exps13.sh
done