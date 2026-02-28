DATE="final"
python job.py --experiment experiments/pretrain.json --date $DATE

# `train` also runs t2.
python job.py --experiment experiments/train.json --date $DATE

python job.py --experiment experiments/t1.json --date $DATE
python job.py --experiment experiments/t3t4.json --date $DATE
python job.py --experiment experiments/layers.json --date $DATE

almostjid1=$(sbatch jobs/pretrain-$DATE.sh)
arr=($almostjid1)
jid1=${arr[3]}

# https://hpc.nih.gov/docs/job_dependencies.html
almostjid2=$(sbatch jobs/train-$DATE.sh --dependency=afterok:$jid1)
arr=($almostjid2)
jid2=${arr[3]}

sbatch jobs/t1-$DATE.sh --dependency=afterok:$jid2
sbatch jobs/t3t4-$DATE.sh --dependency=afterok:$jid2

sbatch jobs/layers-$DATE.sh --dependency=afterok:$jid2