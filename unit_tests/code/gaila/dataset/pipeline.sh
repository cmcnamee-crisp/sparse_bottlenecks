DATE="final"
python3 job.py --experiment dataset/dataset_t1.json --date $DATE
python3 job.py --experiment dataset/dataset_t1_colors.json --date $DATE
python3 job.py --experiment dataset/dataset.json --date $DATE
python3 job.py --experiment dataset/dataset_colors.json --date $DATE

python3 job.py --experiment dataset/dataset_transform.json --date $DATE

# https://hpc.nih.gov/docs/job_dependencies.html
sbatch jobs/dataset_t1-$DATE.sh
sbatch jobs/dataset_t1_colors-$DATE.sh
sbatch jobs/dataset-$DATE.sh
almostjid1=$(sbatch jobs/dataset_colors-$DATE.sh)
arr=($almostjid1)
jid1=${arr[3]}

sbatch --dependency=afterok:$jid1 jobs/dataset_transform_t1-$DATE.sh
sbatch --dependency=afterok:$jid1 jobs/dataset_transform_t1_colors-$DATE.sh
sbatch --dependency=afterok:$jid1 jobs/dataset_transform-$DATE.sh
sbatch --dependency=afterok:$jid1 jobs/dataset_transform_colors-$DATE.sh

sbatch  jobs/dataset_transform-$DATE.sh
