"""Generates jobs/*.sh files from experiments/*.json.

The experiment_*.json define which hp options to run; the output *.sh
file is used with `sbatch` (slurm).
"""
import argparse
import datetime
import itertools
import json
import os


def _filter_options(python_script, options):
    # NOTE: This function is manually set! This removes some complexity
    # from the config files.
    return not _okay_options(python_script, options)


def _okay_options(python_script, options):
    if python_script in {"t1", "t2", "t3", "t4", "ft"}:
        if options["pretrain_model"] == "default":
            return options["finetune_model"] in {"cnn", "RN50NOPRE"}
        else:
            return options["finetune_model"] in {"linear"}
    else:
        # no filtering options set for the dataset jobs
        return True


def main(FLAGS):
    experiment = FLAGS.experiment
    experiment = experiment.replace(".json", "")
    with open(f"./{experiment}.json", "r") as f:
        settings = json.load(f)

    experiment = experiment.split("/")[-1]
    if FLAGS.date is None:
        jobname = datetime.datetime.now().strftime(f"{experiment}-%Y-%m-%d")
    else:
        jobname = f"{experiment}-{FLAGS.date}"

    jobs = generate_jobs(experiment, jobname, settings)
    is_using_gpu = settings["use_gpu"]
    write_jobs(jobs, jobname, experiment, is_using_gpu)


def generate_jobs(experiment, jobname, settings):
    settings["settings"]["jobname"] = [jobname]
    options = list(itertools.product(*settings["settings"].values()))
    keys = list(settings["settings"].keys())
    jobs = []
    options = [o for o in options]

    python_script = settings["script"]
    idx = 0
    for option in options:
        zipped = list(zip(keys, list(option)))
        _options = {k: v for k, v in zipped}
        if _filter_options(python_script, _options):
            continue
        job_text = _template_option(python_script, _options)
        job = setup(job_text, idx, experiment)
        idx += 1
        jobs.append(job)
    return jobs


def write_jobs(jobs, jobname, experiment, is_using_gpu):
    jobs_file = _template_file(jobs, experiment, is_using_gpu)
    os.makedirs("jobs", exist_ok=True)
    os.makedirs(f"results/{jobname}", exist_ok=True)
    with open(f"./jobs/{jobname}.sh", "w") as f:
        f.write(jobs_file)


def _template_file(texts, experiment, is_using_gpu):
    text = "".join(texts)
    if is_using_gpu:
        using_gpu = "#SBATCH -p gpu-he --gres=gpu:1"
        mem = "32G"
    else:
        using_gpu = ""
        # dataset creation should be minimal.
        mem = "8G"
    # using_gpu = ""
    out = f"""#!/bin/sh

# Request half an hour of runtime:
#SBATCH --time=24:00:00

# Ask for the GPU partition and 1 GPU
# skipping this for now.
{using_gpu}

# Use more memory (8GB) and correct partition.
#SBATCH --mem={mem}

# Specify a job name:
#SBATCH -J {experiment}

# Specify an output file
#SBATCH -o ./out/%x-%a.out
#SBATCH -e ./err/%x-%a.out

#SBATCH -a 0-{len(texts) - 1}

# module load python/3.7.4 cuda/11.3.1 cudnn/8.2.0 gcc/10.2
# source gaila/bin/activate
module load python/3.7.4
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate gaila

mkdir -p ./tmp/
mkdir -p ./out/
mkdir -p ./err/
mkdir -p ./log/
mkdir -p ./models/
mkdir -p ./results/

{text}
"""
    return out


def setup(text, index, experiment):
    return f"""
if [ "$SLURM_ARRAY_TASK_ID" -eq {index} ];
then
{text}
exit_code=$?
if [[ $exit_code = 0 ]]; then
echo "{index}\t{text}" >> log/{experiment}_success.tsv
elif [[ $exit_code = 1 ]]; then
echo "{index}\t{text}" >> log/{experiment}_failed.tsv
fi
fi
"""


def _template_option(
    experiment, options,
):
    """Generates the template for an a call to train.
    """

    def _okay(k, v):
        if isinstance(v, bool) and not v:
            # safety: Forces new options to be added here
            # but is not a quiet failure.
            assert k in {"holdout_many", "do_inlp_rand", "layerwise", "colors"}
            # e.g. v is False:
            return False
        return True

    joined_options = "  ".join(
        [f"--{k} {v}" for k, v in options.items() if _okay(k, v)]
    )
    return f"""python {experiment}.py {joined_options}"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=str)
    parser.add_argument("--date", required=False, default=None)
    main(parser.parse_args())
