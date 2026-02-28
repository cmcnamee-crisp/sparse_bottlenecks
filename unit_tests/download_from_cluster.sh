#!/bin/bash

# download_from_cluster.sh
# Downloads the results of run_from_scratch.sh from the cluster to the local machine.

USERNAME="cmcnamee"
LAB="ba_lab"
SOURCE_DIR_CLUSTER="sparse_bottlenecks/unit_tests/code/gaila/results_sae/sae_experiments/"
LOCAL_DEST="./code/gaila/results_sae/sae_experiments/"

CLUSTER_SRC="/n/${LAB}/Everyone/${USERNAME}/${SOURCE_DIR_CLUSTER}"

echo "Step 1: Downloading results from cluster..."
mkdir -p "${LOCAL_DEST}"

rsync -avxz --progress "${USERNAME}@login.rc.fas.harvard.edu:${CLUSTER_SRC}" "${LOCAL_DEST}"

echo "Download complete! Results are in ${LOCAL_DEST}"
