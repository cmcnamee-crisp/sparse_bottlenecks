#!/bin/bash

# upload_to_cluster.sh 
# Uploads the unit_tests code to the cluster and submits the job.

USERNAME="cmcnamee"
LAB="ba_lab"
SOURCE_DIR="./"
DEST_DIR="sparse_bottlenecks/unit_tests"

CLUSTER_DEST="/n/${LAB}/Everyone/${USERNAME}/${DEST_DIR}/"

echo "Step 1: Uploading code to cluster..."
# Ensure destination structure exists
# Rsync the directory, excluding heavy data/results folders and PDFs/zips
rsync -avxz --progress \
    --exclude="data/" \
    --exclude="results/" \
    --exclude="venv_test/" \
    --exclude="checkpoints/" \
    --exclude="times/" \
    --exclude="*.pdf" \
    --exclude="*.zip" \
    "${SOURCE_DIR}" "${USERNAME}@login.rc.fas.harvard.edu:${CLUSTER_DEST}"


echo "Upload complete!"
