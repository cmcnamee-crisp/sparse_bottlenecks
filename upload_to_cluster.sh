#!/bin/bash

# upload_to_cluster.sh
# Script to upload files to the Kempner cluster using rsync

# Configuration
# Replace these with your actual details
USERNAME="cmcnamee"
LAB="ba_lab"
SOURCE_DIR="./" # Directory to upload
DEST_DIR="sparse_bottlenecks" # Destination folder name on the cluster

# The destination path on the cluster's high-performance scratch space
CLUSTER_DEST="/n/${LAB}/Everyone/${USERNAME}/${DEST_DIR}/"

echo "Uploading files from ${SOURCE_DIR} to ${USERNAME}@login.rc.fas.harvard.edu:${CLUSTER_DEST}..."

# Use rsync to upload files
# -a: archive mode (preserves permissions, times, etc.)
# -v: verbose
# -x: don't cross filesystem boundaries
# -z: compress file data during the transfer
# --progress: show progress during transfer
rsync -avxz --progress "${SOURCE_DIR}" "${USERNAME}@login.rc.fas.harvard.edu:${CLUSTER_DEST}"

echo "Upload complete!"
