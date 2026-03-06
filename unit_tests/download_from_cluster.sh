#!/bin/bash

# download_from_cluster.sh
# Downloads results and debug logs from the cluster using a single SSH connection.

USERNAME="cmcnamee"
CLUSTER="login.rc.fas.harvard.edu"
REMOTE_BASE="/n/ba_lab/Everyone/${USERNAME}/sparse_bottlenecks/unit_tests/code/gaila"
LOCAL_BASE="./code/gaila"

# SSH multiplexing: open one authenticated connection, reuse for all transfers.
SOCKET="/tmp/ssh-cluster-$$"

echo "=== Connecting to cluster (authenticate once) ==="
ssh -fNM -S "${SOCKET}" -o ControlPersist=10m "${USERNAME}@${CLUSTER}"

if [ $? -ne 0 ]; then
    echo "ERROR: SSH connection failed."
    exit 1
fi

echo "Connection established. Downloading..."

# Helper: rsync over the shared SSH connection
do_rsync() {
    local remote_path="$1"
    local local_path="$2"
    mkdir -p "${local_path}"
    rsync -avxz --progress -e "ssh -S ${SOCKET}" \
        "${USERNAME}@${CLUSTER}:${remote_path}" "${local_path}"
}

# 1. SAE evaluation results (eval.json + concept_graph.json)
echo ""
echo "--- [1/5] SAE results ---"
do_rsync "${REMOTE_BASE}/results_sae/sae_experiments/" "${LOCAL_BASE}/results_sae/sae_experiments/"

# 2. SLURM stdout logs
echo ""
echo "--- [2/5] SLURM stdout logs ---"
do_rsync "${REMOTE_BASE}/out/" "${LOCAL_BASE}/out/"

# 3. SLURM stderr logs
echo ""
echo "--- [3/5] SLURM stderr logs ---"
do_rsync "${REMOTE_BASE}/err/" "${LOCAL_BASE}/err/"

# 4. List of checkpoint files (don't transfer the large files, just list them)
echo ""
echo "--- [4/5] Listing checkpoint files on cluster ---"
echo "  Backbone checkpoints:"
ssh -S "${SOCKET}" "${USERNAME}@${CLUSTER}" \
    "ls -lh ${REMOTE_BASE}/checkpoints/*.ckpt 2>/dev/null | awk '{print \"    \", \$5, \$NF}'"
echo "  SAE checkpoints:"
ssh -S "${SOCKET}" "${USERNAME}@${CLUSTER}" \
    "ls -lh ${REMOTE_BASE}/sae_checkpoints/*.pt 2>/dev/null | awk '{print \"    \", \$5, \$NF}'"

# 5. List the sae_eval.py on cluster to verify it's the right version
echo ""
echo "--- [5/5] Checking sae_eval.py version on cluster ---"
ssh -S "${SOCKET}" "${USERNAME}@${CLUSTER}" \
    "head -5 ${REMOTE_BASE}/sae_eval.py; echo '...'; grep -n 'load_raw_cnn\|backbone' ${REMOTE_BASE}/sae_eval.py | head -10"

# Clean up SSH connection
echo ""
echo "=== Closing SSH connection ==="
ssh -S "${SOCKET}" -O exit "${USERNAME}@${CLUSTER}" 2>/dev/null

echo ""
echo "Download complete!"
echo "  Results:  ${LOCAL_BASE}/results_sae/sae_experiments/"
echo "  Stdout:   ${LOCAL_BASE}/out/"
echo "  Stderr:   ${LOCAL_BASE}/err/"
