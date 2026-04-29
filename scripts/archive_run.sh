#!/bin/bash

# Create a timestamped archive folder
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_DIR="archives/run_${TIMESTAMP}"

echo "====================================================="
echo "Archiving current run data to ${ARCHIVE_DIR}..."
echo "====================================================="

mkdir -p "${ARCHIVE_DIR}/data"
mkdir -p "${ARCHIVE_DIR}/results"

# Move knowledge JSON files if they exist
if ls data/knowledge_*.json 1> /dev/null 2>&1; then
    mv data/knowledge_*.json "${ARCHIVE_DIR}/data/"
    echo "[OK] Moved knowledge JSON files."
else
    echo "[INFO] No knowledge JSON files found to move."
fi

# Move log text files if they exist
if ls results/log_*.txt 1> /dev/null 2>&1; then
    mv results/log_*.txt "${ARCHIVE_DIR}/results/"
    echo "[OK] Moved result log files."
else
    echo "[INFO] No result log files found to move."
fi

echo "====================================================="
echo "Archive complete!"
echo "====================================================="
