#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get the project root directory assuming the script is located in the 'scripts' folder
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RAW_DATA_DIR="$PROJECT_ROOT/data/raw-mafaulda"
PROCESSED_DATA_DIR="$PROJECT_ROOT/data/processed-mafaulda"

# Set up the required destination directories
mkdir -p "$RAW_DATA_DIR"
mkdir -p "$PROCESSED_DATA_DIR"

BASE_URL="https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda"

DOWNLOAD_ALL=false
UNZIP=false
TARGETS=()

# Parse the arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -all) DOWNLOAD_ALL=true ;;
        -unzip) UNZIP=true ;;
        -normal) TARGETS+=("normal") ;;
        -horizontal) TARGETS+=("horizontal-misalignment") ;;
        -vertical) TARGETS+=("vertical-misalignment") ;;
        -imbalance) TARGETS+=("imbalance") ;;
        -underhang) TARGETS+=("underhang") ;;
        -overhang) TARGETS+=("overhang") ;;
        *) echo "Unknown parameter passed: $1. Supported flags: -all, -unzip, -normal, -horizontal, -vertical, -imbalance, -underhang, -overhang"; exit 1 ;;
    esac
    shift
done

if [ "$DOWNLOAD_ALL" = true ]; then
    TARGETS=("normal" "horizontal-misalignment" "vertical-misalignment" "imbalance" "underhang" "overhang")
fi

if [ ${#TARGETS[@]} -eq 0 ]; then
    echo "Please specify at least one subset to download."
    echo "Usage: $0 [-all] [-unzip] [-normal] [-horizontal] [-vertical] [-imbalance] [-underhang] [-overhang]"
    exit 1
fi

echo "Dataset will be saved to: $RAW_DATA_DIR"

for target in "${TARGETS[@]}"; do
    FILE_URL="${BASE_URL}/${target}.zip"
    DEST_FILE="${RAW_DATA_DIR}/${target}.zip"
    
    echo "----------------------------------------"
    echo "Downloading ${target}.zip..."
    wget --show-progress -q -O "$DEST_FILE" "$FILE_URL" || curl -L --progress-bar -o "$DEST_FILE" "$FILE_URL"
    
    if [ "$UNZIP" = true ]; then
        echo "Extracting ${target}.zip..."
        unzip -q -o "$DEST_FILE" -d "$RAW_DATA_DIR"
    fi
done

echo "----------------------------------------"
echo "Process completed successfully!"
