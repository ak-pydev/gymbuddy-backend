#!/bin/bash
# Helper script to unzip Kinetics 400 skeleton files
# Usage: ./scripts/unzip_kinetics.sh

ZIP_PATH="data/raw/skeleton/kinetics400/kpfiles/OpenMMLab___Kinetics400-skeleton/raw/k400_kpfiles_2d.zip"
DEST_DIR="data/raw/skeleton/kinetics400/kpfiles"

if [ ! -f "$ZIP_PATH" ]; then
    echo "Zip file not found at $ZIP_PATH"
    exit 1
fi

echo "Unzipping $ZIP_PATH to $DEST_DIR..."
unzip -q -j "$ZIP_PATH" -d "$DEST_DIR"

echo "Done."
