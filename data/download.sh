#!/bin/bash
# Download the Container Logistics OCEL 2.0 dataset from Zenodo
#
# The OCEL 2.0 version of the Logistics log:
#   DOI: 10.5281/zenodo.8428084
#   URL: https://zenodo.org/records/8428084
#
# NOTE: The older record 10.5281/zenodo.8289899 uses a pre-OCEL-2.0
# schema that pm4py cannot read. Use the 8428084 record instead.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="$SCRIPT_DIR/ocel2-logistics.sqlite"

if [ -f "$OUTPUT" ]; then
    echo "Dataset already exists at: $OUTPUT"
    echo "Delete it and re-run to re-download."
    exit 0
fi

# The OCEL 2.0 Container Logistics log (Zenodo record 8428084)
# The file inside the record is named "ContainerLogistics.sqlite"
URL="https://zenodo.org/records/8428084/files/ContainerLogistics.sqlite?download=1"

echo "Downloading Container Logistics OCEL 2.0 dataset..."
echo "Source: Zenodo record 8428084"
echo "URL: $URL"

if command -v wget &> /dev/null; then
    wget -q --show-progress -O "$OUTPUT" "$URL"
elif command -v curl &> /dev/null; then
    curl -L -o "$OUTPUT" "$URL"
else
    echo "Error: neither wget nor curl is available."
    exit 1
fi

echo "Saved to: $OUTPUT"
echo "Size: $(du -h "$OUTPUT" | cut -f1)"

# If the real file can't be downloaded, generate synthetic data instead
if [ ! -s "$OUTPUT" ]; then
    echo "Download failed or file is empty. Generating synthetic data..."
    rm -f "$OUTPUT"
    python3 "$SCRIPT_DIR/generate_synthetic.py"
fi