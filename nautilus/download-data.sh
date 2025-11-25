#!/bin/bash

# 1. Ensure we are in the right directory
mkdir -p /mnt/project/data
cd /mnt/project/data

# 2. Create a list of URLs
cat <<EOF > urls.txt
https://zenodo.org/records/13905264/files/grid_g005_high_v1.zip?download=1
https://zenodo.org/records/13905264/files/grid_g005_low_v1.zip?download=1
https://zenodo.org/records/13905264/files/grid_g005_mid_v1.zip?download=1
https://zenodo.org/records/13905264/files/grid_g005_mid_v2.zip?download=1
https://zenodo.org/records/13905264/files/grid_g05_high_v1.zip?download=1
https://zenodo.org/records/13905264/files/grid_g05_low_v1.zip?download=1
https://zenodo.org/records/13905264/files/grid_g05_mid_v1.zip?download=1
https://zenodo.org/records/13905264/files/grid_g05_mid_v2.zip?download=1
https://zenodo.org/records/13905264/files/terrain_g005_high_v1.zip?download=1
https://zenodo.org/records/13905264/files/terrain_g005_low_v1.zip?download=1
https://zenodo.org/records/13905264/files/terrain_g005_mid_v1.zip?download=1
https://zenodo.org/records/13905264/files/terrain_g05_high_v1.zip?download=1
https://zenodo.org/records/13905264/files/terrain_g05_low_v1.zip?download=1
https://zenodo.org/records/13905264/files/terrain_g05_mid_v1.zip?download=1
https://zenodo.org/records/13905264/files/terrain_g1_high_v1.zip?download=1
https://zenodo.org/records/13905264/files/terrain_g1_low_v1.zip?download=1
https://zenodo.org/records/13905264/files/terrain_g1_mid_v1.zip?download=1
EOF

echo "Starting parallel download (4 files at a time)..."

# 3. Run 4 downloads in parallel
# -n 1: Use 1 URL per command
# -P 4: Run 4 processes at a time
# --content-disposition: Tells wget to use the filename sent by the server (fixing the ?download=1 name issue)
cat urls.txt | xargs -n 1 -P 4 wget -q --show-progress --content-disposition

# Force write to disk to clear RAM
sync

echo "Download complete. Unzipping..."

# 4. Unzip sequentially (safest for memory)
for zip in *.zip; do
    dirname="${zip%.zip}"
    mkdir -p "$dirname"
    unzip -q -o "$zip" -d "$dirname" && rm "$zip"
    echo "Unzipped $dirname"
done

echo "All done!"