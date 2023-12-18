#!/bin/bash

set -e

pip install -e rolf
pip install -e r3m
pip install -e vip
pip install -r implicit_q_learning/requirements.txt

download_file() {
  local file_id=$1
  local dest_dir=$2
  local file_name=$3

  # Create destination directory if it does not exist
  mkdir -p $dest_dir

  # Use curl to download from Google Drive
  curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
  curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${file_id}" -o ${dest_dir}/${file_name}

  # Clean up cookie
  rm ./cookie
}

# Download r3m ResNet50 model.
# With Docker container, use the following line instead
# download_file "1Xu0ssuG0N1zjZS54wmWzJ7-nb0-7XzbA" "/root/.r3m/r3m_50" "model.pt"
download_file "1Xu0ssuG0N1zjZS54wmWzJ7-nb0-7XzbA" "$HOME/.r3m/r3m_50" "model.pt"

# Download r3m ResNet50 config
# With Docker container, use the following line instead
# download_file "10jY2VxrrhfOdNPmsFdES568hjjIoBJx8" "/root/.r3m/r3m_50" "config.yaml"
download_file "10jY2VxrrhfOdNPmsFdES568hjjIoBJx8" "$HOME/.r3m/r3m_50" "config.yaml"
