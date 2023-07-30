#!/usr/bin/env bash

# This script will be run at dataset directory, which is guaranteed to be empty.
# And the asset names will be given as an environment variable `ASSET_NAMES`.
# This is a string delimited by colon.
# The number of months we should download is also given as `NUM_MONTHS`.

BASE_URL="https://data.binance.vision/data/spot/monthly/klines"

if [ -z "$ASSET_NAMES" ]; then
  echo "ERROR: no assets are given." >&2
  echo "[help] please define the environment array variable \`ASSET_NAMES\`." >&2
  exit
fi

mkdir raw
cd raw || exit

END_MONTH=$(date +%Y-%m-%d)
START_MONTH=$(date -d "$END_MONTH -$NUM_MONTHS month" +%Y-%m-%d)

# ref: https://unix.stackexchange.com/questions/276614/bash-while-loop-read-from-colon-delimited-list-of-paths-using-ifs

i=0
IFS=:; set -o noglob
for ASSET in $ASSET_NAMES; do
  mkdir "$ASSET"
  mkdir "../$i"

  CURRENT_MONTH="$START_MONTH"
  j=0
  while [[ $CURRENT_MONTH < $END_MONTH ]]; do
    FILE_NAME="$ASSET-1m-${CURRENT_MONTH%-*}"
    URL="$BASE_URL/$ASSET/1m/$FILE_NAME.zip"

    wget -O "./$ASSET/$FILE_NAME.zip" "$URL"
    unzip "./$ASSET/$FILE_NAME.zip" -d "../$i/"
    mv "../$i/$FILE_NAME.csv" "../$i/$j.csv"

    CURRENT_MONTH=$(date -d "$CURRENT_MONTH + 1 month" +%Y-%m-%d)
    j=$((j + 1))
  done

  i=$((i + 1))
done
