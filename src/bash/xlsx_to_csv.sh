#!/bin/bash

#inputs: csv_dir output_dir

CSV_DIR=$1
OUT_DIR=$2

# set flags
set -e
set -u

# change IFS
SAVEIFS=$IFS
IFS=$(echo -en "\n\b")
# get files
XLSX_FILES=$CSV_DIR*.xlsx
for XLSX_FILE in $XLSX_FILES
do
libreoffice --headless --convert-to csv $XLSX_FILE --outdir $OUT_DIR
done

# restore $IFSs
IFS=$SAVEIFS