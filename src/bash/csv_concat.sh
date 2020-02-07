#!/bin/bash

#inputs csv_dir out_dir file_suffix

CSV_DIR=$1
OUT_DIR=$2
SUFFIX=${3:-"checkins"}


# set flags
set -e
set -u

# change IFS
SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

# get files
FILES=$CSV_DIR*.csv
 
# iterate through files and concat files with same prefix (3 letters)
for FILE_PATH in $FILES
do
    IFS='/'
    ARR=($FILE_PATH)
    unset IFS

    PREFIX=`echo ${ARR[-1]} | cut -c1-3`
    OUTFILENAME=${PREFIX,,}_${SUFFIX}.csv

    if [ ! -d $OUT_DIR ]; then
        mkdir $OUT_DIR
    fi

    if [ ! -f "${OUT_DIR}/${OUTFILENAME}" ]; then
        head -n 1 "${FILE_PATH}" | cat >> $OUT_DIR/$OUTFILENAME
    fi

    tail -n +2 "${FILE_PATH}" | cat >> $OUT_DIR/$OUTFILENAME
done