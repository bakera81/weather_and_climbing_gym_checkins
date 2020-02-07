#!/bin/bash

# inputs: sql_path, gym_prefix

set -e
set -u

if [ $# != 3 ]; then
    echo "please enter the path to a SQL file and the gym prefix"
    exit 1
fi

export SQL_PATH=$1
export PREFIX=$2
export CSV_PATH=$3

psql -U postgres \
    -f $SQL_PATH \
    --set AUTOCOMMIT=off \
    --set ON_ERROR_STOP=on \
    --set CSV_PATH=\'$CSV_PATH\' \
    --set PREFIX=$PREFIX \
    climbing_gym

exit 0