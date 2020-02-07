#!/bin/bash

# inputs: sql_path, table_name, csv_path

set -e
set -u

if [ $# != 3 ]; then
    echo "please enter the path to a SQL file, new table name and path to a CSV file"
    exit 1
fi

export SQL_PATH=$1
export TABLE_NAME=$2
export CSV_PATH=$3

psql -U postgres \
    -f $SQL_PATH \
    --set AUTOCOMMIT=off \
    --set ON_ERROR_STOP=on \
    --set CSV_PATH=\'$CSV_PATH\' \
    --set TABLE_NAME=$TABLE_NAME \
    climbing_gym

exit 0
