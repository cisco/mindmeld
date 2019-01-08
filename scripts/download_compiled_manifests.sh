#!/bin/bash
set -e

function usage () {
  echo "Usage: "
  echo " $0 -d WORKBENCH_DIR"
  echo ""
  echo " -d     The workbench path."
  echo " -h|-?  Show this message and exit."
}

while getopts "h?vb:d:" opt; do
  case "$opt" in
    h|\?)
      usage
      exit 0
      ;;
    d)
      WORKBENCH_DIR="$OPTARG"
      ;;
  esac

done

if [ -z ${WORKBENCH_DIR+x} ]; then
  echo "Missing workbench directory path (-d)."
  echo ""
  usage
  exit 1
fi

aws s3 cp --recursive s3://mindmeld-blueprints-develop/workbench_binaries/ "$WORKBENCH_DIR/mmworkbench/resources/"
