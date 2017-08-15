#!/bin/bash
set -e

function usage () {
  echo "Usage: "
  echo " $0 -b BLUEPRINT_NAME -d BLUEPRINT_DIR"
  echo ""
  echo " -b     The name of the blueprint to upload."
  echo " -d     The directory of the blueprint."
  echo " -h|-?  Show this message and exit."
}

while getopts "h?vb:d:" opt; do
  case "$opt" in
    h|\?)
      usage
      exit 0
      ;;
    b)
      BLUEPRINT="$OPTARG"
      ;;
    d)
      BLUEPRINT_DIR="$OPTARG"
      ;;
  esac

done

if [ -z ${BLUEPRINT+x} ]; then
  echo "Missing blueprint name (-b)."
  echo ""
  usage
  exit 1
fi

if [ -z ${BLUEPRINT_DIR+x} ]; then
  echo "Missing blueprint directory (-d)."
  echo ""
  usage
  exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pushd "$SCRIPT_DIR/.." > /dev/null

# Delete and Recreate archive dir
ARCHIVE_DIR=`pwd`/tmp/$BLUEPRINT

if [ -d "$ARCHIVE_DIR" ]; then
  rm -rf $ARCHIVE_DIR
fi

mkdir -p $ARCHIVE_DIR

# Create and upload tarball
pushd "$BLUEPRINT_DIR" > /dev/null

tar -czf "$ARCHIVE_DIR/app.tar.gz" --exclude=data --exclude=indexes --exclude=.generated --exclude=*.pyc --exclude=__pycache__ --exclude=**/*.pyc *
aws s3 cp "$ARCHIVE_DIR/app.tar.gz" "s3://mindmeld-blueprints-${MM_BRANCH}/$BLUEPRINT/"

popd > /dev/null
popd > /dev/null
