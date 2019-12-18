#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( dirname $SCRIPT_DIR )"

pushd $REPO_DIR

pip install --upgrade pip
pip install .
pip install -r test-requirements.txt
if [ "$MM_EXTRAS" == "true" ]; then
  echo "Installing extra requirements"
  pip install -r extras-requirements.txt
  echo "Finished installing extra requirements"
fi

popd
