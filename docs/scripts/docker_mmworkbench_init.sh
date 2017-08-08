#!/usr/bin/env bash

set -e

echo
echo The MindMeld Workbench Docker Installer downloads a Docker image and runs it.
echo

echo Downloading Dockerfile and dependencies
echo

curl -s https://mindmeld.com/docker/wb3.tar.gz | tar xzvf -
# curl -s https://mindmeld.com/docker/mindmeld_workbench.tar.gz | tar xzvf -
cd wb3

echo
echo Building Docker image
./buildme.sh
echo done

echo 
echo Running Docker image
echo 
echo You will be dropped inside the container. You can run the \"mmworkbench\" command and explore.
echo The \"projects\" folder in the container is linked to your native \"projects\" folder so any changes made in one are visible in the other. 
echo   So, you can use the tools and editors you\'re used to for your tasks.
echo
echo try: mmworkbench blueprint food_ordering
echo See the quickstart guide: https://mindmeld.com/docs/quickstart/00_overview.html
echo
./runme.sh
