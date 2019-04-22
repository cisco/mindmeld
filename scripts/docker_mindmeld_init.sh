#!/usr/bin/env bash
#
# Copyright (c) 2015 Cisco Systems, Inc. and others.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

echo
echo The MindMeld Docker Installer downloads a Docker image and runs it.
echo

echo Downloading Dockerfile and dependencies
echo

# curl -s https://mindmeld.com/docker/wb3.tar.gz | tar xzvf -
curl -s https://mindmeld.com/docker/mindmeld_workbench.tar.gz | tar xzvf -
cd mindmeld_workbench

echo
echo Building Docker image
./buildme.sh
echo done

echo 
echo Running Docker image
echo 
echo You will be dropped inside the container. You can run the \"mindmeld\" command and explore.
echo The \"projects\" folder in the container is linked to your native \"projects\" folder so any changes made in one are visible in the other. 
echo   So, you can use the tools and editors you\'re used to for your tasks.
echo
echo try: mindmeld blueprint food_ordering
echo See the quickstart guide: https://mindmeld.com/docs/quickstart/00_overview.html
echo
./runme.sh
