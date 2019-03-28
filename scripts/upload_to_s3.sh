#!/bin/bash
aws s3 sync ./build/html s3://mindmeld-docs/workbench/${CIRCLE_BRANCH}/
aws s3 cp scripts/mmworkbench_init.sh s3://mindmeld-docs/workbench/${CIRCLE_BRANCH}/scripts/mmworkbench_init.sh
aws s3 cp scripts/docker_mmworkbench_init.sh s3://mindmeld-docs/workbench/${CIRCLE_BRANCH}/scripts/docker_mmworkbench_init.sh
