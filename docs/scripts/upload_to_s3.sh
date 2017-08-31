#!/bin/bash
aws s3 sync docs/build/html s3://mindmeld-docs-${CIRCLE_BRANCH}/
aws s3 cp docs/scripts/mmworkbench_init.sh s3://mindmeld-docs-${CIRCLE_BRANCH}/scripts/mmworkbench_init.sh
aws s3 cp docs/scripts/docker_mmworkbench_init.sh s3://mindmeld-docs-${CIRCLE_BRANCH}/scripts/docker_mmworkbench_init.sh
