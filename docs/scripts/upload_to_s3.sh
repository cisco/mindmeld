#!/bin/bash
aws s3 sync docs/build/html s3://mindmeld-docs-${CIRCLE_BRANCH}/
