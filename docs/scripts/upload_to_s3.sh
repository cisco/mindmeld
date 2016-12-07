#!/bin/bash
tar czvf workbench-docs.tgz docs/
sha=`echo ${CIRCLE_SHA1} | cut -c1-6`
target=${CIRCLE_BRANCH}/workbench-docs.tar.gz
bucket=mindmeld-docs
aws s3 cp workbench-docs.tgz s3://${bucket}/${target}
