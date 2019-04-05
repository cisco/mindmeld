#!/bin/bash
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

echo "Processing source text files."

pushd "$( dirname "${BASH_SOURCE[0]}" )"
for file in `ls ../build/text/*.txt`
do
	sed -e 's/+//g' -e 's/=//g' -e 's/--//g' -e 's/\*\**//g' -e 's/|//g' $file > ${file%.txt}.rst.txt
done

cp -r ../build/text/*.rst.txt ../build/html/_sources

popd
echo "Processing done."
