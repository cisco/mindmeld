#!/bin/bash

echo "Processing source text files."

pushd "$( dirname "${BASH_SOURCE[0]}" )"
for file in `ls ../build/text/*.txt`
do
	sed -e 's/+//g' -e 's/=//g' -e 's/--//g' -e 's/\*\**//g' -e 's/|//g' $file > ${file%.txt}.rst.txt
done

cp -r ../build/text/*.rst.txt ../build/html/_sources
cp search.html ../build/html/

popd
echo "Processing done."
