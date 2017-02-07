#!/bin/bash

echo "Processing source text files."

for file in `ls ../build/text/*.txt`
do
	sed -e 's/+//g' -e 's/=//g' -e 's/--//g' -e 's/\*\**//g' -e 's/|//g' $file > ${file%.txt}.rst.txt
done

cp -r ../build/text/*.rst.txt ../build/html/_sources

echo "Processing done."
