#!/usr/bin/env bash

set -e

read -p "Enter mindmeld.com username: " USERNAME
echo -n "Enter mindmeld.com password: " 
read -s PASSWORD
echo

###
# .mmworkbench/config
###

mkdir -p ~/.mmworkbench
cat >~/.mmworkbench/config <<EOL
[mmworkbench]
mindmeld_url = https://mindmeld.com
username = $USERNAME
password = $PASSWORD
EOL

echo ~/.mmworkbench/config created.

####
# pip.conf
###

# create folder if not exists
mkdir -p ~/.pip

# create file if not exists
touch ~/.pip/pip.conf

# this will wipe out your existing pip.conf
cat >~/.pip/pip.conf <<EOL
[global]
extra-index-url = https://$USERNAME:$PASSWORD@mindmeld.com/pypi
trusted-host = mindmeld.com
EOL

echo ~/.pip/pip.conf created.

