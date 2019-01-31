#!/usr/bin/env bash

# needed for pyenv install fix
# xcode-select --install 2&>/dev/null

set -e

NEEDS_DEP_INSTALL=0
NEEDS_JAVA=0
NEEDS_VIRTUALENV=0

function check_macos() {
	platform=$(uname)
	if [[ ! $platform == "Darwin" ]]; then
		echo This Installer is for MacOS only. Quitting.
		exit 1
	fi
}

function check_dependency {
	local command=$1
	echo -n "   " $command "... "

	# output yes or no
	if [[ `which $command` ]]; then
		if [[ $command == "java" ]]; then
			version=$(java -version 2>&1 | head -1 | awk '{print $3}' | sed "s/\"//g")
			if [[ $version == 1.8* ]]; then
				echo yes
			else
				echo older version $version found. 1.8+ needed.
				NEEDS_JAVA=1
				NEEDS_DEP_INSTALL=1
			fi
		elif [[ $command == "virtualenv" ]]; then
			if [[ `$command --version 2> /dev/null` ]]; then
				echo yes
			else
				echo no
				NEEDS_VIRTUALENV=1
				NEEDS_DEP_INSTALL=1
			fi
		else
			echo yes
		fi
	else
		echo no
		NEEDS_DEP_INSTALL=1
	fi
}

function install_dependency {
	local command=$1

	if [[ ! `which $command` ]]; then
		echo "   " $command
		if [[ $command == "brew" ]]; then
			/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
		elif [[ $command == "pip" ]]; then
			sudo -H easy_install pip
		elif [[ $command == "java" ]]; then
			brew tap caskroom/cask
			brew cask install java
	    elif [[ $command == "elasticsearch" ]]; then
	    	brew install elasticsearch
			brew services start elasticsearch
	    elif [[ $command == "virtualenv" ]]; then
	    	sudo -H pip install --upgrade virtualenv
	    else
			brew install $command
		fi
	elif [[ ($command == "java") && (${NEEDS_JAVA} == 1) ]]; then
		echo "   " $command ...
		brew tap caskroom/cask
		brew cask install java
    elif [[ ($command == "virtualenv") && (${NEEDS_VIRTUALENV} == 1) ]]; then
		sudo -H pip install --upgrade virtualenv
	fi
}

function check_service {
	local command=$1
	echo -n "   " $command "... "

	if [[ ($command == "elasticsearch") && (! `curl -s http://localhost:9200/`) ]]; then
		echo no
		echo "   " starting elasticsearch ...
		brew services restart elasticsearch
	else
		echo yes
	fi
}

# Gather Info
echo
echo The MindMeld Workbench Installer checks dependencies and installs them.
echo

check_macos

echo Checking dependencies already installed

check_dependency brew
check_dependency python
check_dependency pip
check_dependency virtualenv
check_dependency java
check_dependency elasticsearch
echo done

if [[ ${NEEDS_DEP_INSTALL} == 1 ]]; then
	echo
	read -p "Do you want to install the missing dependencies (Y/n): " RESPONSE
	# lowercase
	RESPONSE=$(echo "$RESPONSE" | tr '[:upper:]' '[:lower:]')
	if [[ (! $RESPONSE == "") && (! $RESPONSE == "y") && (! $RESPONSE == "yes") ]]; then
		echo exiting
		exit 1
	fi

	# Install stuff
	echo
	echo Installing missing dependencies. You may be asked for sudo permissions.

	install_dependency brew
	install_dependency python
	install_dependency pip
	install_dependency virtualenv
	install_dependency java
	install_dependency elasticsearch

	echo done
fi

echo
echo Checking dependent services already started
check_service elasticsearch
echo done

###
# .mmworkbench/config
###

mkdir -p ~/.mmworkbench
cat >~/.mmworkbench/config <<EOL
[mmworkbench]
mindmeld_url = https://devcenter.mindmeld.com
token = token
EOL

echo ~/.mmworkbench/config created.

####
# pip.conf
###

# create folder if not exists
mkdir -p ~/.pip

# if file already exists, make a backup
if test -f ~/.pip/pip.conf; then
	cp ~/.pip/pip.conf ~/.pip/pip.conf.backup
fi

# create file if not exists
touch ~/.pip/pip.conf

# this will wipe out your existing pip.conf
cat >~/.pip/pip.conf <<EOL
[global]
extra-index-url = https://engci-maven.cisco.com/artifactory/api/pypi/mm_workbench-pypi-group/simple
EOL

echo ~/.pip/pip.conf created.
echo
