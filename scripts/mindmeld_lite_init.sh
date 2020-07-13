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

# needed for pyenv install fix
# xcode-select --install 2&>/dev/null

set -e

NEEDS_DEP_INSTALL=0
NEEDS_VIRTUALENV=0

function check_os_and_execute_deps() {
	platform=$(uname)

	if [[ $platform == "Darwin" ]]; then

		function install_dependency {
			local command=$1

			if [[ ! `which $command` ]]; then
				echo "   " $command
				if [[ $command == "brew" ]]; then
					/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
				elif [[ $command == "pip" ]]; then
					sudo -H easy_install pip
			    elif [[ $command == "virtualenv" ]]; then
			    	sudo -H pip install --upgrade virtualenv
			    else
					brew install $command
				fi
		    elif [[ ($command == "virtualenv") && (${NEEDS_VIRTUALENV} == 1) ]]; then
				sudo -H pip install --upgrade virtualenv
			fi
		}

		function check_dependency {
			local command=$1
			echo -n "   " $command "... "

			# output yes or no
			if [[ `which $command` ]]; then
				if [[ $command == "virtualenv" ]]; then
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

		echo Checking dependencies already installed

		check_dependency brew
		check_dependency python3
		check_dependency pip
		check_dependency virtualenv
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
			install_dependency python3
			install_dependency pip
			install_dependency virtualenv

			echo done
		fi

	elif [[ ($platform == "Linux") && (`lsb_release -a 2>&1 | grep -c 'Ubuntu 16\|Ubuntu 18'` == 1) ]]; then

		function check_dependency {
			local command=$1
			echo -n "   " $command "... "

			if [[ `which $command` ]]; then
				if [[ $command == "virtualenv" ]]; then
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
				if [[ $command == "docker" ]]; then
					sudo apt-get install docker.io
				elif [[ $command == "python3" ]]; then
					sudo apt-get install build-essential libssl-dev libffi-dev python-dev
					sudo apt-get install python3.6
				elif [[ $command == "pip" ]]; then
					sudo apt install python-pip
			  elif [[ $command == "virtualenv" ]]; then
			    sudo apt install virtualenv
				fi
		  elif [[ ($command == "virtualenv") && (${NEEDS_VIRTUALENV} == 1) ]]; then
		    sudo -H pip install --upgrade virtualenv
			fi
		}

		echo Checking dependencies already installed

		check_dependency python3
		check_dependency pip
		check_dependency virtualenv
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

			install_dependency docker
			install_dependency python3
			install_dependency pip
			install_dependency virtualenv

			echo done
		fi

	fi
}

# Gather Info
echo
echo The MindMeld Installer checks dependencies and installs them.
echo

check_os_and_execute_deps

echo
echo Checking dependent services already started
echo done
echo
