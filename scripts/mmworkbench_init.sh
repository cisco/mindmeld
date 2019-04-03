#!/usr/bin/env bash

# needed for pyenv install fix
# xcode-select --install 2&>/dev/null

set -e

NEEDS_DEP_INSTALL=0
NEEDS_ES_DEP_INSTALL=0
NEEDS_JAVA=0
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
				elif [[ $command == "java" ]]; then
					brew tap caskroom/cask
					brew cask install homebrew/cask-versions/java8
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
				brew cask install homebrew/cask-versions/java8
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

		function check_dependency {
			local command=$1
			echo -n "   " $command "... "

			# output yes or no
			if [[ `which $command` ]]; then
				if [[ $command == "java" ]]; then
					version=$(java -version 2>&1 | head -1 | awk '{print $3}' | sed "s/\"//g")
					if [[ $version == 8* ]]; then
						echo yes
					else
						echo older version $version found. 8+ needed.
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

		echo Checking dependencies already installed

		check_dependency brew
		check_dependency python3
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
			install_dependency python3
			install_dependency pip
			install_dependency virtualenv
			install_dependency java
			install_dependency elasticsearch

			echo done
		fi

	elif [[ ($platform == "Linux") && (`lsb_release -a 2>&1 | grep -c 'Ubuntu 16\|Ubuntu 18'` == 1) ]]; then

		function check_service {
			local command=$1
			echo -n "   " $command "... "

			is_200_ok=$(wget http://localhost:9200/ 2>&1 | grep -c '200 OK')

			if [[ ($command == "elasticsearch") && ($is_200_ok == 0) ]]; then
				echo no
				echo "   " starting elasticsearch ...
				sudo docker run -d -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:6.7.0
			else
				echo yes
			fi
		}

		function check_dependency {
			local command=$1
			echo -n "   " $command "... "

			# output yes or no
			if [[ $command == "elasticsearch" ]]; then
				if [[ ! `sudo docker ps | grep docker` ]]; then
					echo no
					NEEDS_DEP_INSTALL=1
					NEEDS_ES_DEP_INSTALL=1
				else
					echo yes
				fi
			else
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
			  elif [[ $command == "elasticsearch" && NEEDS_ES_DEP_INSTALL == 1 ]]; then
			    sudo docker pull docker.elastic.co/elasticsearch/elasticsearch:6.7.0
					sudo docker run -d -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:6.7.0
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

			install_dependency docker
			install_dependency python3
			install_dependency pip
			install_dependency virtualenv
			install_dependency elasticsearch

			echo done
		fi

	fi
}

# Gather Info
echo
echo The MindMeld Workbench Installer checks dependencies and installs them.
echo

check_os_and_execute_deps

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
