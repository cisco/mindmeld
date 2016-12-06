Getting Started
===============


Install XCode and functools

Install XCode from the App Store

Open XCode and 'Agree' to the Terms & Conditions to finish installing CLI tools
Install the functools32 package (it's included in requirements.txt):

Make sure you are running Python 2.7.1x, and then run::

    pip install functools32

These are the steps you'd follow for the Barista app after completing all the steps above.

Check out the repo on the develop branch::

	git clone -b develop git@github.com:expectlabs/barista
	cd barista

Make sure submodules are set up::

	git submodule update --init

Start the numerical parser::

	scripts/start_num_parse.py

Build::

	MM_ENVIRONMENT=dev ./buildme

Run::

	MM_ENVIRONMENT=dev ./runme

Test:
Open browser to http://localhost:7150/barista/test.html

How to specify Workbench as a submodule from within an app::

	git submodule add git@github.com:expectlabs/mindmeld-workbench.git vendor/mindmeld-workbench
