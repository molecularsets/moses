SHELL := /bin/bash


run_tests:
	tox

install_dep:
	sudo pip install tox

run_platform:
	echo "to the moon"
