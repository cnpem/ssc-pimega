RUNDIR=	sscPimega sscPimega/pi450D scPimega/pi135D sscPimega/pi540D cuda/src/ runner/

all: build install

build: 
	python3 setup.py build

install:
	python3 setup.py install --user

uninstall:
	pip3 uninstall sscPimega

clean:
	rm -fr build/ *.egg-info/ dist/	*~
	@for j in ${RUNDIR}; do rm -rf $$j/*.pyc; rm -rf $$j/*.egg-info/; rm -rf $$j/__pycache__/; rm -rf $$j/*~; done

