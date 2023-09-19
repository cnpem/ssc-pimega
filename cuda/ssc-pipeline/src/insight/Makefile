VERBOSE ?= 0

export VERBOSE

.PHONY: clean all
all:
	#gcc -Wall example.c -lhdf5 -o example
	gcc  example.c -lhdf5 -o example
	gcc fpointer2.c -o fpointer2
	gcc fpointer.c -o fpointer
	gcc readh5.c -lhdf5 -o readh5

clean: 
	$(RM) -f example *~ a.out fpointer2 fpointer readh5
