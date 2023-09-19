OPT = -O2
CFLAGS = $(OPT)

PREFIX = /usr/local
INCLUDEDIR = $(PREFIX)/include
LIBDIR = $(PREFIX)/lib

all: lib

lib: sscPimegaBackend/libsscPimegaBackend.so

sscPimegaBackend/libsscPimegaBackend.so: sscPimegaBackend/ssc_pimega_backend.c
	$(CC) $(CFLAGS) -shared -o $@ -fPIC $<

data:
	cd example/python/ && python3 generate.py

main%: lib
main%: example/example%.c
	$(CC) $(CFLAGS) $< -IsscPimegaBackend/ -LsscPimegaBackend/ -lsscPimegaBackend -Wl,-rpath,${PWD}/sscPimegaBackend -o $@

test540D: data lib main540D
	for i in $$(seq 10); do ./main540D example/python/x540D.b example/python/y540D.b example/python/rawSimulated540D.b temp.b; done
	python3 -c "import numpy, matplotlib.pyplot as plt; o=numpy.fromfile('example/python/rawSimulated540D.b',dtype=numpy.float32).reshape([3072,3072]); d=numpy.fromfile('temp.b',dtype=numpy.float32).reshape([3072,3072]); plt.imshow(numpy.hstack((o,d))); plt.show()"

test135D: data lib main135D
	for i in $$(seq 10); do ./main135D example/python/x135D.b example/python/y135D.b example/python/rawSimulated135D.b temp.b; done
	python3 -c "import numpy, matplotlib.pyplot as plt; o=numpy.fromfile('example/python/rawSimulated135D.b',dtype=numpy.float32).reshape([1536,1536]); d=numpy.fromfile('temp.b',dtype=numpy.float32).reshape([1536,1536]); plt.imshow(numpy.hstack((o,d))); plt.show()"

clean:
	$(RM) -f *~ sscPimegaBackend/*.so sscPimegaBackend/*~ main135D main540D temp.b example/python/*.b

install: lib
	install -Dm755 sscPimegaBackend/libsscPimegaBackend.so $(LIBDIR)/
	install -Dm644 sscPimegaBackend/ssc_pimega_backend.h $(INCLUDEDIR)/
