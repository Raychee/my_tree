.PHONY: main test check clean cleanall

main:
	cd src; make

test:
	cd src; make test

check:
	cd src; make check

clean:
	rm -f src/*/*.o src/*.o

cleanall:
	rm -f src/*/*.o src/*.o
	rm -f bin/* bin/*.dSYM