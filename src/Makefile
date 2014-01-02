include ../Makefile.inc

LIBS  +=
OBJ    = main.o MyTree.o MySolver.o
TARGET = main

../bin/$(TARGET): $(OBJ)
	$(LD) $(LFLAGS) $(LIBS) $(OBJ) -o ../bin/$(TARGET)

main.o: main.cpp MyTree.o my_typedefs.h
	$(CC) $(CFLAGS) -c $< -o $@
	
MyTree.o: MyTree.cpp MyTree.hpp Tree.hpp my_typedefs.h MySolver.o
	$(CC) $(CFLAGS) -c $< -o $@

MySolver.o: MySolver.cpp MySolver.hpp SGD.hpp LabelStat.hpp Array.hpp my_typedefs.h
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: test check

test:
# $(CC) $(CFLAGS) $(LFLAGS) $(LIBS) test.cpp -o ../bin/test
	$(CC) $(CFLAGS) test.cpp -o ../bin/test

check:
	$(CC) $(CFLAGS) -fsyntax-only MyTree.cpp MySolver.cpp