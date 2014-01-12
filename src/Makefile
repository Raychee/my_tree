include ../Makefile.inc

LIBS  +=
OBJ    = MyTree.o MySolver.o my_lib.o

.PHONY: all test check

all: ../bin/mytree_train ../bin/mytree_test

../bin/mytree_train: main_train.o $(OBJ)
	$(LD) $(LFLAGS) $(LIBS) $^ -o ../bin/mytree_train

../bin/mytree_test: main_test.o $(OBJ)
	$(LD) $(LFLAGS) $(LIBS) $^ -o ../bin/mytree_test

main_train.o: main_train.cpp MyTree.o my_lib.o my_typedefs.h
	$(CC) $(CFLAGS) -c $< -o $@

main_test.o: main_test.cpp MyTree.o my_typedefs.h
	$(CC) $(CFLAGS) -c $< -o $@
	
MyTree.o: MyTree.cpp MyTree.hpp Tree.hpp my_typedefs.h MySolver.o
	$(CC) $(CFLAGS) -c $< -o $@

MySolver.o: MySolver.cpp MySolver.hpp SGD.hpp LabelStat.hpp Array.hpp my_typedefs.h my_lib.o
	$(CC) $(CFLAGS) -c $< -o $@

my_lib.o: my_lib.cpp my_lib.hpp
	$(CC) $(CFLAGS) -c $< -o $@

test:
# $(CC) $(CFLAGS) $(LFLAGS) $(LIBS) test.cpp -o ../bin/test
	$(CC) $(CFLAGS) test.cpp my_lib.o -o ../bin/test

check:
	$(CC) $(CFLAGS) -fsyntax-only MyTree.cpp MySolver.cpp