include ../Makefile.inc

LIBS   = -L../lib -lc++
OBJ    = main.o MySolver.o
TARGET = main

ifeq ($(OS), LINUX)
LFLAGS += -Wl,-rpath,'$$ORIGIN/../lib'
else
CFLAGS += -mmacosx-version-min=10.7 -stdlib=libc++
LFLAGS += -Wl,-rpath,@executable_path/../lib
endif

# ifeq ($(LD), clang)
# CFLAGS += -stdlib=libc++
# LIBS   += -lc++
# endif


../bin/$(TARGET): $(OBJ)
	$(LD) $(LFLAGS) $(LIBS) $(OBJ) -o ../bin/$(TARGET)

main.o: main.cpp MySolver.hpp my_typedefs.h
	$(CC) $(CFLAGS) -c $< -o $@
	
# MyTree.o: MyTree.cpp MyTree.h Tree.h
# 	$(CC) $(CFLAGS) -c $< -o $@

MySolver.o: MySolver.cpp MySolver.hpp SGD.hpp LabelStat.hpp Array.hpp my_typedefs.h
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: test check

test:
# $(CC) $(CFLAGS) $(LFLAGS) $(LIBS) test.cpp -o ../bin/test
	$(CC) $(CFLAGS) test.cpp -o ../bin/test

check:
	$(CC) $(CFLAGS) -fsyntax-only MyTree.cpp MySolver.cpp