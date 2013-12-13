include ../Makefile.inc

LIBS   = -L../lib
OBJ    = MyTree.o
TARGET = main

ifeq ($(OS), LINUX)
LFLAGS += -Wl,-rpath,'$$ORIGIN/../lib'
else
CFLAGS += -mmacosx-version-min=10.7
LFLAGS += -Wl,-rpath,@executable_path/../lib
endif

# ifeq ($(LD), clang)
# CFLAGS += -stdlib=libc++
# LIBS   += -lc++
# endif


# ../bin/$(TARGET): $(OBJ)
# 	$(LD) $(LFLAGS) $(LIBS) $(OBJ) -o ../bin/$(TARGET)
	
MyTree.o: MyTree.cpp MyTree.h Tree.h
	$(CC) $(CFLAGS) -c $< -o $@


.PHONY: test

test:
# $(CC) $(CFLAGS) $(LFLAGS) $(LIBS) test.cpp -o ../bin/test
	$(CC) $(CFLAGS) test.cpp -o ../bin/test