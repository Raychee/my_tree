MAC := $(shell uname -a|grep -i darwin)

DEBUG := 1


ifdef MAC
LIB_SUFFIX = dylib
CC         = clang++
else
LIB_SUFFIX = so
CC         = g++
endif

LD   = $(CC)
LIBS = -L../lib
CFLAGS = -std=c++11 -pipe

ifdef DEBUG
CFLAGS += -Wall -Wextra -pedantic -ggdb3
else
CFLAGS += -O3
endif

ifdef MAC
CFLAGS += -stdlib=libc++ -mmacosx-version-min=10.7
LFLAGS += -Wl,-rpath,@executable_path/../lib
LIBS   += -lc++
else
LFLAGS += -Wl,-rpath,'$$ORIGIN/../lib'
endif
