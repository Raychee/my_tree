OS ?= MAC
SCHEME ?= DEBUG

ifeq ($(OS), LINUX)
LIB_SUFFIX = so
else	# $(OS) = MAC
LIB_SUFFIX = dylib
endif

CC = clang++
LD = clang++

ifeq ($(SCHEME), DEBUG)
CFLAGS = -Wall -Wextra -pedantic -pipe -ggdb3 -std=c++11 -stdlib=libc++
LFLAGS = 
else
CFLAGS = -pipe -O3
LFLAGS =
endif