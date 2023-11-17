OUTPUTDIR := bin/

CFLAGS := -std=c++14 -fvisibility=hidden -lpthread -I../lib

ifeq (,$(CONFIGURATION))
	CONFIGURATION := release
endif

ifeq (debug,$(CONFIGURATION))
CFLAGS += -g
else
CFLAGS += -O2 -fopenmp
endif

SOURCES := src/*.cpp
HEADERS := $(wildcard src/*.h)

TARGETBIN := $(OUTPUTDIR)pm-$(CONFIGURATION)

# Recipe to build the target binary
$(TARGETBIN): $(SOURCES) $(HEADERS)
	mkdir -p $(OUTPUTDIR)
	$(CXX) -o $@ $(CFLAGS) $(SOURCES)

.SUFFIXES:
.PHONY: all clean

all: $(TARGETBIN)

format:
	clang-format -i src/*.cpp src/*.h

clean:
	rm -rf ./pm-$(CONFIGURATION)

FILES = src/*.cpp \
		src/*.h