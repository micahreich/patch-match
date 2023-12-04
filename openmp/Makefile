OUTPUTDIR := bin/

ifeq ($(SYSTYPE),mac)
	

	# Flags for the C Preprocessor
	CPPFLAGS := -I/opt/homebrew/opt/jpeg/include -I/opt/homebrew/opt/libpng/include -I../lib -I/opt/X11/include

	# Flags for the Linker
	LDFLAGS := -L/opt/homebrew/opt/jpeg/lib -L/opt/homebrew/opt/libpng/lib -L/opt/X11/lib 

	OPENCV_DIR := $(shell brew --prefix opencv)
	CPPFLAGS += -I$(OPENCV_DIR)/include/opencv4
	LDFLAGS += -L$(OPENCV_DIR)/lib

	# Compiler flags
	CFLAGS := -std=c++17 -fvisibility=hidden -lpthread

	# Library flags
	LIBS := -lX11 -lpng -ljpeg -lopencv_core -lopencv_imgproc -lopencv_highgui
else ifeq ($(SYSTYPE),ghc)
	# Flags for the C Preprocessor
	CPPFLAGS := # Add GHC specific flags here

	# Flags for the Linker
	LDFLAGS := # Add GHC specific flags here

	# Compiler flags
	CFLAGS := # Add GHC specific flags here

	# Library flags
	LIBS := # Add GHC specific flags here
endif

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
	$(CXX) $(CPPFLAGS) $(CFLAGS) -o $@ $(SOURCES) $(LDFLAGS) $(LIBS)

.SUFFIXES:
.PHONY: all clean

all: $(TARGETBIN)

format:
	clang-format -i src/*.cpp src/*.h

clean:
	rm -rf ./pm-$(CONFIGURATION)

FILES = src/*.cpp \
        src/*.h