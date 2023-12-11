OUTPUTDIR := bin
SYSTYPE := mac

ifeq ($(SYSTYPE),mac)
#	CXX := /opt/homebrew/opt/llvm/bin/clang++
#	ifeq ($(CXX),)
#		CXX := /opt/homebrew/opt/llvm/bin/clang++ # fallback path
#	endif

	# Flags for the C Preprocessor
	CPPFLAGS := -I/opt/homebrew/opt/jpeg/include -I/opt/homebrew/opt/libpng/include -I../lib -I/opt/X11/include

	# Flags for the Linker
	LDFLAGS := -L/opt/homebrew/opt/jpeg/lib -L/opt/homebrew/opt/libpng/lib -L/opt/X11/lib -L/opt/homebrew/opt/gperftools/lib/

	ifeq ($(NAME), david)
		CPPFLAGS += -I/./omp/local/include -Xclang -fopenmp
		LDFLAGS += -L./omp/local/lib -lomp
	else
		LDFLAGS += -fopenmp
	endif

	OPENCV_DIR := $(shell brew --prefix opencv)
	CPPFLAGS += -I$(OPENCV_DIR)/include/opencv4
	LDFLAGS += -L$(OPENCV_DIR)/lib

	# Compiler flags
	CFLAGS := -std=c++17 -fvisibility=hidden -lpthread

	# Library flags
	LIBS := -lX11 -lpng -ljpeg -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
else ifeq ($(SYSTYPE),ghc)
	# CXX := /usr/bin/clang++

	# Flags for the C Preprocessor
	CPPFLAGS := -I/usr/include/libpng -I../lib -I/usr/include/X11

	LDFLAGS := -L/usr/lib/x86_64-linux-gnu

	# Flags for the Linker

	OPENCV_DIR := /usr/local/depot/fsl/fslpython/envs/fslpython/
	CPPFLAGS += -I$(OPENCV_DIR)/include/opencv4
	LDFLAGS += -L$(OPENCV_DIR)/lib

	# Compiler flags
	CFLAGS := -std=c++17 -fvisibility=hidden -lpthread

	# Library flags
	LIBS := -lX11 -lpng -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio

	# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/depot/fsl/fslpython/envs/fslpython/lib
else
	$(error Compiling for other system types is not supported.)
endif

ifeq (,$(CONFIGURATION))
	CONFIGURATION := release
endif

ifeq (debug,$(CONFIGURATION))
	CFLAGS += -g -fopenmp
else
	CFLAGS += -O2
	ifneq ($(NAME), david)
		CFLAGS += -fopenmp
	endif
endif

SOURCES := src/*.cpp
HEADERS := src/*.h

TARGETBIN := $(OUTPUTDIR)/pm-$(CONFIGURATION)

# Recipe to build the target binary
$(TARGETBIN): $(SOURCES) $(HEADERS)
	mkdir -p $(OUTPUTDIR)
	$(CXX) $(CPPFLAGS) $(CFLAGS) -o $@ $(SOURCES) $(LDFLAGS) $(LIBS)

.SUFFIXES:
.PHONY: all clean

all: $(TARGETBIN)

format:
	find . -iname *.h -o -iname *.cpp | xargs clang-format -i

clean:
	rm -rf ./$(OUTPUTDIR)/pm-*

FILES = src/*.cpp \
        src/*.h