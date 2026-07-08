# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -I. -Iwavelet
LDFLAGS = -lm  # Link math library for functions like M_PI, sqrt, log, etc.

# Add debug flag if DEBUG=1 is passed to make
# Example usage: make DEBUG=1
#              make run-debug
ifdef DEBUG
	BUILD_TYPE = Debug
	CXXFLAGS += -DDEBUG_LOGGING -g # Add -g for debugger symbols
else
	BUILD_TYPE = Release
	# Release flags already include -O2
endif

# Executable name
TARGET = detection_pipeline_test

# Source files
# Find all .cc files in the current directory
SRCS_CC = $(wildcard *.cc)
# Explicitly list needed wavelet library source files, excluding its main.cpp
SRCS_CPP = wavelet/matrix.cpp wavelet/wavelet.cpp

# Combine source files
SRCS = $(SRCS_CC) $(SRCS_CPP)

# Object files: replace .cc/.cpp extensions with .o
OBJS = $(patsubst %.cc, %.o, $(SRCS_CC)) $(patsubst wavelet/%.cpp, wavelet/%.o, $(SRCS_CPP))

# Default target: build the release executable
all: $(TARGET)

# Rule to link the executable
$(TARGET): $(OBJS)
	@echo "Linking ($@) [$(BUILD_TYPE)]..."
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS) # Note: CXXFLAGS are used in compilation, not linking directly here

# Rule to compile .cc files in the current directory
%.o: %.cc
	@echo "Compiling ($<) [$(BUILD_TYPE)]..."
	@echo "CXXFLAGS = $(CXXFLAGS)" # Print flags for debugging
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to compile .cpp files in the wavelet subdirectory
# The object files will be created in the wavelet directory
wavelet/%.o: wavelet/%.cpp $(wildcard wavelet/*.h)
	@echo "Compiling ($<) [$(BUILD_TYPE)]..."
	@echo "CXXFLAGS = $(CXXFLAGS)" # Print flags for debugging
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Target to run the default (release) executable
run: all
	./$(TARGET)

# Target to specifically build the debug version (cleans first to ensure rebuild)
debug-build: clean
	$(MAKE) all DEBUG=1

# Target to run the debug executable (builds if necessary)
run-debug:
	$(MAKE) all DEBUG=1 # Ensure debug build exists
	./$(TARGET)

# Target to clean up generated files
clean:
	@echo "Cleaning up..."
	rm -f $(TARGET) $(OBJS) wavelet/*.o

# Phony targets (targets that don't represent files)
.PHONY: all run clean debug-build run-debug 