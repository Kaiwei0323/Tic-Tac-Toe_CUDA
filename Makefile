# IDIR=./
CXX = nvcc

# Paths for CUDA and OpenCV
CUDA_INCLUDE_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include"
OPENCV_INCLUDE_PATH = "C:\Program Files\opencv\build\include"
OPENCV_LIB_PATH = "C:\Program Files\opencv\build\x64\vc16\lib"

# OpenCV libraries to link against
OPENCV_LIBS = -lopencv_world480 -lopencv_world480d

.PHONY: clean build run

build: tic-tac-toe.cu tic-tac-toe.h 
	$(CXX) tic-tac-toe.cu --std c++17 -o tic-tac-toe.exe -I$(CUDA_INCLUDE_PATH) -I$(OPENCV_INCLUDE_PATH) -L$(OPENCV_LIB_PATH) $(OPENCV_LIBS) -lcuda

clean:
	rm -f tic-tac-toe.exe

run:
	./tic-tac-toe.exe

all: clean build run
