CPP_FILES = $(wildcard *.cpp)
H_FILES = $(wildcard *.h)
OUTPUT_NAME = runnable
CC = g++-6
FLAGS = -framework Accelerate -O3 -DARMA_NO_DEBUG -march=native -larmadillo -Wall -o

OUTPUT_NAME : $(CPP_FILES) $(H_FILES)
	$(CC) $(CPP_FILES) $(FLAGS) $(OUTPUT_NAME)
