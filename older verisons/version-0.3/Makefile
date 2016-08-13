CC = g++
FILES = $(wildcard *.cpp)
ARMA = -larmadillo
ACC = -framework Accelerate
FILE_NAME = output

$(FILE_NAME) : main.cpp ann.cpp ann.h
	$(CC) $(FILES) $(ARMA) $(ACC) -o $(FILE_NAME)

run : $(FILE_NAME)
	./$(FILE_NAME)

clean : $(FILE_NAME)
	rm -f $(FILE_NAME)