CC = g++
CFLAGS = -g -Wall
LIBS = `pkg-config --libs opencv`

CPU/laplacianOfGaussian : CPU/log.cpp
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

.PHONY : clean

clean : 
			rm CPU/laplacianOfGaussian
			
			
			
