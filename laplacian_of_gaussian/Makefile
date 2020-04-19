CC = g++
CFLAGS = -g -Wall
LIBS = `pkg-config --libs opencv`

laplacianOfGaussian : log.cpp
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

laplacianOfGaussian-cu : log.cu
	nvcc -o $@ $< $(LIBS)

.PHONY : clean

clean : 
			rm laplacianOfGaussian