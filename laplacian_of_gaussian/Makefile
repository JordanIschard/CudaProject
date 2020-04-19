CC = gcc
CFLAGS = -g -Wall
PROG = LaplacianOfGaussian

LIBS = `pkg-config opencv --cflags --libs`

LaplacianOfGaussian : log.cpp
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

LaplacianOfGaussian-cu : log.cu
	nvcc -o $@ $< $(LIBS)

.PHONY : clean

clean : tm LaplacianOfGaussian