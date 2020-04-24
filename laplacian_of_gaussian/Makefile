CC = g++
CFLAGS = -g -Wall
LIBS = `pkg-config --libs opencv`

V1/laplacianOfGaussian : V1/log.cpp
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

V1.5/laplacianOfGaussian-cu : V1.5/log.cu
	nvcc -o $@ $< $(LIBS)

V2/laplacianOfGaussian-cu : V2/log.cu
	nvcc -o $@ $< $(LIBS)

V3/laplacianOfGaussian-cu : V3/log.cu
	nvcc -o $@ $< $(LIBS)

.PHONY : clean

clean : 
			rm V1/laplacianOfGaussian
			rm V1.5/laplacianOfGaussian-cu
			rm V2/laplacianOfGaussian-cu
			rm V3/laplacianOfGaussian-cu