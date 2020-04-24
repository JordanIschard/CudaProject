#include <iostream>
#include <opencv2/opencv.hpp> 
#include <vector>
#include <stdlib.h>
#include <stdio.h>

// Matrix de convolution 
//
//  0  0 -1  0  0
//  0 -1 -2 -1  0
// -1 -2 16 -2 -1
//  0 -1 -2 -1  0
//  0  0 -1  0  0
__global__ void laplacian_of_gaussian(unsigned char* data_in, unsigned char* data_out, size_t rows, size_t cols)
{
    extern __shared__ unsigned char sh[];

    // On récupère les coordonnées du pixel
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;

    auto li = threadIdx.x;
    auto lj = threadIdx.y;

    if(i < rows && j < cols)
        sh[li * blockDim.x + lj] = ( 307 * data_in[3* (i * cols + j)] + 604 * data_in[3 * (i * cols + j) + 1] + 113 * data_in[3 * (i * cols + j) + 2] ) /1024;

    __syncthreads();

    auto result = 0;

    auto colsSH = blockDim.x;

    if( li >= 2 && li < (blockDim.x - 2) && lj >= 2 && lj < (blockDim.y - 2) )
    {
        // Tous les pixels que l'on multiplie par 16
        result = sh[(li * colsSH + lj)] * 16

        // Tous les pixels que l'on multiplie par -2
        + ( sh[((li-1) * colsSH + lj)] + sh[((li+1) * colsSH + lj)] + sh[(li * colsSH + (lj-1))] + sh[(li * colsSH + (lj+1))] ) * -2

        // Tous les pixels que l'on multiplie par -1
        + ( sh[((li-2) * colsSH + lj)] + sh[((li+2) * colsSH + lj)] + sh[(li * colsSH + (lj-2))] + sh[(li * colsSH + (lj+2))] 
            + sh[((li-1) * colsSH + (lj-1))] + sh[((li-1) * colsSH + (lj+1))] + sh[((li+1) * colsSH + (lj-1))] + sh[((li+1) * colsSH + (lj+1))] ) * -1;

        result = result * result;
        result > 255*255 ? result = 255*255 : result;

        data_out[ i * cols + j ] = sqrt((float)result);
    }
}

int main(int argc, char** argv)
{
    printf("Number of argument : %d\n", argc);

    if(argc >= 2){

        int threadSize = 32;

        if(argc == 3){
            threadSize = atoi(argv[2]);
        }

        // Mesure de temps
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        std::cout << "Création du timer faite" << std::endl;

        // Récupère l'image
        cv::Mat image_in = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
        // Récupère les informations des pixels
        auto data_in = image_in.data;
        auto rows = image_in.rows;
        auto cols = image_in.cols;

	
        std::cout << "rows = " << rows << " columns = " << cols << std::endl;

        // On crée les informations de sorties 
        std::vector<unsigned char> out(rows * cols); 
        // On crée l'image de sortie
        cv::Mat image_out(rows, cols, CV_8UC1, out.data());

        std::cout << "Image et données de sortie initialisées" << std::endl;

        // On copie l'image d'entrée sur le device
        unsigned char * image_in_device;
        // On crée une copie des informations de sortie sur le device
        unsigned char* data_out_device;

        cudaMalloc(&image_in_device, 3 * rows * cols);
        cudaMalloc(&data_out_device, rows * cols);

        std::cout << "Image sur le device allouée" << std::endl;

        std::cout << "Données de sortie sur le device allouées" << std::endl;

        cudaMemcpy(image_in_device, data_in,  rows * cols, cudaMemcpyHostToDevice );
                                                                                    
        std::cout << "Image d'entrée mise sur le device" << std::endl;

        dim3 threads(threadSize, threadSize );
        dim3 blocks(( cols -1 ) / threads.x + 1 , ( rows - 1) / threads.y + 1);

        std::cout << "Nombre de threads = " << threads.x << "  " << threads.y << std::endl;
        std::cout << "Nombre de blocks = " << blocks.x << "  " << blocks.y << std::endl;

        // Lancement du timer
        cudaEventRecord(start);

        std::cout << "Lancement du timer" << std::endl;
        
        // lancement du programme
        laplacian_of_gaussian<<< blocks , threads >>>(image_in_device, data_out_device, rows, cols);

        // On arrête le timer
        cudaEventRecord(stop);

        std::cout << "Fin du timer" << std::endl;

        cudaDeviceSynchronize();
        auto err = cudaGetLastError();
        if( err != cudaSuccess )
        {
            printf("Errors found :\n %s", cudaGetErrorString(err));
        }

        // On copie les informations de sortie du device vers le host
        cudaMemcpy(out.data(), data_out_device, rows * cols, cudaMemcpyDeviceToHost );
        
        // On récupère le temps d'exécution
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Execution time : %f\n",milliseconds);

        cv::imwrite( "out/outCudaV2.jpg", image_out);

        // On libère l'espace sur le device
        cudaFree(image_in_device);
        cudaFree(data_out_device);
    }

    return 0;
}
