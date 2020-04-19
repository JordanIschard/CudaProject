#include <iostream>
#include <opencv2/opencv.hpp> 
#include <vector>

using namespace cv;
using namespace std;

__global__ void laplacian_of_gaussian(unsigned char* data_in, unsigned char* data_out, int rows, int cols)
{
    // On récupère les coordonnées du pixel
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;

    int c = 0;
    for (int shift_row = i-2; shift_row <= i+2; shift_row++)
    {
        for (int shift_col = j-2; shift_col <= j+2; shift_col++)
        {
            if(0 <= shift_col && shift_col < cols && 0 <= shift_row && shift_row < rows)
            {
                if(shift_col == j && shift_row == i)
                {
                    c += data_in[3 * (shift_row * cols + shift_col)] * 16;         
                }
                else
                {
                    if((shift_col == j && (shift_row == i-1 || shift_row == i+1)) 
                    || (shift_row == i && (shift_col == j-1 || shift_col == j+1)))
                    {
                        c += data_in[3 * (shift_row * cols + shift_col)] * -2;   
                    }
                    else
                    {
                        if(((shift_row == i-1 || shift_row == i+1) && (shift_col == j-1 || shift_col == j+1))
                        || (shift_col == j && (shift_row == i-2 || shift_row == i+2))
                        || (shift_row == i && (shift_col == j-2 || shift_col == j+2)))
                        {
                            c += data_in[3 * (shift_row * cols + shift_col)] * -1;
                        }   
                    }
                }               
            }
        }
    }
    c = c*c;
    c > 255*255 ? c = 255*255 : c;

    data_out[ i * cols + j ] = sqrt(c);
}

int main(int argc, char** argv)
{
    printf("Number of argument : %d\n", argc);

    if(argc == 2){

        // Récupère l'image
        Mat image_in = imread(argv[1], IMREAD_UNCHANGED);
        // Récupère les informations des pixels
        auto data_in = image_in.data();
        auto rows = image_in.rows;
        auto cols = image_in.cols;


        // On crée les informations de sorties 
        unsigned char* data_out = (unsigned char*)malloc((cols * rows)*sizeof(unsigned char)); 
        // On crée l'image de sortie
        Mat image_out(rows, cols, CV_8UC1, data_out);

        // On copie l'image d'entrée sur le device
        unsigned char* image_in_device;
        cudaMalloc(&image_in_device, 3 * rows * cols);
        cudaMemcpy(image_in_device, image_in, 3 * rows * cols, cudaMemcpyHostToDevice );

        // On crée une copie des informations de sortie sur le device
        unsigned char* data_out_device;
        cudaMalloc(&data_out_device, rows * cols);

        dim3 threads(32, 32 );
        dim3 blocks(( cols -1 ) / threads.x + 1 , ( rows - 1) / threads.y + 1);

        laplacian_of_gaussian<<< blocks , threads >>>(image_in_device, data_out_device, rows, cols);

        cudaDeviceSynchronize();
        auto err = cudaGetLastError();
        if( err != cudaSuccess )
        {
            printf("Errors found :\n %s", cudaGetErrorString(err));
        }

        // On copie les informations de sortie du device vers le host
        cudaMemcpy(data_out, data_out_device, rows * cols, cudaMemcpyDeviceToHost );
        
        imwrite( "outCuda.jpg", image_out);

        // On libère l'espace sur le device
        cudaFree(image_in_device);
        cudaFree(data_out_device);
    }

    return 0;
}