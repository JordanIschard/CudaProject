#include <iostream>
#include <opencv2/opencv.hpp> 
#include <vector>

using namespace cv;
using namespace std;

// Matrix de convolution 
//
//  0  0 -1  0  0
//  0 -1 -2 -1  0
// -1 -2 16 -2 -1
//  0 -1 -2 -1  0
//  0  0 -1  0  0
__global__ void laplacian_of_gaussian(unsigned const char* data_in, unsigned char* data_out, size_t rows, size_t cols)
{
    // On récupère les coordonnées du pixel
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;

    auto result = 0;

    if( i >= 2 && i < (rows - 2) && j >= 2 && j < (cols - 2) )
    {
        // Tous les pixels que l'on multiplie par 16
        result = data_in[3 * (i * cols + j)] * 16

        // Tous les pixels que l'on multiplie par -2
        + ( data_in[3 * ((i-1) * cols + j)] + data_in[3 * ((i-+1) * cols + j)] + data_in[3 * (i * cols + (j-1))] + data_in[3 * (i * cols + (j+1))] ) * -2

        // Tous les pixels que l'on multiplie par -1
        + ( data_in[3 * ((i-2) * cols + j)] + data_in[3 * ((i+2) * cols + j)] + data_in[3 * (i * cols + (j-2))] + data_in[3 * (i * cols + (j+2))] 
            + data_in[3 * ((i-1) * cols + (j-1))] + data_in[3 * ((i-1) * cols + (j+1))] + data_in[3 * ((i+1) * cols + (j-1))] + data_in[3 * ((i+1) * cols + (j+1))] ) * -1;

        result = result * result;
        result > 255*255 ? result = 255*255 : result;

        data_out[ i * cols + j ] = sqrt((float)result);
    }
}

int main(int argc, char** argv)
{
    printf("Number of argument : %d\n", argc);

    if(argc == 2){

        // Mesure de temps
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cout << "Création du timer faite" << endl;

        // Récupère l'image
        Mat image_in = imread(argv[1], IMREAD_UNCHANGED);
        // Récupère les informations des pixels
        auto data_in = image_in.data;
        auto rows = image_in.rows;
        auto cols = image_in.cols;

	    printf("First data : %d %d %d\n",data_in[0],data_in[1],data_in[2]);
        cout << "rows = " << rows << " columns = " << cols << endl;

        // On crée les informations de sorties 
        unsigned char* data_out = (unsigned char*)malloc((cols * rows)*sizeof(unsigned char)); 
        // On crée l'image de sortie
        Mat image_out(rows, cols, CV_8UC1, data_out);

        cout << "Image et données de sortie initialisées" << endl;

        // On copie l'image d'entrée sur le device
        unsigned char * image_in_device;
        cudaMalloc(&image_in_device, 3 * rows * cols * sizeof(unsigned char));
    
        cout << "Image sur le device allouée" << endl;

        cudaMemcpy(image_in_device, data_in, 3 * rows * cols * sizeof(unsigned char), cudaMemcpyHostToDevice );
                                                                                    
        cout << "image d'entrée mise sur le device" << endl;

        // On crée une copie des informations de sortie sur le device
        unsigned char* data_out_device;
        cudaMalloc(&data_out_device, rows * cols);

        cout << "Données de sortie misent sur le device" << endl;

        dim3 threads(32, 32 );
        dim3 blocks(( cols -1 ) / threads.x + 1 , ( rows - 1) / threads.y + 1);

        cout << "Nombre de threads = " << threads.x << "  " << threads.y << endl;
        cout << "Nombre de blocks = " << blocks.x << "  " << blocks.y << endl;

        // Lancement du timer
        cudaEventRecord(start);

        cout << "Lancement du timer" << endl;

        // lancement du programme
        laplacian_of_gaussian<<< blocks , threads >>>(image_in_device, data_out_device, rows, cols);

        // On arrête le timer
        cudaEventRecord(stop);

        cout << "Fin du timer" << endl;

        cudaDeviceSynchronize();
        auto err = cudaGetLastError();
        if( err != cudaSuccess )
        {
            printf("Errors found :\n %s", cudaGetErrorString(err));
        }

        // On copie les informations de sortie du device vers le host
        cudaMemcpy(data_out, data_out_device, rows * cols, cudaMemcpyDeviceToHost );
        
        // On récupère le temps d'exécution
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Execution time : %f",milliseconds);

        imwrite( "outCuda.jpg", image_out);

        // On libère l'espace sur le device
        cudaFree(image_in_device);
        cudaFree(data_out_device);
    }

    return 0;
}
