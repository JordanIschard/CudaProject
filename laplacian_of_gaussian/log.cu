#include <iostream>
#include <opencv2/opencv.hpp> 
#include <vector>


__global__ void grayscale(unsigned const char* data_in, unsigned char* const data_out, size_t rows, size_t cols)
{
    // On récupère les coordonnées du pixel
    auto i = threadIdx.x;
    auto j = threadIdx.y;
    auto bi = blockIdx.x * blockDim.x + i;
    auto bj = blockIdx.y * blockDim.y + j;

    if(i < rows && j < cols)
        data_out[i * blockDim.x + j] = ( 307 * data_in[3* (bi * cols + bj)] + 604 * data_in[3 * (bi * cols + bj) + 1] + 113 * data_in[3 * (bi * cols + bj) + 2] ) /1024;
}

// Matrix de convolution 
//
//  0  0 -1  0  0
//  0 -1 -2 -1  0
// -1 -2 16 -2 -1
//  0 -1 -2 -1  0
//  0  0 -1  0  0
__global__ void laplacian_of_gaussian(unsigned const char* data_in, unsigned char* const data_out, size_t rows, size_t cols)
{
    // On récupère les coordonnées du pixel
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;

    auto result = 0;

    if( i >= 2 && i < (rows - 2) && j >= 2 && j < (cols - 2) )
    {
        // Tous les pixels que l'on multiplie par 16
        result = data_in[(i * cols + j)] * 16

        // Tous les pixels que l'on multiplie par -2
        + ( data_in[((i-1) * cols + j)] + data_in[((i+1) * cols + j)] + data_in[(i * cols + (j-1))] + data_in[(i * cols + (j+1))] ) * -2

        // Tous les pixels que l'on multiplie par -1
        + ( data_in[((i-2) * cols + j)] + data_in[((i+2) * cols + j)] + data_in[(i * cols + (j-2))] + data_in[(i * cols + (j+2))] 
            + data_in[((i-1) * cols + (j-1))] + data_in[((i-1) * cols + (j+1))] + data_in[((i+1) * cols + (j-1))] + data_in[((i+1) * cols + (j+1))] ) * -1;

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
        // On crée les informations de sorties 
        std::vector<unsigned char> image_gray(rows * cols); 
        // On crée l'image de sortie
        cv::Mat image_out(rows, cols, CV_8UC1, out.data());

        std::cout << "Image et données de sortie initialisées" << std::endl;

        // On copie l'image d'entrée sur le device
        unsigned char * image_in_device;
        // On crée une copie des données d'entrée en noir et blanc
        unsigned char * image_gray_device;
        // On crée une copie des informations de sortie sur le device
        unsigned char* data_out_device;

        cudaMalloc(&image_in_device, 3 * rows * cols);
        cudaMalloc(&image_gray_device, rows * cols);
        cudaMalloc(&data_out_device, rows * cols);
    
	    auto err1 = cudaGetLastError();
        if(err1 != cudaSuccess){
            std::cout << "You're fuck up " << cudaGetErrorString(err1) << std::endl;
        }else{
	        std::cout << "kuvheuio" << std::endl;
	    }
        std::cout << "Image sur le device allouée" << std::endl;

        std::cout << "Données de sortie sur le device allouées" << std::endl;

        cudaMemcpy(image_in_device, data_in,  3 * rows * cols, cudaMemcpyHostToDevice );
                                                                                    
        std::cout << "Image d'entrée mise sur le device" << std::endl;

        dim3 threads(32, 32 );
        dim3 blocks(( cols -1 ) / threads.x + 1 , ( rows - 1) / threads.y + 1);

        std::cout << "Nombre de threads = " << threads.x << "  " << threads.y << std::endl;
        std::cout << "Nombre de blocks = " << blocks.x << "  " << blocks.y << std::endl;

        // Lancement du timer
        cudaEventRecord(start);

        std::cout << "Lancement du timer" << std::endl;

        grayscale<<< blocks , threads >>>(image_in_device, image_gray_device, cols, rows);

        cudaMemcpy(image_gray.data(), image_gray_device,  rows * cols, cudaMemcpyDeviceToHost );
        cudaMemcpy(image_gray_device, image_gray.data(),  rows * cols, cudaMemcpyHostToDevice );
        
        // lancement du programme
        laplacian_of_gaussian<<< blocks , threads >>>(image_gray_device, data_out_device, rows, cols);

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
        printf("Execution time : %f",milliseconds);

        cv::imwrite( "outCuda.jpg", image_out);

        // On libère l'espace sur le device
        cudaFree(image_in_device);
        cudaFree(image_gray_device);
        cudaFree(data_out_device);
    }

    return 0;
}
