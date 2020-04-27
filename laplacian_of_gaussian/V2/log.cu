#include <opencv2/opencv.hpp>
#include <vector>

__global__ void grayscale(unsigned char * data_rgb, unsigned char * data_gray, std::size_t rows, std::size_t cols)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;

    if( i < cols && j < rows )
    {
        data_gray[ j * cols + i ] = ( 
                307 * data_rgb[ 3 * (j * cols + i) ]
            +   604 * data_rgb[ 3 * (j * cols + i) + 1 ]
            +   113 * data_rgb[ 3 * (j * cols + i) + 2 ]
        ) / 1024;
    }
}

__global__ void laplacian_of_gaussian(unsigned char const * const data_gray, unsigned char * const data_out, std::size_t rows, std::size_t cols)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;


    if( i > 2 && i < (cols - 2) && j > 2 && j < (rows - 2))
    {
        // Tous les pixels que l'on multiplie par 16
        auto result = data_gray[(j * cols + i)] * 16

        // Tous les pixels que l'on multiplie par -2
        + ( data_gray[((j-1) * cols + i)] + data_gray[((j+1) * cols + i)] + data_gray[(j * cols + (i-1))] + data_gray[(j * cols + (i+1))] ) * -2

        // Tous les pixels que l'on multiplie par -1
        + ( data_gray[((j-2) * cols + i)] + data_gray[((j+2) * cols + i)] + data_gray[(j * cols + (i-2))] + data_gray[(j * cols + (i+2))] 
            + data_gray[((j-1) * cols + (i-1))] + data_gray[((j-1) * cols + (i+1))] + data_gray[((j+1) * cols + (i-1))] + data_gray[((j+1) * cols + (i+1))] ) * -1;


        result = result * result;
        result = result > 255*255 ? result = 255*255 : result;
        data_out[ j * cols + i ] = sqrt((float) result);
    }
}

int main(int argc, char** argv)
{
    //printf("Number of argument : %d\n", argc);

    if(argc >= 2){

        int threadSize = 32;

        if(argc == 3){
            threadSize = atoi(argv[2]);
        }

        // Mesure de temps
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        //std::cout << "Création du timer faite" << std::endl;

        // Récupère l'image
        cv::Mat image_in = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
        // Récupère les informations des pixels
        auto data_rgb = image_in.data;
        auto rows = image_in.rows;
        auto cols = image_in.cols;

	
        //std::cout << "rows = " << rows << " columns = " << cols << std::endl;

        // On crée les informations de sorties 
        std::vector<unsigned char> out(rows * cols); 
        // On crée l'image de sortie
        cv::Mat image_out(rows, cols, CV_8UC1, out.data());

        //std::cout << "Image et données de sortie initialisées" << std::endl;

        // On copie l'image d'entrée sur le device
        unsigned char * data_rgb_device;
        unsigned char * data_gray_device;
        // On crée une copie des informations de sortie sur le device
        unsigned char* data_out_device;

        cudaMalloc(&data_rgb_device, 3 * rows * cols);
        cudaMalloc(&data_gray_device, rows * cols);
        cudaMalloc(&data_out_device, rows * cols);

        //std::cout << "Image sur le device allouée" << std::endl;

        //std::cout << "Données de sortie sur le device allouées" << std::endl;

        cudaMemcpy(data_rgb_device, data_rgb,  3 * rows * cols, cudaMemcpyHostToDevice );
                                                                                    
        //std::cout << "Image d'entrée mise sur le device" << std::endl;

        dim3 threads(threadSize, threadSize );
        dim3 blocks(( cols -1 ) / threads.x + 1 , ( rows - 1) / threads.y + 1);

        //std::cout << "Nombre de threads = " << threads.x << "  " << threads.y << std::endl;
        //std::cout << "Nombre de blocks = " << blocks.x << "  " << blocks.y << std::endl;

        // Lancement du timer
        cudaEventRecord(start);

        //std::cout << "Lancement du timer" << std::endl;
        
        grayscale<<< blocks , threads >>>(data_rgb_device, data_gray_device, rows, cols);

        // lancement du programme
        laplacian_of_gaussian<<< blocks , threads >>>(data_gray_device, data_out_device, rows, cols);

        // On arrête le timer
        cudaEventRecord(stop);

        //std::cout << "Fin du timer" << std::endl;

        cudaDeviceSynchronize();
        auto err = cudaGetLastError();
        /*if( err != cudaSuccess )
        {
            printf("Errors found :\n %s", cudaGetErrorString(err));
        }*/

        // On copie les informations de sortie du device vers le host
        cudaMemcpy(out.data(), data_out_device, rows * cols, cudaMemcpyDeviceToHost );
        
        // On récupère le temps d'exécution
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Execution time : %f\n",milliseconds);

        cv::imwrite( "outCudaV1.jpg", image_out);

        // On libère l'espace sur le device
        cudaFree(data_rgb_device);
        cudaFree(data_gray_device);
        cudaFree(data_out_device);
    }

    return 0;
}
