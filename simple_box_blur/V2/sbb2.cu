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

__global__ void simple_box_blur(unsigned char const * const data_gray, unsigned char * const data_out, std::size_t rows, std::size_t cols)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;

    if( i >= 1 && i < (cols - 1) && j >= 1 && j < (rows - 1))
    {
		auto result = (data_gray[(j * cols + i)] 
		+ data_gray[((j-1) * cols + i)] + data_gray[((j+1) * cols + i)]
		+ data_gray[(j * cols + (i-1))] + data_gray[(j * cols + (i+1))] 	
		+ data_gray[((j-1) * cols +(i-1))] + data_gray[((j-1) * cols +(i+1))]
		+ data_gray[((j+1) * cols +(i-1))] + data_gray[((j+1) * cols +(i+1))]	
		) /9;

        result = result * result;
        result = result > 255*255 ? result = 255*255 : result;
        data_out[ j * cols + i ] = sqrt((float) result);
    }
}

__global__ void edge_detection(unsigned char const * const data_in, unsigned char * const data_out, std::size_t rows, std::size_t cols)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;

    if( i >= 1 && i < (cols - 1) && j >= 1 && j < (rows - 1))
    {
		auto result = data_in[(j * cols + i)] *8 +(
		+ data_in[((j-1) * cols + i)] + data_in[((j+1) * cols + i)]
		+ data_in[(j * cols + (i-1))] + data_in[(j * cols + (i+1))] 	
		+ data_in[((j-1) * cols +(i-1))] + data_in[((j-1) * cols +(i+1))]
		+ data_in[((j+1) * cols +(i-1))] + data_in[((j+1) * cols +(i+1))]	
		) *(-1);

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

        // Récupère l'image
        cv::Mat image_in = cv::imread(argv[1], cv::IMREAD_UNCHANGED);

        // Récupère les informations des pixels
        auto data_rgb = image_in.data;
        auto rows = image_in.rows;
        auto cols = image_in.cols;

	
        std::cout << "rows = " << rows << " columns = " << cols << std::endl;

        // On crée les informations de sorties 
        std::vector<unsigned char> out(rows * cols); 
        // On crée l'image de sortie
        cv::Mat image_out(rows, cols, CV_8UC1, out.data());

        // On copie l'image d'entrée sur le device
        unsigned char * data_rgb_device;
        unsigned char * data_gray_device;
        // On crée une copie des informations de sortie sur le device
        unsigned char* data_out_device;
        // unsigned char * data_edge_device;

        cudaMalloc(&data_rgb_device, 3 * rows * cols);
        cudaMalloc(&data_gray_device, rows * cols);
        cudaMalloc(&data_out_device, rows * cols);
        // cudaMalloc(&data_edge_device, rows * cols);

        cudaMemcpy(data_rgb_device, data_rgb,  3 * rows * cols, cudaMemcpyHostToDevice );

        dim3 threads(threadSize, threadSize );
        dim3 blocks(( cols -1 ) / threads.x + 1 , ( rows - 1) / threads.y + 1);

        std::cout << "Nombre de threads = " << threads.x << "  " << threads.y << std::endl;
        std::cout << "Nombre de blocks = " << blocks.x << "  " << blocks.y << std::endl;

        // Lancement du timer
        cudaEventRecord(start);
        
        grayscale<<< blocks , threads >>>(data_rgb_device, data_gray_device, rows, cols);

        // lancement du programme
        simple_box_blur<<< blocks , threads >>>(data_gray_device, data_out_device, rows, cols);

		//edge_detection<<< blocks , threads >>>(data_gray_device, data_edge_device, rows, cols);

        // On arrête le timer
        cudaEventRecord(stop);

        cudaDeviceSynchronize();
        /*auto err = cudaGetLastError();
        if( err != cudaSuccess )
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

        cv::imwrite( "outSbbV2.jpg", image_out);

        // On libère l'espace sur le device
        cudaFree(data_rgb_device);
        cudaFree(data_gray_device);
        cudaFree(data_out_device);
		// cudaFree(data_edge_device);
    }

    return 0;
}
