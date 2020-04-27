#include <opencv2/opencv.hpp>
#include <vector>

__global__ void laplacian_of_gaussian(unsigned char * data_rgb, unsigned char * const data_out, std::size_t rows, std::size_t cols)
{
    auto i = blockIdx.x * (blockDim.x - 4)+ threadIdx.x;
    auto j = blockIdx.y * (blockDim.y - 4) + threadIdx.y;

    auto gray_i = threadIdx.x;
    auto gray_j = threadIdx.y;

    extern __shared__ unsigned char data_gray[];

    auto cols_gray = blockDim.x;


    data_gray[ gray_j * cols_gray + gray_i ] = ( 
            307 * data_rgb[ 3 * (j * cols + i) ]
        +   604 * data_rgb[ 3 * (j * cols + i) + 1 ]
        +   113 * data_rgb[ 3 * (j * cols + i) + 2 ]
    ) / 1024;
        

    __syncthreads();


    if( gray_i > 1 && gray_i < (cols_gray - 2) && gray_j > 1 && gray_j < (blockDim.y - 2))
    {
        // Tous les pixels que l'on multiplie par 16
        auto result = data_gray[(gray_j * cols_gray + gray_i)] * 16

        // Tous les pixels que l'on multiplie par -2
        + ( data_gray[((gray_j-1) * cols_gray + gray_i)] + data_gray[((gray_j+1) * cols_gray + gray_i)] + data_gray[(gray_j * cols_gray + (gray_i-1))] + data_gray[(gray_j * cols_gray + (gray_i+1))] ) * -2

        // Tous les pixels que l'on multiplie par -1
        + ( data_gray[((gray_j-2) * cols_gray + gray_i)] + data_gray[((gray_j+2) * cols_gray + gray_i)] + data_gray[(gray_j * cols_gray + (gray_i-2))] + data_gray[(gray_j * cols_gray + (gray_i+2))] 
            + data_gray[((gray_j-1) * cols_gray + (gray_i-1))] + data_gray[((gray_j-1) * cols_gray + (gray_i+1))] + data_gray[((gray_j+1) * cols_gray + (gray_i-1))] + data_gray[((gray_j+1) * cols_gray + (gray_i+1))] ) * -1;


        result = result * result;
        result = result > 255*255 ? result = 255*255 : result;
        data_out[ j * cols + i ] = sqrt((float) result);
    }
}

int main(int argc, char** argv)
{
    printf("Number of argument : %d\n", argc);

    if(argc >= 2){

        cudaError_t err;

        int threadSize = 32;
        int streamsNumber = 2;

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
        auto size_data_in = (rows  / streamsNumber) * cols * 3;
        auto size_data_out = (rows / streamsNumber)  * cols;

        unsigned char ** data_in_streams; 


        err = cudaMallocHost(&data_in_streams, streamsNumber);
        //if( err != cudaSuccess ) { std::cerr << "Erreur malloc data_in_streams" << std::endl; } 

        for( std::size_t i = 0 ; i < streamsNumber ; ++i ){
            err = cudaMalloc(&data_in_streams[i], size_data_in);
            //if( err != cudaSuccess ) { std::cerr << "Erreur malloc data_in_streams[" << i << "]" << std::endl; } 
        }

        unsigned char ** data_out_streams; 


        err = cudaMallocHost(&data_out_streams, streamsNumber);
        //if( err != cudaSuccess ) { std::cerr << "Erreur malloc data_out_streams" << std::endl; } 

        for( std::size_t i = 0 ; i < streamsNumber ; ++i ){
            err = cudaMalloc(&data_out_streams[i], size_data_out);
            //if( err != cudaSuccess ) { std::cerr << "Erreur malloc data_out_streams[" << i << "]" << std::endl; } 
        }

        //std::cout << "rows = " << rows << " columns = " << cols << std::endl;

        // On crée les informations de sorties 
        std::vector<unsigned char> out(rows * cols); 
        // On crée l'image de sortie
        cv::Mat image_out(rows, cols, CV_8UC1, out.data());

        //std::cout << "Image et données de sortie initialisées" << std::endl;

        //std::cout << "Image sur le device allouée" << std::endl;

        //std::cout << "Données de sortie sur le device allouées" << std::endl;

        cudaStream_t streams[ streamsNumber ];

        for( std::size_t i = 0 ; i < streamsNumber ; ++i ){
            cudaStreamCreate( &streams[i]);
        }

        for( std::size_t i = 0 ; i < streamsNumber ; ++i ){
            cudaMemcpyAsync( data_in_streams[i], data_rgb + i * size_data_in + 4 * (streamsNumber-(i+1)), size_data_in, cudaMemcpyHostToDevice, streams[i]);
        }
                                                                    
        //std::cout << "Image d'entrée mise sur le device" << std::endl;

        dim3 threads(threadSize, threadSize );
        dim3 blocks((( cols -1 ) / (threads.x-4)) + 1 , (( rows - 1) / (threads.y-4)) / streamsNumber + 1);

        //std::cout << "Nombre de threads = " << threads.x << "  " << threads.y << std::endl;
        //std::cout << "Nombre de blocks = " << blocks.x << "  " << blocks.y << std::endl;

        // Lancement du timer
        cudaEventRecord(start);

        //std::cout << "Lancement du timer" << std::endl;
        
        for( std::size_t i = 0 ; i < streamsNumber ; ++i ){
            // lancement du programme
            laplacian_of_gaussian<<< blocks , threads , threadSize * threadSize, streams[i]>>>(data_in_streams[i], data_out_streams[i], rows, cols);
        }

        
        // On arrête le timer
        cudaEventRecord(stop);

        //std::cout << "Fin du timer" << std::endl;

        cudaDeviceSynchronize();
        err = cudaGetLastError();
        /*if( err != cudaSuccess )
        {
            printf("Errors found :\n %s", cudaGetErrorString(err));
        }*/

        for( std::size_t i = 0 ; i < streamsNumber ; ++i ){
            cudaMemcpyAsync( out.data() + i * size_data_out, data_out_streams[i], size_data_out, cudaMemcpyDeviceToHost, streams[i]);
        }
        
        // On récupère le temps d'exécution
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Execution time : %f\n",milliseconds);

        cv::imwrite( "outCudaV2.jpg", image_out);


        for( std::size_t i = 0 ; i < streamsNumber ; ++i ){
            cudaStreamDestroy(streams[i]);
            cudaFree(data_in_streams[i]);
            cudaFree(data_out_streams[i]);
        }
    }

    return 0;
}
