#include <opencv2/opencv.hpp> 
#include <vector>
#include <chrono>

using namespace cv;
using namespace std;

// Matrix de convolution 
//
//  0  0 -1  0  0
//  0 -1 -2 -1  0
// -1 -2 16 -2 -1
//  0 -1 -2 -1  0
//  0  0 -1  0  0
void laplacian_of_gaussian_bis(unsigned char* data_in, unsigned char* data_out, int rows, int cols)
{

    int result;
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            result = 0;
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

                data_out[ i * cols + j] = sqrt(result);
            }
        }
    }
}

void laplacian_of_gaussian(unsigned char* in_data, unsigned char* out_data, int rows, int cols)
{
    
    for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                int c = 0;
                for (int shift_row = i-2; shift_row <= i+2; shift_row++)
                {
                    for (int shift_col = j-2; shift_col <= j+2; shift_col++)
                    {
                        if(0 <= shift_col && shift_col < cols && 0 <= shift_row && shift_row < rows)
                        {
                            if(shift_col == j && shift_row == i)
                            {
                                c += in_data[3 * (shift_row * cols + shift_col)] * 16;         
                            }
                            else
                            {
                                if((shift_col == j && (shift_row == i-1 || shift_row == i+1)) 
                                || (shift_row == i && (shift_col == j-1 || shift_col == j+1)))
                                {
                                    c += in_data[3 * (shift_row * cols + shift_col)] * -2;   
                                }
                                else
                                {
                                    if(((shift_row == i-1 || shift_row == i+1) && (shift_col == j-1 || shift_col == j+1))
                                    || (shift_col == j && (shift_row == i-2 || shift_row == i+2))
                                    || (shift_row == i && (shift_col == j-2 || shift_col == j+2)))
                                    {
                                        c += in_data[3 * (shift_row * cols + shift_col)] * -1;
                                    }   
                                }
                            }               
                        }
                    }
                }
                c = c*c;
                c > 255*255 ? c = 255*255 : c;

                out_data[ i * cols + j ] = sqrt(c);
            }
        }
}

int main(int argc, char** argv)
{
    printf("Number of argument : %d\n", argc);

    if(argc == 2){
        Mat image = imread(argv[1]);

        unsigned char* data_out = (unsigned char*)malloc((image.cols * image.rows)*sizeof(unsigned char)); 

        Mat out( image.rows , image.cols , CV_8UC1 , data_out);

        auto start = chrono::high_resolution_clock::now(); 

        laplacian_of_gaussian_bis(image.data,data_out,image.rows,image.cols);
        
        auto stop = chrono::high_resolution_clock::now(); 

        imshow("out.jpg",out);

        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start); 
  
        cout << duration.count() << endl; 

        waitKey(0);
    }
    

    return 0; 
}
