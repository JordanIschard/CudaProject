#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

using namespace cv;
using namespace std;

/**
 * À partir d'une image en couleur, on crée une réplique
 *  en noir et blanc
 *
 * @param data_in les données de l'image d'entrée
 * @param rows le nombre de lignes de l'image
 * @param cols le nombre de colonnes de l'image
 *
 * @param data_out les données de la réplique en noir et blanc de l'image originelle
 */
void grayscale(unsigned char* data_in, unsigned char* data_out, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            data_out[i * cols + j] = ( 307 * data_in[3* (i * cols + j)] + 604 * data_in[3 * (i * cols + j) + 1] + 113 * data_in[3 * (i * cols + j) + 2] ) /1024;
        }
    }
}

/**
 * Applique la matrice de convolution ci-dessous sur chaque pixel
 * de l'image en noir et blanc d'entrée
 *
 *  1/9  1/9  1/9
 *  1/9  1/9  1/9
 *  1/9  1/9  1/9
 *
 * @param data_in les données de l'image d'entrée
 * @param rows le nombre de lignes de l'image
 * @param cols le nombre de colonnes de l'image
 *
 * @param data_out les données de la réplique, avec la convolution appliquée, de l'image originelle
 */
 void simple_box_blur(unsigned char* data_in, unsigned char* data_out, int rows, int cols)
{

    int result;
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            result = 0;
            if( i >= 1 && i < (rows - 1) && j >= 1 && j < (cols - 1) )
            {
                result = (data_in[(i * cols + j)]
                          + data_in[((i-1) * cols + j)] + data_in[((i+1) * cols + j)] + data_in[(i * cols + (j-1))] + data_in[(i * cols + (j+1))]
                          + data_in[((i-1) * cols + (j-1))] + data_in[((i-1) * cols + (j+1))] + data_in[((i+1) * cols + (j-1))] + data_in[((i+1) * cols + (j+1))]
                        )* (1/9);

                result = result * result;
                result > 255*255 ? result = 255*255 : result;

                data_out[ i * cols + j] = sqrt(result);
            }
        }
    }
}

int main(int argc, char** argv)
{
    //printf("Number of argument : %d\n", argc);

    if(argc == 2){
        Mat image = imread(argv[1]);


        unsigned char* tmp = (unsigned char*)malloc((image.cols * image.rows)*sizeof(unsigned char));
        unsigned char* data_out = (unsigned char*)malloc((image.cols * image.rows)*sizeof(unsigned char));

        Mat out( image.rows , image.cols , CV_8UC1 , data_out);

        auto start = chrono::high_resolution_clock::now();

        grayscale(image.data,tmp,image.rows,image.cols);
        simple_box_blur(tmp,data_out,image.rows,image.cols);

        auto stop = chrono::high_resolution_clock::now();

        imwrite("out/outCPPsbb.jpg",out);
		print("outCPPsbb Fait normallement peut etre j'espere ")

        auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);

        cout << duration.count() << endl;

    }

    return 0;
}
