#include <png.h>
#include <zlib.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define SCALE 8


__device__  int mask[MASK_N][MASK_X][MASK_Y] = {
        {{ -1, -4, -6, -4, -1},
                { -2, -8,-12, -8, -2},
                {  0,  0,  0,  0,  0},
                {  2,  8, 12,  8,  2},
                {  1,  4,  6,  4,  1}},
        {{ -1, -2,  0,  2,  1},
                { -4, -8,  0,  8,  4},
                { -6,-12,  0, 12,  6},
                { -4, -8,  0,  8,  4},
                { -1, -2,  0,  2,  1}}
};


static const  int cpu_mask[MASK_N][MASK_X][MASK_Y] = {
        {{ -1, -4, -6, -4, -1},
                { -2, -8,-12, -8, -2},
                {  0,  0,  0,  0,  0},
                {  2,  8, 12,  8,  2},
                {  1,  4,  6,  4,  1}},
        {{ -1, -2,  0,  2,  1},
                { -4, -8,  0,  8,  4},
                { -6,-12,  0, 12,  6},
                { -4, -8,  0,  8,  4},
                { -1, -2,  0,  2,  1}}
};
// clang-format on

int read_png(const char* filename, unsigned char** image, unsigned* height, unsigned* width,
    unsigned* channels) {
    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8)) return 1; /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) return 4; /* out of memory */

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4; /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32 i, rowbytes;
    png_bytep row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int)png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char*)malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0; i < *height; ++i) {
        row_pointers[i] = *image + i * rowbytes;
    }

    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width,
    const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}


void sobel(unsigned char* s, unsigned char* t, unsigned height, unsigned width, unsigned channels) {
    int x, y, i, v, u;
    int R, G, B;
    double val[MASK_N * 3] = {0.0};
    int adjustX, adjustY, xBound, yBound;
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            for (i = 0; i < MASK_N; ++i) {
                adjustX = (MASK_X % 2) ? 1 : 0;
                adjustY = (MASK_Y % 2) ? 1 : 0;
                xBound = MASK_X / 2;
                yBound = MASK_Y / 2;

                val[i * 3 + 2] = 0.0;
                val[i * 3 + 1] = 0.0;
                val[i * 3] = 0.0;

                for (v = -yBound; v < yBound + adjustY; ++v) {
                    for (u = -xBound; u < xBound + adjustX; ++u) {
                        if ((x + u) >= 0 && (x + u) < width && y + v >= 0 && y + v < height) {
                            R = s[channels * (width * (y + v) + (x + u)) + 2];
                            G = s[channels * (width * (y + v) + (x + u)) + 1];
                            B = s[channels * (width * (y + v) + (x + u)) + 0];
                            val[i * 3 + 2] += R * cpu_mask[i][u + xBound][v + yBound];
                            val[i * 3 + 1] += G * cpu_mask[i][u + xBound][v + yBound];
                            val[i * 3 + 0] += B * cpu_mask[i][u + xBound][v + yBound];
                        }
                    }
                }
            }

            double totalR = 0.0;
            double totalG = 0.0;
            double totalB = 0.0;
            for (i = 0; i < MASK_N; ++i) {
                totalR += val[i * 3 + 2] * val[i * 3 + 2];
                totalG += val[i * 3 + 1] * val[i * 3 + 1];
                totalB += val[i * 3 + 0] * val[i * 3 + 0];
            }

            totalR = sqrt(totalR) / SCALE;
            totalG = sqrt(totalG) / SCALE;
            totalB = sqrt(totalB) / SCALE;
            const unsigned char cR = (totalR > 255.0) ? 255 : totalR;
            const unsigned char cG = (totalG > 255.0) ? 255 : totalG;
            const unsigned char cB = (totalB > 255.0) ? 255 : totalB;
            t[channels * (width * y + x) + 2] = cR;
            t[channels * (width * y + x) + 1] = cG;
            t[channels * (width * y + x) + 0] = cB;
        }
    }
}

__global__ void GPUSobel(unsigned char* s, unsigned char* t, unsigned int height, unsigned int width, unsigned channels)
{
    int x, y, i, v, u;
    int R, G, B;
    double val[MASK_N * 3] = {0.0};


    int adjustX, adjustY, xBound, yBound;

/*
    __shared__ int mask[MASK_N][MASK_X][MASK_Y];
    u = threadIdx.x ;
    v = threadIdx.y ;
    {
        mask[0][u][v] = dev_mask[0][u][v];
        mask[1][u][v] = dev_mask[1][u][v];
    }

    __syncthreads();// wait for each thread to copy its elemene
*/
    //    for (y = 0; y < height; ++y)
    // TODO: 轉 x,y
    x=blockIdx.y*blockDim.y + threadIdx.x;
    y=blockIdx.x*blockDim.x + threadIdx.y;
    if( x >= width ) return ;
    if( y >= height ) return ;



  //  if( threadIdx.x == 0 ) if( threadIdx.y == 0 )
  //  for( u = 0 ; u < MASK_X ; ++u )
   //     for( v = 0 ; v < MASK_Y ; ++v )

    //  for( ; y < height ; y += threadCount  )
    {

   //     x =  gridDim.y  ;
   //     for( ; x < width ; x += threadCount )
        //   for (x = 0; x < width; ++x)
        {
            for (i = 0; i < MASK_N; ++i) {
                adjustX = (MASK_X % 2) ? 1 : 0;
                adjustY = (MASK_Y % 2) ? 1 : 0;
                xBound = MASK_X / 2;
                yBound = MASK_Y / 2;
/*
                val[i * 3 + 2] = 0.0;
                val[i * 3 + 1] = 0.0;
                val[i * 3] = 0.0;
*/
                for (v = -yBound; v < yBound + adjustY; ++v)
                {
                    if(( y + v )>= 0 && ( y + v )< height )
                    for (u = -xBound; u < xBound + adjustX; ++u)
                    {
            //            if ((x + u) >= 0 && (x + u) < width && y + v >= 0 && y + v < height)
                        if ((x + u) >= 0 && (x + u) < width )
                        {
                            R = s[channels * (width * (y + v) + (x + u)) + 3];
                            G = s[channels * (width * (y + v) + (x + u)) + 2];
                            B = s[channels * (width * (y + v) + (x + u)) + 1];
                            val[i * 3 + 2] += R * mask[i][u + xBound][v + yBound];
                            val[i * 3 + 1] += G * mask[i][u + xBound][v + yBound];
                            val[i * 3 + 0] += B * mask[i][u + xBound][v + yBound];
                        }
                    }
                }
            }

            double totalR = 0.0;
            double totalG = 0.0;
            double totalB = 0.0;
            /*
            for (i = 0; i < MASK_N; ++i) {
                totalR += val[i * 3 + 2] * val[i * 3 + 2];
                totalG += val[i * 3 + 1] * val[i * 3 + 1];
                totalB += val[i * 3 + 0] * val[i * 3 + 0];
            }*/

            totalR += val[ 2] * val[ 2];
            totalG += val[ 1] * val[ 1];
            totalB += val[ 0] * val[ 0];
            totalR += val[ 3 + 2] * val[3  + 2];
            totalG += val[ 3 + 1] * val[3  + 1];
            totalB += val[ 3 + 0] * val[3 + 0];

            totalR = sqrt(totalR) / SCALE;
            totalG = sqrt(totalG) / SCALE;
            totalB = sqrt(totalB) / SCALE;
            const unsigned char cR = (totalR > 255.0) ? 255 : totalR;
            const unsigned char cG = (totalG > 255.0) ? 255 : totalG;
            const unsigned char cB = (totalB > 255.0) ? 255 : totalB;
            t[channels * (width * y + x) + 3] = cR;
            t[channels * (width * y + x) + 2] = cG;
            t[channels * (width * y + x) + 1] = cB;
        }
    }
}
int main(int argc, char** argv) {
    assert(argc == 3);

    unsigned int height, width, channels;
    unsigned char* src_img = NULL;

    read_png(argv[1], &src_img, &height, &width, &channels);
  //  printf( "%s %d %d %d\n" , argv[1] , height , width , channels );
  //  assert(channels == 3);

    unsigned int memorySize = height * width * channels * sizeof(unsigned char) ;
    unsigned char* dst_img =
        (unsigned char*)malloc( memorySize );

    // 指定 gpu
//    cudaSetDevice(1);

    if(( width * height ) < 560000 )
    {
        sobel(src_img , dst_img ,  height, width, channels  );
    }else
    {
        // GPU
        unsigned char *gpu_Input ;
        cudaMalloc(&gpu_Input, memorySize);
        // copy 到 gpu
        cudaMemcpy(gpu_Input, src_img, memorySize, cudaMemcpyHostToDevice);

        unsigned char *gpu_Output ;
        cudaMalloc(&gpu_Output, memorySize);
    /*
        dim3 threadsPerBlock (4, 3, 1);
        dim3 numBlocks ((width+threadsPerBlock.x -1)/
                        threadsPerBlock.x, height+threadsPerBlock.y-1/
                                                                threadsPerBlock.y, 1);

        static const unsigned int threadCount = 256 ;
    //*/
        // 改以 2d 方式處理
        int gridSize = 8 ;
        // 把圖分成這些區塊
        dim3 grid(
                height/gridSize+1
                  ,width/gridSize+1,1);  //
        dim3 block(gridSize,gridSize,1);         //區塊設定
        GPUSobel<<<grid,block>>>(gpu_Input , gpu_Output ,  height, width, channels  );
        // 執行
      //  GPUSobel<<<threadCount,threadCount>>>(gpu_Input , gpu_Output ,  height, width, channels , threadCount );
      //  sobel(src_img, dst_img, height, width, channels);
        // 等待完成
        cudaDeviceSynchronize();
        // 完成， gpu  copy 回來
        cudaMemcpy(dst_img, gpu_Output, memorySize, cudaMemcpyDeviceToHost);
        //
        cudaDeviceSynchronize();


        cudaFree(gpu_Input);
        cudaFree(gpu_Output);

    }
  //  sobel(src_img, dst_img, height, width, channels);
    write_png(argv[2], dst_img, height, width, channels);

    //
    free(src_img);
    free(dst_img);

    return 0;
}
