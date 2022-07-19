#include <png.h>
#include <zlib.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <ctime>

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define SCALE 8

#define _RUN_COUNT 5
#define KEY_CODE 0x40

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
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_GRAY , PNG_INTERLACE_NONE,
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

//
__device__ void gpu_getCount( int *outCount , int x , int y , int w , int h , unsigned char* s )
{
    int dx , dy ;
    int n = 0 ;
    (*outCount) = 0 ;
    unsigned int pp = y*w + dx ;
    if(( x - 1 ) >= 0 )
    {

        dx = x - 1 ;
        n += ( s[ pp - 1 ] > KEY_CODE );
        dy = y - 1 ;//
        if( dy >= 0 )
            n += ( s[ pp - 1 - w ] > KEY_CODE );

   //     dy = y + 1 ;//
        if( dy < h )
            n += ( s[ pp - 1 + w ] > KEY_CODE );

    }

    if(( x + 1 ) < w )
    {

        dx = x + 1 ;
        n += ( s[ pp + 1 ] > KEY_CODE );
        dy = y - 1 ;//
        if( dy >= 0 )
            n += ( s[ pp + 1- w ] > KEY_CODE );

        dy = y + 1 ;//
        if( dy < h )
            n += ( s[ pp + 1 + 2 ] > KEY_CODE );

    }
    //
    if(( y - 1 ) >= 0 )
        n += ( s[ pp - w ] > KEY_CODE );
    if(( y + 1 ) < h )
        n += ( s[ pp + w ] > KEY_CODE );

    ( *outCount )= n ;


}
//
__global__ void _gpu_problem2(unsigned char* s, unsigned char* t, unsigned int height, unsigned int width, unsigned channels , int runCount )
{
    unsigned int imageSize = height * width ;
    unsigned int i ;
    unsigned int index ;
    unsigned char* back ;
    int x , y ;
    int outCount ;
    __shared__ int k ;
    for( i = 0 ; i < runCount ; ++i )
    {
        k = 0 ;
        // 同步
        __syncthreads();
        while( true )
        {
            index = k++ ;
            if( index >= imageSize )
                break ;
            // 跑圖
            x = index % width ;
            y = index / width ;

            //
            gpu_getCount( &outCount , x , y , width , height , s );

            if(outCount == 3 )
                t[index] = 0xFF ;
            else  if(outCount != 2 )
                t[index] = 0 ;
            else
                t[index] =  s[index] ;

        }
        // 同步
        __syncthreads();
        // 交換
        k = 0 ;
        while( true )
        {
            index = k++ ;
            if( index >= imageSize )
                break ;
            // 跑圖
            x = index % width ;
            y = index / width ;

            //
            gpu_getCount( &outCount , x , y , width , height , t );

            if(outCount == 3 )
                s[index] = 0xFF ;
            else  if(outCount != 2 )
                s[index] = 0 ;
            else
                s[index] =  t[index] ;

        }

    }

}





//
void cpu_getCount( int *outCount , int x , int y , int w , int h , unsigned char* s )
{
    int dx , dy ;
    int n = 0 ;
    if(( x - 1 ) >= 0 )
    {

        dx = x - 1 ;
        n += ( s[y*w + dx ] > KEY_CODE );
        dy = y - 1 ;//
        if( dy >= 0 )
            n += ( s[dy*w + dx ] > KEY_CODE );

        dy = y + 1 ;//
        if( dy < h )
            n += ( s[dy*w + dx ] > KEY_CODE );

    }

    if(( x + 1 ) < w )
    {

        dx = x + 1 ;
        n += ( s[y*w + dx ] > KEY_CODE );
        dy = y - 1 ;//
        if( dy >= 0 )
            n += ( s[dy*w + dx ] > KEY_CODE );

        dy = y + 1 ;//
        if( dy < h )
            n += ( s[dy*w + dx ] > KEY_CODE );

    }
    //
    if(( y - 1 ) >= 0 )
        n += ( s[( y - 1 )*w + x ] > KEY_CODE );
    if(( y + 1 ) < h )
        n += ( s[( y + 1 )*w + x ] > KEY_CODE );

    ( *outCount )= n ;


}
//
void _cpu_problem2(unsigned char* s, unsigned char* t, unsigned int height, unsigned int width, unsigned channels , int runCount )
{
    unsigned int imageSize = height * width ;
    unsigned int i ;
    unsigned int index ;
    unsigned char* back ;
    int x , y ;
    int outCount ;
    int k ;

    k = 0 ;
    // 同步

#pragma omp parallel for
    for( index = 0 ; index < imageSize ; ++index )
    {
        // 跑圖
        x = index % width ;
        y = index / width ;

        //
        cpu_getCount( &outCount , x , y , width , height , s );

        if(outCount == 3 )
            t[index] = 0xFF ;
        else  if(outCount != 2 )
            t[index] = 0 ;
        else
            t[index] =  s[index] ;

    }

    // 交換
    k = 0 ;
#pragma omp parallel for
    for( index = 0 ; index < imageSize ; ++index )
    {
        // 跑圖
        x = index % width ;
        y = index / width ;

        //
        cpu_getCount( &outCount , x , y , width , height , t );

        if(outCount == 3 )
            s[index] = 0xFF ;
        else  if(outCount != 2 )
            s[index] = 0 ;
        else
            s[index] = t[index] ;

    }



}
int main(int argc, char** argv)
{
    assert(argc == 5);

    int runCount = atoi( argv[3] ) ;
    int isGPU = atoi( argv[4] ) ;
    unsigned int height, width, channels;
    unsigned char* src_img = NULL;
    int k ;

    read_png(argv[1], &src_img, &height, &width, &channels);
    printf( "%s %d %d %d %d %d\n" , argv[1] , height , width , channels , runCount , isGPU );
  //  assert(channels == 3);

    unsigned int memorySize = height * width * channels * sizeof(unsigned char) ;
    unsigned char* dst_img =
        (unsigned char*)malloc( memorySize );

    if(src_img == NULL )
        printf( "src_img == NULL \n" , memorySize );


    // 指定 gpu
//    cudaSetDevice(1);
    std::clock_t start = std::clock();

    /* Your algorithm here */


    printf( "memory size: %d\n" , memorySize );
    printf( " -> 1 \n" );
    if( isGPU )
    {
        // GPU
        unsigned char *gpu_Input ;
        printf( " -> 1 -1\n" );
        cudaMalloc(&gpu_Input, memorySize);
        if( gpu_Input == NULL )
            printf( " -> 1 -1-err\n" );
        // copy 到 gpu
        printf( " -> 1 -2\n" );
        cudaMemcpy(gpu_Input, src_img, memorySize, cudaMemcpyHostToDevice);

        printf( " -> 1 -3 \n" );

        unsigned char *gpu_Output ;
        cudaMalloc(&gpu_Output, memorySize);
        printf( " -> 2 \n" );


        dim3 gpuGrid(
                1
                ,1,1);  //
        const int gridSize = 16 ;
        dim3 gpuBlock(gridSize,gridSize,1);

        //  printf( "\t 2 start gpu: %d %d %d \n" , n , planet , asteroid );
        //   _gpu_problem2<<<gpuGrid,gpuBlock,0, s0>>>
        printf( " -> 3 \n" );
        _gpu_problem2<<<1,64,0>>>(gpu_Input , gpu_Output ,  height, width, channels , runCount );
        printf( " -> 4 \n" );
        // 執行
      //  GPUSobel<<<threadCount,threadCount>>>(gpu_Input , gpu_Output ,  height, width, channels , threadCount );
      //  sobel(src_img, dst_img, height, width, channels);
        // 等待完成
        cudaDeviceSynchronize();
        // 完成， gpu  copy 回來
        cudaMemcpy(dst_img, gpu_Output, memorySize, cudaMemcpyDeviceToHost);
        //
        cudaDeviceSynchronize();
        printf( " -> 5 \n" );


        cudaFree(gpu_Input);
        cudaFree(gpu_Output);
        printf( " -> 6 \n" );

    }else //
    {
        for( k = 0 ; k < runCount ; ++k )
            _cpu_problem2(src_img , dst_img ,  height, width, channels , runCount );
    }
    double duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout<<"runTime: "<< duration <<'\n';
  //  sobel(src_img, dst_img, height, width, channels);
    write_png(argv[2], dst_img, height, width, channels);


    //
    free(src_img);
    free(dst_img);

    return 0;
}
