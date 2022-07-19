
/*

 *       這個有三個版本
 *       1. threadRun： 初版，用 thread 建立後，進來一次取一行去算
 *       2. __threadRun : MPI 版，算出幾台能處理後，每台平均分配運算的數量，有用 MP
 *       3. __threadRun3 : MPI + Thread 版，也就是 1 與 2 的合體，有用 MP
 *                         MPI 後但 thread 開不起來，所以沒用
 *

 */
// 這裡可以設定是要用 OPEN_MPI (oj) 還是 openMP

#define _OPEN_MPI 1


#define OJ_RUN 1


#include <omp.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <lodepng.h>
#include <unistd.h>

#include <mpi.h>

#include <omp.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <lodepng.h>
#include <unistd.h>


#define GLM_FORCE_SWIZZLE  // vec3.xyz(), vec3.xyx() ...ect, these are called "Swizzle".
// https://glm.g-truc.net/0.9.1/api/a00002.html
//
#include <glm/glm.hpp>
// for the usage of glm functions
// please refer to the document: http://glm.g-truc.net/0.9.9/api/a00143.html
// or you can search on google with typing "glsl xxx"
// xxx is function name (eg. glsl clamp, glsl smoothstep)


#define SEND_INT 99

#define pi 3.1415926535897932384626433832795

typedef glm::dvec2 vec2;  // doube precision 2D vector (x, y) or (u, v)
typedef glm::dvec3 vec3;  // 3D vector (x, y, z) or (r, g, b)
typedef glm::dvec4 vec4;  // 4D vector (x, y, z, w)
typedef glm::dmat3 mat3;  // 3x3 matrix

unsigned int num_threads;  // number of thread
unsigned int width;        // image width
unsigned int height;       // image height
vec2 iResolution;          // just for convenience of calculation

static const int AA = 2;  // anti-aliasing

static const double power = 8.0;           // the power of the mandelbulb equation
static const double md_iter = 24;          // the iteration count of the mandelbulb
static const double ray_step = 10000;      // maximum step of ray marching
static const double shadow_step = 1500;    // maximum step of shadow casting
static const double step_limiter = 0.2;    // the limit of each step length
static const double ray_multiplier = 0.1;  // prevent over-shooting, lower value for higher quality
static const double bailout = 2.0;         // escape radius
static const double eps = 0.0005;          // precision
static const double FOV = 1.5;             // fov ~66deg
static const double far_plane = 100.;      // scene depth

vec3 camera_pos;  // camera position in 3D space (x, y, z)
vec3 target_pos;  // target position in 3D space (x, y, z)

unsigned char* raw_image;  // 1D image
unsigned char** image;     // 2D image

unsigned lodepng_encode32_file(const char* filename, const unsigned char* image, unsigned w, unsigned h);
// save raw_image to PNG file
void write_png(const char* filename) {

    unsigned error = lodepng_encode32_file(filename, raw_image, width, height);

    if (error) printf("png error %u: %s\n", error, lodepng_error_text(error));

}

// mandelbulb distance function (DE)
// v = v^8 + c
// p: current position
// trap: for orbit trap coloring : https://en.wikipedia.org/wiki/Orbit_trap
// return: minimum distance to the mandelbulb surface
/*
using namespace std ;
hash_map<double, double> _hash_Cos ;
double getCos( double c )
{
    double ret ;
    hash_map<double, double>::iterator it = _hash_Cos.find( c );
    if (it == _hash_Cos.end()){
        ret = cos( c );
        _hash_Cos.insert( c , ret );
      //  _hash_Cos[c] = ret ;
    }else
    {
        ret = it->second ;
    }
}*/

double md(vec3 p, double& trap) {
    vec3 v = p;
    double dr = 1.;             // |v'|
    double r = glm::length(v);  // r = |v| = sqrt(x^2 + y^2 + z^2)
    trap = r;

    for (int i = 0; i < md_iter; ++i) {
        double theta = glm::atan(v.y, v.x) * power;
        double phi = glm::asin(v.z / r) * power;
        double phiCos = cos(phi) ;
        dr = power * glm::pow(r, power - 1.) * dr + 1.;
        v = p + glm::pow(r, power) *
                vec3(cos(theta) * phiCos, phiCos * sin(theta), -sin(phi));  // update vk+1

        // orbit trap for coloring
        trap = glm::min(trap, r);

        r = glm::length(v);      // update r
        if (r > bailout) break;  // if escaped
    }
    return 0.5 * log(r) * r / dr;  // mandelbulb's DE function
}

// scene mapping
double map(vec3 p, double& trap, int& ID) {
    static const vec2 rt = vec2(cos(pi / 2.), sin(pi / 2.));
    static const mat3 rpBase = mat3(1., 0., 0., 0., rt.x, -rt.y, 0., rt.y, rt.x);
    vec3 rp = rpBase *   p;  // rotation matrix, rotate 90 deg (pi/2) along the X-axis
    ID = 1;
    return md(rp, trap);
}

// dummy function
// becase we dont need to know the ordit trap or the object ID when we are calculating the surface
// normal
double map(vec3 p) {
    double dmy;  // dummy
    int dmy2;    // dummy2
    return map(p, dmy, dmy2);
}

// simple palette function (borrowed from Inigo Quilez)
// see: https://www.shadertoy.com/view/ll2GD3
vec3 pal(double t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * glm::cos(2. * pi * (c * t + d));
}

// second march: cast shadow
// also borrowed from Inigo Quilez
// see: http://www.iquilezles.org/www/articles/rmshadows/rmshadows.htm
double softshadow(vec3 ro, vec3 rd, double k) {
    double res = 1.0;
    double t = 0.;  // total distance
    for (int i = 0; i < shadow_step; ++i) {
        double h = map(ro + rd * t);
        res = glm::min(
                res, k * h / t);  // closer to the objects, k*h/t terms will produce darker shadow
        if (res < 0.02) return 0.02;
        t += glm::clamp(h, .001, step_limiter);  // move ray
    }
    return glm::clamp(res, .02, 1.);
}

// use gradient to calc surface normal
vec3 calcNor(vec3 p) {
    vec2 e = vec2(eps, 0.);
    return normalize(vec3(map(p + e.xyy()) - map(p - e.xyy()),  // dx
                          map(p + e.yxy()) - map(p - e.yxy()),                    // dy
                          map(p + e.yyx()) - map(p - e.yyx())                     // dz
    ));
}

// first march: find object's surface
double trace(vec3 ro, vec3 rd, double& trap, int& ID) {
    double t = 0;    // total distance
    double len = 0;  // current distance

    for (int i = 0; i < ray_step; ++i) {
        len = map(ro + rd * t, trap,
                  ID);  // get minimum distance from current ray position to the object's surface
        if (glm::abs(len) < eps || t > far_plane) break;
        t += len * ray_multiplier;
    }
    return t < far_plane
           ? t
           : -1.;  // if exceeds the far plane then return -1 which means the ray missed a shot
}

vec4 fcolFoo( int i , int j  )
{
    vec4 fcol(0.);  // final color (RGBA 0 ~ 1)

    // anti aliasing
    for (int m = 0; m < AA; ++m)
    {
        for (int n = 0; n < AA; ++n) {
            vec2 p = vec2(j, i) + vec2(m, n) / (double)AA;

            //---convert screen space coordinate to (-ap~ap, -1~1)
            // ap = aspect ratio = width/height
            vec2 uv = (-iResolution.xy() + 2. * p) / iResolution.y;
            uv.y *= -1;  // flip upside down
            //---

            //---create camera
            static const vec3 ro = camera_pos;               // ray (camera) origin
            static const vec3 ta = target_pos;               // target position
            static const vec3 cf = glm::normalize(ta - ro);  // forward vector
            static const vec3 cs =
                    glm::normalize(glm::cross(cf, vec3(0., 1., 0.)));  // right (side) vector
            static const vec3 cu = glm::normalize(glm::cross(cs, cf));          // up vector
            vec3 rd = glm::normalize(uv.x * cs + uv.y * cu + FOV * cf);  // ray direction
            //---

            //---marching
            double trap;  // orbit trap
            int objID;    // the object id intersected with
            double d = trace(ro, rd, trap, objID);
            //---

            //---lighting
            vec3 col(0.);                          // color
            static const vec3 sd = glm::normalize(camera_pos);  // sun direction (directional light)
            static const vec3 sc = vec3(1., .9, .717);          // light color
            //---

            //---coloring
            if (d < 0.) {        // miss (hit sky)
                col = vec3(0.);  // sky color (black)
            } else {
                vec3 pos = ro + rd * d;              // hit position
                vec3 nr = calcNor(pos);              // get surface normal
#if(0)
                if( nr.x == 0 )
                {
                    col = vec3(0.);  // sky color (black)
                }else
#endif
                {
                    vec3 hal = glm::normalize(sd - rd);  // blinn-phong lighting model (vector
                    // h)
                    // for more info:
                    // https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_shading_model

                    // use orbit trap to get the color
                    col = pal(trap - .4, vec3(.5), vec3(.5), vec3(1.),
                              vec3(.0, .1, .2));  // diffuse color
                    vec3 ambc = vec3(0.3);  // ambient color
                    double gloss = 32.;     // specular gloss

                    // simple blinn phong lighting model
                    double amb =
                            (0.7 + 0.3 * nr.y) *
                            (0.2 + 0.8 * glm::clamp(0.05 * log(trap), 0.0, 1.0));  // self occlution
                    double sdw = softshadow(pos + .001 * nr, sd, 16.);         // shadow
                    double dif = glm::clamp(glm::dot(sd, nr), 0., 1.) * sdw;   // diffuse
                    double spe = glm::pow(glm::clamp(glm::dot(nr, hal), 0., 1.), gloss) *
                                 dif;  // self shadow

                    vec3 lin(0.);
                    lin += ambc * (.05 + .95 * amb);  // ambient color * ambient
                    lin += sc * dif * 0.8;            // diffuse * light color * light intensity
                    col *= lin;

                    col = glm::pow(col, vec3(.7, .9, 1.));  // fake SSS (subsurface scattering)
                    col += spe * 0.8;                       // specular
                }
            }
            //---

            col = glm::clamp(glm::pow(col, vec3(.4545)), 0., 1.);  // gamma correction
            fcol += vec4(col, 1.);
        }
    }
    return fcol ;
}



double current_pixel = 0;
double total_pixel = 0 ;
// TODO: 算單一行 <- 0 ~ wi->
int runColumn( int i )
{
    int j ;
    int k = width / 2 ;
    int count = 0 ;
    // 算右邊
    // 如果有超過 5 個都是 黑色，後面就不處理
    for ( j = k ; j < width ; ++j )
    {

        if( count < 5 )
        {
            vec4 fcol = fcolFoo( i , j );
            fcol /= (double)(AA * AA);
            // convert double (0~1) to unsigned char (0~255)
            fcol *= 255.0;
            image[i][4 * j + 0] = (unsigned char)fcol.r;  // r
            image[i][4 * j + 1] = (unsigned char)fcol.g;  // g
            image[i][4 * j + 2] = (unsigned char)fcol.b;  // b

            if( image[i][4 * j + 0] == 0 )
                if( image[i][4 * j + 1] == 0 )
                    if( image[i][4 * j + 2] == 0 )
                        count ++ ;
        }else
        {
            image[i][4 * j + 0] = (unsigned char)0;  // r
            image[i][4 * j + 1] = (unsigned char)0;  // g
            image[i][4 * j + 2] = (unsigned char)0;  // b
        }
        image[i][4 * j + 3] = 255;                    // a


#if (OJ_RUN==0)
        current_pixel++;
        // print progress
        printf("rendering...%5.2lf%%\r", current_pixel / total_pixel * 100.);
        printf("(%d,%d)\n" , i , j );
#endif
    }

    // left
    count = 0 ;
    // 算左邊，
    // try 過後這個數值才能完美過，
    //#pragma omp parallel for
    for ( j = k -1 ; j >= 0 ; --j )
    {


          if( count < 100 )// 目前最佳 120
        {
            vec4 fcol = fcolFoo( i , j );
            fcol /= (double)(AA * AA);
            // convert double (0~1) to unsigned char (0~255)
            fcol *= 255.0;
            image[i][4 * j + 0] = (unsigned char)fcol.r;  // r
            image[i][4 * j + 1] = (unsigned char)fcol.g;  // g
            image[i][4 * j + 2] = (unsigned char)fcol.b;  // b

            if(( image[i][4 * j + 0] == 0 )&&
                ( image[i][4 * j + 1] == 0 )&&
                    ( image[i][4 * j + 2] == 0 ))
                        count ++ ;
        }else
        {
            image[i][4 * j + 0] = (unsigned char)0;  // r
            image[i][4 * j + 1] = (unsigned char)0;  // g
            image[i][4 * j + 2] = (unsigned char)0;  // b
        }
        image[i][4 * j + 3] = 255;                    // a

#if (OJ_RUN==0)
        current_pixel++;

        // print progress
        printf("rendering...%5.2lf%%\r", current_pixel / total_pixel * 100.);
        printf("(%d,%d)\n" , i , j );
#endif
    }
    return 0 ;
}

int rank, world_size;
pthread_mutex_t p_thread_lock ;
int _thread_run_index = 0 ;
// TODO: thread run
void *threadRun( void * data )
{
    int start ;
    int i ;
    int dest ;
    while( 1 )
    {
        pthread_mutex_lock(&p_thread_lock);
        if( _thread_run_index >= height )
        {
            pthread_mutex_unlock(&p_thread_lock);
            break ;
        }
        start = _thread_run_index ;
        _thread_run_index ++ ;
        pthread_mutex_unlock(&p_thread_lock);

//#pragma omp parallel for
     //   for ( i = start; i < dest; ++i)
        {
            runColumn( start );
        }
    }
    return NULL ;
}
// 一般的
// TODO: thread run
void *__threadRun( void * data )
{
    int i ;
    int rank = *(int*)data ;
    int count = height / world_size ;
    int start = rank * count ;
    int dest = start + count ;
    if( rank == ( world_size - 1 ))
        dest = height ;

    {
#pragma omp parallel for
        for ( i = start; i < dest; ++i)
        {
            runColumn( i );
        }
    }
    return NULL ;
}

// 一般的
// TODO: thread run
static void *__threadRun3( void * data )
{

    int i ;

  //  int rank = rank ;//*(int*)data ;
    int count = height / world_size ;
    int start = rank * count ;
    int dest = start + count ;
    if( rank == ( world_size - 1 ))
        dest = height ;

// #pragma omp parallel
    {
        {
            while( 1 )
            {
                pthread_mutex_lock(&p_thread_lock);
                if( _thread_run_index >= dest )
                {
                    pthread_mutex_unlock(&p_thread_lock);
                    break ;
                }
                start = _thread_run_index ;
                _thread_run_index ++ ;
                pthread_mutex_unlock(&p_thread_lock);

//#pragma omp parallel for
                //   for ( i = start; i < dest; ++i)

                {
                    runColumn( start );
                }
            }
        }
    }
    return NULL ;
}


// TODO: thread run
static void *__threadRun4( void * data )
{

    int i , j ;

    //  int rank = rank ;//*(int*)data ;
    int count = height / world_size ;
    int start = rank * count ;
    int dest = start + count ;
    if( rank == ( world_size - 1 ))
        dest = height ;


    for( i = start ; i < dest ; ++i )
    {
#pragma omp parallel for
        for( j = 0 ; j < width ; ++j )
        {
            vec4 fcol = fcolFoo( i , j );
            fcol /= (double)(AA * AA);
            // convert double (0~1) to unsigned char (0~255)
            fcol *= 255.0;
            image[i][4 * j + 0] = (unsigned char)fcol.r;  // r
            image[i][4 * j + 1] = (unsigned char)fcol.g;  // g
            image[i][4 * j + 2] = (unsigned char)fcol.b;  // b
            image[i][4 * j + 3] = 255;                    // a

        }
    }
    return NULL ;
}

int main(int argc, char** argv)
{
    int count ;
    rank = 0 ;
    world_size = 1 ;
    int i ;

    int start ;
    int dest ;
    int id ;
    pthread_t *threadBuf = NULL ;
    int* threadBufId ;

    pthread_mutex_init(&p_thread_lock, NULL);

    // ./source [num_threads] [x1] [y1] [z1] [x2] [y2] [z2] [width] [height] [filename]
    // num_threads: number of threads per process
    // x1 y1 z1: camera position in 3D space
    // x2 y2 z2: target position in 3D space
    // width height: image size
    // filename: filename

    assert(argc == 11);

    //---init arguments
    num_threads = atoi(argv[1]);
    camera_pos = vec3(atof(argv[2]), atof(argv[3]), atof(argv[4]));
    target_pos = vec3(atof(argv[5]), atof(argv[6]), atof(argv[7]));
    width = atoi(argv[8]);
    height = atoi(argv[9]);
    const char *fileName = argv[10] ;
    //---create image
    total_pixel = width * height;
    iResolution = vec2(width, height);
    const int imageSize = width * height * 4 ;
    const int widthSize = width * 4 ;
    raw_image = new unsigned char[imageSize];
    unsigned char* raw_image_recv = new unsigned char[imageSize];
    image = new unsigned char*[height];

    // #pragma omp parallel for
    for (int i = 0; i < height; ++i) {
        image[i] = raw_image + i * width * 4;
    }


    MPI_Init(&argc, &argv);
    int mpiThread = 0 ;
    //MPI_Init_thread(&argc, &argv , MPI_THREAD_MULTIPLE , &mpiThread );
    // TODO: 開 MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);


    // TODO: 算 rank 要處理的資料量
    if( world_size < 1 )
        world_size = 1 ;
    // printf( "%lld - " , r );

    count = height / world_size ;

    start = rank * count ;
    dest = start + count ;
    if( rank == ( world_size - 1 ))
        dest = height ;
#if(1)
//# pragma opm parallel num_threads()
  //  omp_set_num_threads( num_threads );
    // 建  thread
    //
  //  struct sched_param param_square = {.sched_priority = 99 };
   // struct sched_param param_irq = {.sched_priority = 98 };

    _thread_run_index = start ;
    pthread_attr_t ThAttr;    // pthread attribute
    pthread_attr_init(&ThAttr);
    pthread_attr_setdetachstate(&ThAttr, PTHREAD_CREATE_JOINABLE);
  //  pthread_attr_setinheritsched(&ThAttr, PTHREAD_EXPLICIT_SCHED);
  //  pthread_attr_setschedpolicy(&ThAttr, SCHED_FIFO);
  //  pthread_attr_setschedparam(&ThAttr, &param_square);

    if( num_threads > 1 )
    {
        threadBuf = new pthread_t[num_threads] ;
        threadBufId = new int[num_threads] ;
        for( i = 0 ; i < num_threads ; ++i )
        {
            threadBufId[i] = i ;
            pthread_create(&threadBuf[i] ,&ThAttr, __threadRun3,(void*)&threadBufId[i] );
        }

        // 本機算
        i = 0 ;
        __threadRun3((void*) &i );
        // 等待結束
      //  if( num_threads > 1 )

        for( i = 0 ; i < num_threads ; ++i )
            pthread_join( threadBuf[i] , NULL );

    }else
    {
    }
#endif
#if(0)// openMP 秒數壓不下來
    omp_set_nested(1);
    omp_set_num_threads (num_threads);
#pragma omp parallel
    {
        i = omp_get_thread_num();
        __threadRun4((void*) &i );

    }
#endif
    //-------------------------------
    MPI_Status status;
    if( rank == 0 )
    {

        // TODO: rank0 ，接收並存檔
        for( i = 1 ; i < world_size ; ++i )
        {
            // receive operation result
            // MPI_Recv(資料, buffer, 資料大小, 格式, 從那個核心丟來的, 格式, 預設, 目前狀況)

            MPI_Recv( raw_image_recv , imageSize , MPI_CHAR , i,
                      SEND_INT, MPI_COMM_WORLD, &status );
            // 發送一個告訴老完成(已免當機
            MPI_Send( &i , sizeof(i) , MPI_CHAR, i,
                      SEND_INT, MPI_COMM_WORLD);

            start = i * count ;
            dest = start + count ;
            if( i == ( world_size - 1 ))
                dest = height ;
            // 放到完成的位置
            memcpy( raw_image + start * widthSize ,
                    raw_image_recv + start * widthSize ,
                    ( dest - start ) * widthSize );

            /*
            start = i * count ;
            if( i == ( world_size - 1 ))
                dest = height ;
            MPI_Recv( raw_image + start * widthSize , ( dest - start ) * widthSize , MPI_CHAR , i,
                      SEND_INT, MPI_COMM_WORLD, &status );*/
        }

        // save
        //---saving image
        // 存檔
        printf( "saving image\n" );
        write_png( fileName );
    }else
    {
        /*
        MPI_Send(  raw_image + start * widthSize , ( dest - start ) * widthSize , MPI_CHAR, 0,
                  SEND_INT, MPI_COMM_WORLD);*/
        // output the result of the operation
        // MPI_Send(資料, 資料的大小, 資料的格式(1byte), 丟到那一個核心, 訊息代碼, 預設值)
        // TODO: 其他的 rank ，將資料丟到 rank 0

        MPI_Send( raw_image , imageSize , MPI_CHAR, 0,
                  SEND_INT, MPI_COMM_WORLD);
        // 接收確定完成
        MPI_Recv( &i , sizeof(i) , MPI_CHAR , 0,
                  SEND_INT, MPI_COMM_WORLD, &status );
    }

    //---

    //---finalize
    delete[] raw_image;
    delete[] image;
    delete[] raw_image_recv ;
    //---

    MPI_Finalize();
    return 0;
}

