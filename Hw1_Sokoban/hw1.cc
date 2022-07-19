
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include <time.h>

#include <pthread.h>
#include <unistd.h>

// #include <windows.h>
#define _M_DEBUG_ 0

#define PAUSE system("pause" );

//
#define DYDX_COUNT 4
typedef struct _DYDX
{
    char key;
    char dx;
    char dy;
} DYDX;
/*
'W': (-1, 0),
'A': (0, -1),
'S': (1, 0),
'D': (0, 1),
*/
static const DYDX _dady[DYDX_COUNT] =
        {
                { 'W' ,  0, -1 } ,
                { 'A' , -1 , 0 } ,
                { 'S' ,  0 , 1 } ,
                { 'D' ,  1 , 0 } ,
           };
//
// each time map structure
typedef struct _MAP
{
    int x , y;
    int heigh; // MAP Width , Height
    int width;
    int key; // directions 
    int level ; // depth
  //  int newsId ;// direction id
    char* buf; // char length
    char** map;// map
    struct _MAP* from;
    struct _MAP* to ;
    struct _MAP* findStack ; // HashMap linket

    struct _MAP* next; // all linket

    size_t hashCode ; // HashMap code
}MAP , * LPMAP;


// buffer size
#define STACK_SIZE 100

/*
#define STACK_SIZE 20
#define STACK_MODE 8
LPMAP _findStack[STACK_SIZE][STACK_SIZE][STACK_MODE] ;*/

// HashMap
#define _HASH_MAP_SIZE 20000000
LPMAP _hashMap[_HASH_MAP_SIZE] ;

// for accelerate
static char _checkBuff[STACK_SIZE][STACK_SIZE] ;
// num of exe for debug 
static int map_count = 0;


// end or not
int is_solved( LPMAP m )
{
//	return m->notAss >
    return NULL == strchr( m->map[0] , 'x' );
}

// not use
size_t strchrcount( const char *str , const char c )
{
    size_t ret = 0 ;
    while( str[0] != 0 )
    {
        if( str[0] == c )
            ret ++ ;
        str ++ ;
    }
    return ret ;
}

// find the start position
int mapInitPos( LPMAP map )
{
    char* buf = map->buf;
    int i;
    // 找 x , y
    buf = strchr( map->buf , 'o' );
    if( buf == NULL )
        buf = strchr( map->buf , 'O' );
    if( buf == NULL )
        return 0;
    i = buf - map->buf;
    map->x = i % map->width;
    map->y = i / map->width;
    //
    return 1;

};

// check in map or not
int ptInMap( int x , int y , int width , int height )
{
    if( x >= 0 ) if( y >= 0 )
            if( x < width ) if( y < height )
                    return 1;
    return 0;
}
//
// from a position to a starting point,
// and connect each path
LPMAP toStart( LPMAP run )
{
    if( run == NULL )
        return NULL ;
    run->to = NULL ;
    while( run )
    {
        if( run->from == NULL )
            return run ;
        run->from->to = run ;
        run = run->from ;

    }
    return NULL ;
}

// check this direction movable
int checkNewsW( char ** mm , int x , int y , int width )
{
    static const char *ccStr = "#xX" ;
    static const char cc = '#';
    static const char dd = 'x';
    const char ss = mm[y][x] == dd ;
    int a = 0 ;
    int b = 0 ;
    int k = 0 ;
    int i ;
    //
    /*
    if( x == 3 )
        if( y == 1 )
            if( mm[y][x+1] == 'X' )
            if( mm[y][x] == 'x' )
            {
                a = 0 ;
                k = 1 ;
            }*/
    if(0)
        if( y == 1 ) 
        {
            a = 0 ;
            for( i = 0 ; i < width ; ++i )
                if( mm[1][i] == 'x' )
                    a ++ ;
                else if( mm[1][i] == 'X' )
                    a++;
            if( a >=2 )
                return 0 ;

        }

    if( strchr( ccStr , mm[y - 1][x - 1] ) != NULL )
        if( strchr( ccStr , mm[y][x-1] ) != NULL )
            if( strchr( ccStr , mm[y-1][x] ) != NULL )
            {
                if( ss ) return 0 ;
                if( mm[y - 1][x - 1] == dd )
                    return 0;
                if( mm[y    ][x - 1] == dd )
                    return 0;
                if( mm[y - 1][x   ] == dd )
                    return 0;
                return 1 ;
            }

    if( strchr( ccStr , mm[y - 1][x + 1] ) != NULL )
        if( strchr( ccStr , mm[y][x + 1] ) != NULL )
            if( strchr( ccStr , mm[y - 1][x] ) != NULL )
            {
                if( ss ) return 0;
                if( mm[y - 1][x + 1] == dd )
                    return 0;
                if( mm[y    ][x + 1] == dd )
                    return 0;
                if( mm[y - 1][x    ] == dd )
                    return 0;
                return 1;
            }
    if( strchr( ccStr , mm[y + 1][x - 1] ) != NULL )
        if( strchr( ccStr , mm[y][x - 1] ) != NULL )
            if( strchr( ccStr , mm[y + 1][x] ) != NULL )
            {
                if( ss ) return 0;
                if( mm[y + 1][x - 1] == dd )
                    return 0;
                if( mm[y    ][x - 1] == dd )
                    return 0;
                if( mm[y + 1][x    ] == dd )
                    return 0;
                return 1;
            }
    if( strchr( ccStr , mm[y + 1][x + 1] ) != NULL )
        if( strchr( ccStr , mm[y][x + 1] ) != NULL )
            if( strchr( ccStr , mm[y + 1][x] ) != NULL )
            {
                if( ss ) return 0;
                if( mm[y + 1][x + 1] == dd )
                    return 0;
                if( mm[y    ][x + 1] == dd )
                    return 0;
                if( mm[y + 1][x    ] == dd )
                    return 0;
                return 1;
            }
    // ok
    if( mm[y][x] == 'X' )
        return 1 ;
    //
    a = b = 0 ;
    if( mm[y][x-1] == cc )
        a = 1 ;
    else if( mm[y][x+1] == cc )
        a = 1;

    if( mm[y-1][x] == cc )
        b = 1 ;
    else if( mm[y+1][x] == cc )
        b = 1 ;

    return ( a + b) <= 1 ;
}
//
// check whether the eight point around center are full
int isD4( char ** m , int x , int y , int w , int h )
{
    static const int nx[] = { 0 , 1 , 1 , 1 , 0 , -1 , -1 , -1  } ;
    static const int ny[] = { 1 , 1 , 0 , -1 , -1 , -1 , 0 , 1 } ;
    int i ;
    for( i = 0 ; i < 8 ; ++i )
        if( ptInMap( x + nx[i] , y + ny[i]  , w , h ))
            if( strchr( ". " , m[y + ny[i]][x + nx[i]] ) != NULL )
                return 1 ;
    return 0 ;
}

// Is it possible to move to x,y
int try_move( LPMAP outMap , LPMAP map , int x , int y , int dx , int dy )
{
    const int w = map->width;
    const int h = map->heigh;
    char** m = map->map;
    char** n = outMap->map;
    const int yy = y + dy;
    const int yyy = yy + dy;
    const int xx = x + dx;
    const int xxx = xx + dx;



    if( 0 == ptInMap( x , y , w , h ) )
        return 0;
    if( 0 == ptInMap( xx , yy , w , h ) )
        return 0;
    const char cc = m[yy][xx];
    char ccc;//= m[yyy][xxx] ;
    //
    if( cc == ' ' )
        n[yy][xx] = 'o';
    else if( cc == '.' )
        n[yy][xx] = 'O';
    else if( cc == '@' )
        n[yy][xx] = '!';
    else if( ( ( cc == 'x' ) || ( cc == 'X' ) ) )
    {
        if( ptInMap( xxx , yyy , w , h ) == 0 )
            return 0;
        ccc = m[yyy][xxx];
        if( ( ccc == ' ' ) || ( ccc == '.' ) )
        {
            if( _checkBuff[yyy][xxx] )
                return 0;

            const char nxy2 = n[yy][xx] ;
            const char nxy3 = n[yyy][xxx] ;
            //
            if( m[yy][xx] == 'x' )
                n[yy][xx] = 'o';
            else
                n[yy][xx] = 'O';
            //
            if( m[yyy][xxx] == ' ' )
            {
                n[yyy][xxx] = 'x';

                // d4
             //   if( isD4( n , xxx + dx , yyy + dy , w , y ) == 0 )
             //       return -1 ;

            } else
                n[yyy][xxx] = 'X';

            if( checkNewsW( n , xxx , yyy , map->width ) == 0 )
            {
                n[yy][xx] = nxy2 ;
                n[yyy][xxx] = nxy3 ;
                return 0 ;
                if(0)		printf( "\n%d, %d:  %c\n%s\n\n", xxx , yyy , n[yyy][xxx] , outMap->buf );
                return -1;
            }

            // check error

            /*
        //	if( yyy == 1 ) if( xxx == 3 )
                if( n[1][4] == 'X' )
                    if( n[1][3] == 'x' )
                        checkNewsW( n , xxx , yyy );
*/
        } else
        {
            return 0;
        }
    } else
        return 0;

    if( m[y][x] == 'o' )
        n[y][x] = ' ';
    else if( m[y][x] == '!' )
        n[y][x] = '@';
    else
        n[y][x] = '.';
    /*
    if( map_count >= 5282 )
        if( n[1][4] == 'X' )
        if( n[1][3] == 'x' )
            checkNewsW( n , xxx , yyy );
            */
    return 1;
}

// add to queue 
LPMAP addMap( LPMAP from , LPMAP list , LPMAP mapData )
{
    int i , size;
    LPMAP ret = (LPMAP) malloc( sizeof( MAP ) );
    memcpy( ret , mapData , sizeof( MAP ) );
    map_count ++ ;
    if( from != NULL )
    {
        ret->next = from->next ;
        from->next = ret ;
    }else
        ret->next = NULL ;
     ret->from = from;
  //  if( list != NULL )
  //      list->next = ret;
    // char
    size = strlen( mapData->buf );
    ret->buf = (char*) malloc( size + 1 );
    strcpy( ret->buf , mapData->buf );
    ret->map = (char**) malloc( sizeof( char* ) * ret->heigh );
    for( i = 0; i < ret->heigh; ++i )
        ret->map[i] = ret->buf + ret->width * i ;( mapData->map[i] - mapData->buf );
    return ret;

};

// Remove the memory
// don't use it for speedup
LPMAP freeMap( LPMAP root )
{
    return NULL ;
    LPMAP next;
    while( root != NULL )
    {
        next = root->next;
        free( root->buf );
        free( root->map );
        free( root );
        root = next;
    }
    return NULL;
}

// hashMap code
size_t hashMapCode( const char *str , int count )
{
    size_t ret = 0 ;
 //   while( str[0] != 0 )
    while( count > 0 )
    {
        ret = ret * 31 + str[0] ;
        str ++ ;
        count -- ;
    }
    return ret % _HASH_MAP_SIZE ;
}

// check map for duplicates
// Calculate with HashMap
int checkMap( LPMAP root , LPMAP newData , int isAdd )
{

    if( isAdd == 0 )
    {
        newData->hashCode = hashMapCode( newData->map[0] , ( newData->heigh -1 ) * newData->width  );
        root = _hashMap[newData->hashCode] ;
        /*
        root = _findStack[x][y][c] ;*/
        /*
        if( root->x != newData->x )
            return 0 ;
        if( root->y != newData->y )
            return 0 ;*/
        while( root )
        {
            if( 0 == strcmp( root->map[0] , newData->map[0] ) )
                return 0;
            root = root->findStack;
        }
        return 1 ;
    }else
    {
        newData->findStack = _hashMap[newData->hashCode] ;
        _hashMap[newData->hashCode] = newData ;
        /*
        newData->findStack = _findStack[x][y][c] ;
        _findStack[x][y][c] = newData ;*/
    }
    return 1;
}

// Load file
LPMAP loadFile( const char* fileName )
{
    int i;
    char* buf;
    LPMAP ret;
    size_t size;
    char* list[1024];
  //  char writeFile[1024] ;

  //  sprintf( writeFile , "%s.txt" , fileName );
  //  FILE* savefile = fopen( writeFile , "w" );
    FILE* file = fopen( fileName , "r" );
    if( file == NULL )
        return NULL;
    fseek( file , 0 , SEEK_END );
    size = ftell( file );
    fseek( file , 0 , SEEK_SET );
    buf = (char*) malloc( size + 1 );

    buf[size] = 0;
    fread( buf , size , 1 , file );
    fclose( file );

/*
    fwrite( buf , size , 1 , savefile );
    fclose( savefile );
*/

    ret = (LPMAP) malloc( sizeof( MAP ) );
    memset( ret , 0 , sizeof( MAP ) );
    ret->buf = buf;
    // create
    for( i = 0; ( buf != NULL ); ++i )
    {
        if( buf[0] == 0 )
            break;
        list[i] = buf;
        buf = strchr( buf , '\n' );

   //     fwrite( buf , strlen( buf ) , 1 , writeFilesavefilesavefile );
        buf++;
    }
    //
    ret->heigh = i;
    if( i > 1 )
        ret->width = list[1] - list[0];
    else
        ret->width = strlen( list[0] );
    ret->map = (char**) malloc( sizeof( char* ) * i );
    memcpy( ret->map , list , sizeof( char* ) * i );
    // 找 x , y
    buf = strchr( ret->buf , 'o' );
    if( buf == NULL )
        buf = strchr( ret->buf , 'O' );
    if( buf == NULL )
        buf = strchr( ret->buf , '!' );
    if( buf == NULL )
    {
        freeMap( ret );
        return NULL;
    }
    i = buf - ret->buf;
    ret->x = i % ret->width;
    ret->y = i / ret->width;
    //
    mapInitPos( ret );
    //
  //  ret->map[ret->heigh-1][0] = 0 ;
    //
    return ret;
}

// Initial marked area
void initCheckRunMakeR( LPMAP run , int row )
{
    static const char *check = "# \n" ;
    int c ;
    if( row < 0 )return ;
    if( row >= run->heigh )return ;
    for( c = 0 ; c < run->width ; ++c )
        if( NULL == strchr( check , run->map[row][c] ))
            return ;
    for( c = 0; c < run->width; ++c )
        if( _checkBuff[row][c] == 0 )
            _checkBuff[row][c] = 'B';

}

// Initial marked area
void initCheckRunMakeC( LPMAP run , int col )
{
    static const char* check = "# \n";
    int r;
    if( col < 0 )return;
    if( col >= run->width )return;
    for( r = 0; r < run->heigh; ++r )
        if( NULL == strchr( check , run->map[r][col] ) )
            return;
    for( r = 0; r < run->heigh; ++r )
        if( _checkBuff[r][col] == 0 )
            _checkBuff[r][col] = 'B' ;

}

// catch the flags of map 
char mapGetFlags( LPMAP run , int x , int y )
{
    if( ptInMap( x , y , run->width , run->heigh ))
        return run->map[y][x] ;
    return 0 ;
}

// Initial marked area
void init2Pos( LPMAP run , int x , int y )
{
    static const char cc = '#' ;
    if( mapGetFlags( run , x - 1 , y  ) == cc )
        if( mapGetFlags( run , x , y - 1 ) == cc )
        {
            _checkBuff[y][x] = 'R';
            return ;
        }

    if( mapGetFlags( run , x + 1 , y ) == cc )
        if( mapGetFlags( run , x , y + 1 ) == cc )
        {
            _checkBuff[y][x] = 'R';
            return;
        }


    if( mapGetFlags( run , x + 1 , y ) == cc )
        if( mapGetFlags( run , x , y - 1 ) == cc )
        {
            _checkBuff[y][x] = 'R';
            return;
        }

    if( mapGetFlags( run , x - 1 , y ) == cc )
        if( mapGetFlags( run , x , y + 1 ) == cc )
        {
            _checkBuff[y][x] = 'R' ;
            return;
        }


}
// Initial marked area
void initCheckRun(LPMAP run)
{
    int rr , cc ;
    int r , c;
    int n ;
    const char *buf ;
    memset( _checkBuff , 0 , sizeof( _checkBuff ));
    for( r = 0 ; r < run->heigh ; ++r )
    {
        buf = run->map[r] ;
        n = 0 ;
        for( c = 0 ; c < run->width ; ++c )
        {
            if( buf[c] == '#' )
            {
                ++n ;
                _checkBuff[r][c] = '#' ;
            }else if( buf[c] == ' ' )
                init2Pos( run , c , r );
            else if( buf[c] == 'o' )
                ++n ;
        }
        // mark
        n ++ ;
        if( n == run->width )
        {
            initCheckRunMakeR( run , r-1 );
            initCheckRunMakeR( run , r+1 );
        }
    }

    // col
    for( c = 0; c < run->width; ++c )
    {
        n = 0;
        for( r = 0; r < run->heigh; ++r )
        {

            switch( run->map[r][c] )
            {
                case '#' :
                case 'o' :
                    ++n ;
            }
        }
        // mark
        if( n == run->heigh )
        {
            initCheckRunMakeC( run , c - 1 );
            initCheckRunMakeC( run , c + 1 );
        }
    }

    if( _M_DEBUG_ == 0 )
        return ;
        //
    printf( "build: \n" );
#if !defined(_DEBUG)
    return ;
#endif

    // test out
    printf( "\n\n" );
    printf( run->buf );
    printf( "\n\n" );
    for( r = 0; r < run->heigh; ++r )
    {
        buf = _checkBuff[r];
        for( c = 0; c < run->width; ++c )
        {
            if( buf[c] != 0 )
                printf( "%c" , buf[c] );
            else
                printf( " " );
        }
        printf( "\n" );
    }
    printf( "\n\n" );


}

// execute
int _run( char *outStr , size_t *outTime , const char *fileName )
{
    //size_t time = timeGetTime();
    LPMAP root , next ;
    LPMAP run , from = NULL ;
    LPMAP typeMap = NULL;
    int i , k ;
    int isWin = 0;
    int ret ;

    //memset( _findStack , 0 , sizeof( _findStack ));
    if( _M_DEBUG_ ) printf( "\n\n" );
    if( _M_DEBUG_ ) printf( "fileName:%s\n" , fileName );
    memset( _hashMap , 0 , sizeof( _hashMap ));
    root = next = 0;
    run = root = next = loadFile( fileName );
    if( run == NULL )
    {
        printf( "file error\n" );
        return 0 ;
    }
    // mark
    initCheckRun( run );
    if( _M_DEBUG_ ) printf( "\nstart: \n" );

    // build temp storage
    typeMap = addMap( NULL , NULL , run );//
    //
    int isCopy ;
    // execute
    int runAA = 0 ;
    // fint the queue
    while( run )
    {
#if defined(_DEBUG)
        runAA++;
		if(( runAA % 1000 )== 0 )
		{
		//	runAA  = run->level ;
			printf( "%d. %d\n" , run->level , runAA );
			printf( run->buf );
			printf( "\n\n" );
		}
#else
        if( _M_DEBUG_ )
            if( runAA != run->level )
        {
            printf( "%d. %d\n" , run->level , map_count );
            runAA = run->level;
        }
#endif
        //
        isCopy = 1 ;
        // Check if the four directions of the location are OK
     //   i = typeMap->newsId ;
        for( i = 0 ; i < DYDX_COUNT; ++i )
        {/*
            ++i ;
            if( i >= DYDX_COUNT )
                i = 0 ;*/
            if( isCopy )
            {
                strcpy( typeMap->buf , run->buf );
                isCopy = 0 ;
            }
            ret = try_move( typeMap , run , run->x , run->y , _dady[i].dx , _dady[i].dy ) ;
            if( ret < 0 )
                isCopy = 1 ;
            if( ret == 1 )
            {
                isCopy = 1 ;
                typeMap->x = run->x + _dady[i].dx;
                typeMap->y = run->y + _dady[i].dy;

                if( checkMap( root , typeMap , 0 ) )
                {
                    // direction
                    typeMap->key = _dady[i].key;
                    //
                    typeMap->level = run->level + 1 ;
                    //
            //        typeMap->newsId = i ;
                    // ans
                    if( is_solved( typeMap ) )
                    {
                        typeMap->from = run;
                        isWin = 1;
                        break;
                    }
                    // Add this position, the bug is the last of this linket, not the last
                    next = addMap( run , next , typeMap );
                    checkMap( root , next , 1 );
                }
            }
        }
        // end
        if( isWin )
            break;
        //
        run = run->next;
    }

    // output
    if( isWin )
    {
        from = run = toStart( typeMap );
        while( run != NULL )
        {
            if( run->key != 0 )
            {
                printf( "%c" ,  run->key );
                if( _M_DEBUG_ )
                {
                    *outStr = run->key ;
                    outStr ++ ;
                }
            }
            run = run->to;
        }



    } else
    {
        printf( "no solution" );
    }

    if( isWin )if( _M_DEBUG_ )
        {

        run = from ;
        while( run != NULL )
        {
            printf( "\n\n%d: %c\n" , run->level , run->key );
            printf( run->buf );
            run = run->to ;
        }
    }

    // final
    *outStr = 0 ;
    //
    freeMap( root );
    freeMap( typeMap );
    //
    printf( "\n" );

    //printf( "runTime: %d\n" , *outTime = timeGetTime() - time );
   // printf( "map count: %d\n" , map_count );
    //
//	PAUSE;
    return 0;
}

void* _threadsRun( void *pp)
{
    int id ;
    char ch[256] ;
    char strBuf[22][1024] = {0} ;

    size_t runTime[22] = {0} ;
    _run( strBuf[0] , runTime  , (const char *)pp );

    return 0 ;
}

int main( int args , const char *argc[] )
{
    if( 1 )
    {
#if(1)
        _threadsRun( (void*)argc[1] );
#else
        // thread
        pthread_t t; 

        if( _M_DEBUG_ ) printf( "begin thread..\n" );
        pthread_create(&t, NULL, _threadsRun, (void*)argc[1] ); 

        pthread_join(t, NULL); // Wait for the child thread to finish executing
        if( _M_DEBUG_ ) printf( "end thread\n" );
#endif
        return 0 ;
    }

    // TEST TEST TEST
    int id ;
    char ch[256] = {0};
    char strBuf[22][1024] = {0} ;
    size_t runTime[22] = {0} ;

    int i;

    while( 1 )
    {

        printf( "Enter text id (1-21 , 0.all ,-1 quit): ");
        scanf( "%d" , &id );
        if(( id > 0 )&&( id <= 21 ))
        {
            sprintf( ch , "samples/%02d.txt" , id );
            i = id ;
             // thread
            pthread_t t; 

            if( _M_DEBUG_ ) printf( "begin thread..\n" );
            pthread_create(&t, NULL, _threadsRun, ch ); 

            pthread_join(t, NULL); // Wait for the child thread to finish executing
            if( _M_DEBUG_ ) printf( "end thread\n" );

    //        _run( strBuf[i] , runTime + i , ch );
        }else if( id == 0 )
        {
            for( i = 1 ; i <= 21 ; ++i )
            {
                sprintf( ch , "samples/%02d.txt" , i );
                _run( strBuf[i] , runTime  + i , ch );
            }
            printf( "\n\n" );
            printf( "output list: \n" );
            for( i = 1; i <= 21; ++i )
            {
                printf( "%2d. runTime =%ld \n %s\n\n" , i , runTime[i] , strBuf[i] );
            }
        }else if( id == -1 )
            break ;
    }

    return 0 ;
}
