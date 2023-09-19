#include <stdio.h>
#include <stdlib.h>

typedef struct {

  int n; 
  float alpha;
  
}params;


typedef void (*ftype)(float *, float *, void *);



void pblock(float *output, float *input, void *data)  
{
  params *D = (params *) data;
  
  for( int k = 0 ; k < D->n; k++)
    {
      output[k] = D->alpha * input[k] / 2.0;
    }
}

void executor( float *output, float *input, void *data, ftype functionPointer)  
{
  params *D = (params *) data;
  
  (*functionPointer)(output, input, D);

  for(int k = 0; k < D->n; k++)
    {
      fprintf(stderr,"%lf\n",output[k]);
    }
  
}

void main( void )
{
  float input[10] = {1,2,3,4,5,6,7,8,9,10};
  float output[10] = {0,0,0,0,0,0,0,0,0,0}; 
  params D;
  
  D.n = 10;
  D.alpha = 1;
  
  executor( output, input, &D, &pblock );
}
