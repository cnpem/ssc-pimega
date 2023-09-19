#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "ssc_pimega_backend.h"

int main (int argc, char *argv[])
{
  float *raw;
  int *ix, *iy;   

  int blocksize = 1;

  // input from the backend: raw, iy, iy:
  raw    = (float *)malloc( blocksize * N540D * N540D * sizeof(float) );
  iy     = (int *)malloc( N540D * N540D * sizeof(int) );
  ix     = (int *)malloc( N540D * N540D * sizeof(int) );

  // output for visualization
  float *output = (float *)malloc( blocksize * N540D * N540D * sizeof(float) );
  
  FILE *fpx = fopen( argv[1], "rb+");
  FILE *fpy = fopen( argv[2], "rb+");
  FILE *fpr = fopen( argv[3], "rb+");

  fread(ix, sizeof(int), N540D * N540D, fpx);
  fread(iy, sizeof(int), N540D * N540D, fpy);
  fread(raw, sizeof(float), blocksize * N540D * N540D, fpr);
  
  ssc_pimega_backend_plan workspace;
    
  ssc_pimega_backend_create_plan( &workspace, blocksize, SSC_PIMEGA_540D );    

  ssc_pimega_backend_set_plan( &workspace, ix, iy);

  clock_t begin = clock();

  ssc_pimega_backend_restoration( output, raw, &workspace );
  
  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

  ssc_pimega_backend_free_plan( &workspace );    

  fprintf(stderr,"Elapsed time: %lf\n", time_spent);
  
  //check output:

  FILE *fpo = fopen( argv[4], "w");
  
  fwrite(output, sizeof(int), blocksize * N540D * N540D, fpo);
 
  fclose(fpo);

  
  free(raw);
  free(ix);
  free(iy);

  fclose(fpx);
  fclose(fpy);
  fclose(fpr);
 
  
  return EXIT_SUCCESS;
}
