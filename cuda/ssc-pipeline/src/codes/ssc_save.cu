#include <stdlib.h>
#include <math.h>
#include <hdf5.h>

#include "ssc_types.h"
#include "ssc_save.h"

extern "C" {
#include <time.h>
}

__constant__ float TOL = 0.0001f;
__constant__ float TOL2 = 0.0000001f;

__host__ void ssc_save_block(ssc_volume* workspace)
{
  if( workspace->save )
    {
      char fileName[500];
      //char fileNamei[500];
      //char fileNameo[500];
      
      char* fileNameNumberPosition;
      //char* fileNameNumberPositioni;
      //char* fileNameNumberPositiono;
      
      FILE* file;
      //FILE* filei; 
      //FILE* fileo;

      int imageSize;

      /*      
      int z, w, i, j;
      
      
      // writing input/output average of each block

       if (workspace->H5->order == SSC_XY )
	{
	  imageSize = workspace->H5->Nx * workspace->H5->Ny;
	}
      else if (workspace->H5->order == SSC_XZ )
	{
	  imageSize = workspace->H5->Nx * workspace->H5->Nz;
	}
            
      float *avgInput, *avgOutput;

      avgInput  = (float *)malloc(sizeof(float)*imageSize);
      avgOutput = (float *)malloc(sizeof(float)*imageSize); 

      for (w = 0; w < imageSize; w ++)
	{
	  i = w / workspace->H5->Nx;
	  j = w % workspace->H5->Nx;
	  
	  avgInput[  i * workspace->H5->Nx + j] = 0; 
	  avgOutput[  i * workspace->H5->Nx + j]  = 0;
	  
	  for( z = 0; z < workspace->blockSize; z ++)
	    {
	      avgInput[  i * workspace->H5->Nx + j] += workspace->volume[ z * imageSize + i * workspace->H5->Nx + j] / workspace->blockSize;
	      avgOutput[ i * workspace->H5->Nx + j] += workspace->outputwc[ z * imageSize + i * workspace->H5->Nx + j] / workspace->blockSize;
	    }
	}
      
      strcpy(fileNamei, workspace->outputPath);
      strcat(fileNamei, "/ssc_temp_avgInput_");
      strcat(fileNamei, workspace->uuid);
      strcat(fileNamei, "_");

      fileNameNumberPositioni = fileNamei + strlen(fileNamei);

      sprintf(fileNameNumberPositioni, "%d.b", workspace->nblock);
      
      filei = fopen(fileNamei, "w");
      
      fwrite( avgInput, sizeof(float), imageSize, filei);
      
      fclose(filei);

      strcpy(fileNameo, workspace->outputPath);
      strcat(fileNameo, "/ssc_temp_avgOutput_");
      strcat(fileNameo, workspace->uuid);
      strcat(fileNameo, "_");

      fileNameNumberPositiono = fileNameo + strlen(fileNameo);

      sprintf(fileNameNumberPositiono, "%d.b", workspace->nblock);
      
      fileo = fopen(fileNameo, "w");
      
      fwrite( avgOutput, sizeof(float), imageSize, fileo);
      
      fclose(fileo);

      free(avgInput);
      free(avgOutput);
      */
      
      // save output


      if (workspace->H5->order == SSC_XY )
	{
	  imageSize = workspace->oNx * workspace->oNy;
	}
      else if (workspace->H5->order == SSC_XZ )
	{
	  imageSize = workspace->oNx * workspace->oNz;
	}
       
      strcpy(fileName, workspace->outputPath);
      strcat(fileName, "/ssc_temp_");
      strcat(fileName, workspace->uuid);
      strcat(fileName, "_");

      fileNameNumberPosition = fileName + strlen(fileName);

      sprintf(fileNameNumberPosition, "%d.b", workspace->nblock);
     
      file = fopen(fileName, "w");
     
      fwrite( workspace->output, sizeof(float), workspace->blockSize * imageSize, file);
      
      fclose(file);


    }
      
}




