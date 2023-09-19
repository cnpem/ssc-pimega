#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>
#include <math.h>

#include "../codes/ssc_types.h"
#include "../codes/ssc_macros.h"
#include "../codes/ssc_workspace.h"

#define SSC_DISP 0

extern "C" {
#include <time.h>
}

__host__ int ssc_create_h5info(ssc_h5 *H5,
			       char *rank,
			       char *path,
			       char *datasetName,
			       char *volOrder)
{
  hid_t file,dataset, dataspace, datatype;
  int rankn;
  
  file      = H5Fopen(path,H5F_ACC_RDONLY,H5P_DEFAULT);
  dataset   = H5Dopen(file,datasetName,H5P_DEFAULT);    
  datatype  = H5Dget_type(dataset);
  dataspace = H5Dget_space(dataset);
  rankn     = H5Sget_simple_extent_ndims(dataspace);

  H5Sget_simple_extent_dims(dataspace, H5->dimsin, NULL);
  
  H5->file        = file;
  H5->dataset     = dataset;
  H5->dataspace   = dataspace;
  H5->datatype    = datatype;
  H5->path        = path;
  H5->rankn       = rankn;
  H5->datasetName = datasetName;
  H5->rankS       = rank;
  H5->orderS      = volOrder;
  
  if ( strcmp( rank, "ztyx" ) == 0 )
    {
      H5->Nz = H5->dimsin[0]; 
      H5->Ny = H5->dimsin[2];
      H5->Nx = H5->dimsin[3];
    }
  else
    {
      if ( strcmp( rank, "zyx") == 0 )
	{
	  H5->Nz = H5->dimsin[0]; 
	  H5->Nx = H5->dimsin[1]; 
	  H5->Ny = H5->dimsin[2];
	}
      else
	{
	  fprintf(stderr,"\nssc-pipeline error: wrong rank model - choose 'ztyx' or 'zyx'\n\n");
	  return EXIT_FAILURE;
	}
    }
  
  
  if(strcmp(volOrder, "yx" ) == 0)
    {
      H5->order = SSC_XY;
    }
  else
    {
      if(strcmp(volOrder, "zx" ) == 0)
	{
	  H5->order = SSC_XZ;
	}
      else
	{
	  fprintf(stderr,"\nssc-pipeline error: wrong order definition - choose 'yx' or 'zx'\n\n");
	  return EXIT_FAILURE;
	}
    }
  
  return EXIT_SUCCESS;
}


__host__ void ssc_destroy_h5info(ssc_h5 H5)
{
  H5Dclose (H5.dataset);
  H5Sclose (H5.dataspace);
  H5Fclose (H5.file);
}

/*----------------------------------
  Read data from HDF file
  ---------------------------------*/ 	      

__host__ void ssc_read_block_xy(ssc_volume *workspace)
{
  if( workspace->H5->rankn == 3) // zyx
    {      
      hsize_t dims_out[2];
      hsize_t start[3],count[3];
      hsize_t offset_out[2],count_out[2];
      hid_t dataspace,memspace,dataset;
      int Nx, Ny;
      int Init  = workspace->Init;
      int Final = workspace->Final;
      int blockSize;
      
      struct timespec TimeStart, TimeEnd;

      dataspace = workspace->H5->dataspace;
      dataset   = workspace->H5->dataset;
      Nx        = workspace->H5->Nx;
      Ny        = workspace->H5->Ny;
      blockSize = workspace->blockSize;
  
      int delta  = SSC_MIN(Final,workspace->ei) - workspace->si;
      int offset = 0;

      clock_gettime(CLOCK, &TimeStart);
      
      for( int slice = 0 ; slice < delta ; slice ++ )
	{
      
	  start[0] = Init + (workspace->nblock * blockSize) + slice;
	  start[1] = 0;
	  start[2] = 0;
	  
	  count[0] = 1;
	  count[1] = Ny;
	  count[2] = Nx;
	  
	  H5Sselect_hyperslab(dataspace,H5S_SELECT_SET,start,NULL,count,NULL);
	  
	  dims_out[0] = Nx;
	  dims_out[1] = Ny;
	  memspace = H5Screate_simple(2,dims_out,NULL);
	  
	  offset_out[0] = 0;
	  offset_out[1] = 0;
	  
	  count_out[0] = Nx;
	  count_out[1] = Ny;
	  
	  H5Sselect_hyperslab(memspace,H5S_SELECT_SET,offset_out,NULL,count_out,NULL);
	  
	  H5Dread(dataset,
		  H5T_NATIVE_FLOAT,
		  memspace,
		  dataspace,
		  H5P_DEFAULT,
		  workspace->volume + offset * Nx * Ny);
	  
	  offset ++;
	}
      
      clock_gettime(CLOCK, &TimeEnd);
      if(SSC_DISP)	
	fprintf(stderr,"\n\t%lf\t\t|-->H5Dread function",TIME(TimeEnd,TimeStart));
 	      
      H5Sclose(memspace);

    }
  else
    {

      if ( workspace->H5->rankn == 4) // ztyx 
	{
	  hsize_t dims_out[2];
	  hsize_t start[4],count[4];
	  hsize_t offset_out[2],count_out[2];
	  hid_t dataspace,memspace,dataset;
	  int blockSize;
	  int Nx, Ny;
	  int Init  = workspace->Init;
	  int Final = workspace->Final;
      
	  struct timespec TimeStart, TimeEnd;
      
	  dataspace = workspace->H5->dataspace;
	  dataset   = workspace->H5->dataset;
	  Nx        = workspace->H5->Nx;
	  Ny        = workspace->H5->Ny;
	  blockSize = workspace->blockSize;
      
	  int delta  = SSC_MIN(Final,workspace->ei) - workspace->si;
	  int offset = 0;

	  clock_gettime(CLOCK, &TimeStart);
	  
	  for( int slice = 0; slice < delta ; slice ++)
	    {
	      start[0] = Init + (workspace->nblock*blockSize) + slice;
	      start[1] = 0;
	      start[2] = 0;
	      start[3] = 0;
	      
	      count[0] = 1;
	      count[1] = 1;
	      count[2] = Ny;
	      count[3] = Nx;
	      
	      H5Sselect_hyperslab(dataspace,H5S_SELECT_SET,start,NULL,count,NULL);
	      
	      dims_out[0] = Ny;
	      dims_out[1] = Nx;
	      
	      memspace = H5Screate_simple(2,dims_out,NULL);
	      
	      offset_out[0] = 0;
	      offset_out[1] = 0;
	      
	      count_out[0] = Ny;
	      count_out[1] = Nx;
	      
	      H5Sselect_hyperslab(memspace,H5S_SELECT_SET,offset_out,NULL,count_out,NULL);
	      
	      H5Dread(dataset,
		      H5T_NATIVE_FLOAT,
		      memspace,
		      dataspace,
		      H5P_DEFAULT,
		      workspace->volume + offset * Ny * Nx);

	      offset ++;
	    }
	  
	  clock_gettime(CLOCK, &TimeEnd);
	  if(SSC_DISP)	
	    fprintf(stderr,"\n\t%lf\t\t|-->H5Dread function",TIME(TimeEnd,TimeStart));

	  H5Sclose(memspace);

	}
      else
	{
	  fprintf(stderr,"\n\t\t|-->ssc-pipeline error! undefined rank number");
	}
    }
}

__host__ void ssc_read_block_xz(ssc_volume *workspace)
{
  if( workspace->H5->rankn == 3) //zyx
    {
      hsize_t dims_out[2];
      hsize_t start[3],count[3];
      hsize_t offset_out[2],count_out[2];
      hid_t dataspace,memspace,dataset;
      int Nx, Nz;
      int Init  = workspace->Init;
      int Final = workspace->Final;
      int blockSize;
      
      struct timespec TimeStart, TimeEnd;
      
      dataspace = workspace->H5->dataspace;
      dataset   = workspace->H5->dataset;
      Nz        = workspace->H5->Nz;
      Nx        = workspace->H5->Nx;
      blockSize = workspace->blockSize;
      
      int delta  = SSC_MIN(Final,workspace->ei) - workspace->si;
      int offset = 0;

      clock_gettime(CLOCK, &TimeStart);
	  
      for (int slice = 0; slice < delta ; slice ++)
	{
	  start[0] = 0;
	  start[1] = Init + (workspace->nblock*blockSize) + slice;
	  start[2] = 0;
	  
	  count[0] = Nz;
	  count[1] = 1;
	  count[2] = Nx;
	  
	  H5Sselect_hyperslab(dataspace,H5S_SELECT_SET,start,NULL,count,NULL);
	  
	  dims_out[0] = Nx;
	  dims_out[1] = Nz;
	  
	  memspace = H5Screate_simple(2,dims_out,NULL);
	  
	  offset_out[0] = 0;
	  offset_out[1] = 0;
	  
	  count_out[0] = Nx;
	  count_out[1] = Nz;
	  
	  H5Sselect_hyperslab(memspace,H5S_SELECT_SET,offset_out,NULL,count_out,NULL);
	  
	  
	  H5Dread(dataset,
		  H5T_NATIVE_FLOAT,
		  memspace,
		  dataspace,
		  H5P_DEFAULT,
		  workspace->volume + offset * Nx * Nz);
	  
	  offset ++;
	}
	  
      clock_gettime(CLOCK, &TimeEnd);
      if(SSC_DISP)	
	fprintf(stderr,"\n\t%lf\t\t|-->H5Dread function",TIME(TimeEnd,TimeStart));

      H5Sclose(memspace);
      
    }
  else
    {
      if (workspace->H5->rankn == 4)
	{

	  hsize_t dims_out[2];
	  hsize_t start[4],count[4];
	  hsize_t offset_out[2],count_out[2];
	  hid_t dataspace,memspace,dataset;
	  int Nx, Nz;
	  int Init  = workspace->Init;
	  int Final = workspace->Final;
	  int blockSize;
      
	  struct timespec TimeStart, TimeEnd;
      
	  dataspace = workspace->H5->dataspace;
	  dataset   = workspace->H5->dataset;
	  Nx        = workspace->H5->Nx;
	  Nz        = workspace->H5->Nz;
	  blockSize = workspace->blockSize;
            
	  int delta  = SSC_MIN(Final,workspace->ei) - workspace->si;
	  int offset = 0;

	  clock_gettime(CLOCK, &TimeStart);
	  
	  for ( int slice = 0; slice < delta ; slice ++)
	    {
	      start[0] = 0;
	      start[1] = 0;
	      start[2] = Init + (workspace->nblock*blockSize) + slice;
	      start[3] = 0;
      	      
	      count[0] = Nz;
	      count[1] = 1;
	      count[2] = 1;
	      count[3] = Nx;
	      
	      H5Sselect_hyperslab(dataspace,H5S_SELECT_SET,start,NULL,count,NULL);
	      
	      dims_out[0] = Nx;
	      dims_out[1] = Nz;

	      memspace = H5Screate_simple(2,dims_out,NULL);
	      
	      offset_out[0] = 0;
	      offset_out[1] = 0;
	      
	      count_out[0] = Nx;
	      count_out[1] = Nz;
	      
	      H5Sselect_hyperslab(memspace,H5S_SELECT_SET,offset_out,NULL,count_out,NULL);
      	      
	      H5Dread(dataset,
		      H5T_NATIVE_FLOAT,
		      memspace,
		      dataspace,
		      H5P_DEFAULT,
		      workspace->volume + offset * Nx * Nz);

	      offset ++;
	    }
	      
	  clock_gettime(CLOCK, &TimeEnd);
	  if(SSC_DISP)	
	    fprintf(stderr,"\n\t%lf\t\t|-->H5Dread function",TIME(TimeEnd,TimeStart));
 	      
	  H5Sclose(memspace);
	}
      else
	{
 	  fprintf(stderr,"\n\t\t|-->ssc-pipeline error! undefined rank number");
 	}
    }
}
      

