/************************************************************
  
  This example shows how to write and read a hyperslab.  It 
  is derived from the h5_read.c and h5_write.c examples in 
  the "Introduction to HDF5".

 ************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include "hdf5.h"

//#define FILE        "/home/ABTLUS/eduardo.miqueles/test/cat.h5"
//#define DATASETNAME "entry/data/data" 

#define FILE        "file.h5"
#define DATASETNAME "entry/data/data"

int
main (void)
{
  hid_t file,dataset, dataspace, datatype, memspace;
  //herr_t status_hdf;
  hsize_t dims_in[4];
  //H5T_class_t class;
  //H5T_order_t order;
  size_t size, dataSize, offset;
  int rank;
  //int status_n;
  
  /*open HDF5 file*/
  file    = H5Fopen(FILE,H5F_ACC_RDONLY,H5P_DEFAULT);
    
  /*open dataset */
  dataset = H5Dopen(file,DATASETNAME,H5P_DEFAULT);
    
  /*
   * Get datatype and dataspace handles and then query
   * dataset class, order, size, rank and dimensions. */
    
  datatype = H5Dget_type(dataset);
  //class    = H5Tget_class(datatype);
  //order	   = H5Tget_order(datatype);
  offset   = H5Tget_offset(datatype);
  size 	   = H5Tget_size(datatype);
  dataSize = H5Dget_storage_size(dataset);
  
  /*get dataspace*/
  dataspace = H5Dget_space(dataset);

  /*get Number of dimensions*/
  rank      = H5Sget_simple_extent_ndims(dataspace);

  H5Sget_simple_extent_dims(dataspace,dims_in,NULL);

  fprintf(stderr,"H5 Dataset: %s\n", DATASETNAME);
  fprintf(stderr,"H5 Offset: %lu\n",offset);
  fprintf(stderr,"H5 Data size is: %lu\n",size);
  fprintf(stderr,"H5 Number of points x Data size: %lu\n",dataSize);
  fprintf(stderr,"H5 rank %d \nH5 zyx dimensions Nz=%lu / Nt=%lu / Ny=%lu / Nx=%lu\n", 
	  rank,(unsigned long)(dims_in[0]),(unsigned long)(dims_in[1]), (unsigned long)(dims_in[2]), (unsigned long)(dims_in[3]) );
    

  //-----------------------------
  
  int Nx = 3072;
  int Ny = 3072;
  int delta = 10;
  int blockSize = 10;
  
  float *temp = (float *)malloc(sizeof(float) * Nx * Ny * blockSize);  
  
  hsize_t start[4],count[4];
  hsize_t offset_out[2],count_out[2];
  hsize_t dims_out[2];

  offset = 0;
 
  for(int slice=0;slice<delta;slice++)
    {
      start[0] = 0;
      start[1] = 0;
      start[2] = 0;
      start[3] = 0;
      
      // count indicates how many elements will be selected in each dimension 
     
      count[0] = 1;
      count[1] = 1;
      count[2] = Ny;
      count[3] = Nx;
     
      // This function selects the stated hyperslab 
     
      H5Sselect_hyperslab(dataspace,H5S_SELECT_SET,start,NULL,count,NULL);
     
      // define the dimensions of the output image
     
      dims_out[0] = Ny;
      dims_out[1] = Nx;
      memspace = H5Screate_simple(2,dims_out,NULL);
     
      // define the final image offset 
      
      offset_out[0] = 0;
      offset_out[1] = 0;
     
      // count_out indicates how many elements will be transferred in each dimension 
     
      count_out [0] = Ny;
      count_out [1] = Nx;
     
      H5Sselect_hyperslab(memspace,H5S_SELECT_SET,offset_out,NULL,count_out,NULL);
     
      H5Dread(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT, temp + offset * Nx * Ny);
     
      fprintf(stderr,"-->H5Dread function (slice %d)\n",slice);
 	      
      offset++;
    }

  free(temp);
  
  H5Sclose(memspace);
  

  H5Dclose (dataset);
  H5Sclose (dataspace);
  H5Fclose (file);
  
  
}     
