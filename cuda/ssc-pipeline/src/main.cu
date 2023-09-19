#include <stdio.h>
#include "ssc_pipeline.h"

typedef struct {

  float alpha;
  int size;
  
}params;


void processBlock(float *output, float *output2, float *input, void *data)  
{
  params *D = (params *) data;

  /*  
  for( int k = 0 ; k < D->size; k++)
    {
      output[k] = D->alpha * input[k];
    }
  */

}  

int main (int argc,char *argv[], char **env)
{
  unsetenv("CUDA_VISIBLE_DEVICES");
  
  char path[500], outputPath[500];
  char volOrder[100], rank[100], datasetName[100];
  char uuid[500];

  ssc_volume workspace;
  ssc_h5 H5;
  ssc_gpus *gpus;
  int *oshape;

  //////// input (without gengetopt ) /////////////
  /*
    
  strcpy( path, "/home/ABTLUS/eduardo.miqueles/test/cat.h5" );
  strcpy( outputPath, "/home/ABTLUS/eduardo.miqueles/test/");
  strcpy( volOrder, "yx"  );
  strcpy( rank, "ztyx" );
  strcpy( datasetName, "entry/data/data" );
  strcpy( uuid, "1xx666uyuio" )

  int shape[2]     = {3072, 3072};
  int ngpus        = 1;
  int gpu[1]       = {5};
  int Init         = 0;
  int Final        = 319;
  int blockSize    = 10;
  bool printTiming = true;
  bool save        = true;
  */
    
  /////////////////////////////////////////////////
  ////// Gengetopt (optional for your application)
  
  struct gengetopt_args_info arg;

  if(cmdline_parser(argc, argv, &arg)!=0)
    return EXIT_FAILURE;
  
  strcpy(path, arg.path_arg);
  strcpy(outputPath, arg.outpath_arg);
  strcpy(volOrder, arg.order_arg );
  strcpy(rank, arg.rank_arg );
  strcpy(datasetName, arg.dataset_arg );
  strcpy(uuid, arg.uuid_arg );
  
  int *shape        = arg.shape_arg;
  int ngpus         = arg.gpu_given;
  int *gpu          = arg.gpu_arg;
  int Init          = arg.initial_arg;
  int Final         = arg.final_arg;
  int blockSize     = arg.block_arg;
  bool printTiming  = arg.timing_given;
  bool save         = arg.save_given;
  
  //////////////////////////////////////
  
  gpus = ssc_configure_gpus( ngpus, gpu );

  oshape = ssc_configure_oshape( shape );
  
  ssc_create_h5info( &H5, rank, path, datasetName, volOrder);
  
  ssc_create_workspace(&workspace,
		       &H5,
		       gpus,
		       oshape,
		       blockSize,
		       outputPath,
		       uuid,
		       printTiming,
		       save,
		       Init,
		       Final);
    
  
  params data;
  data.size = blockSize * workspace.oshape[1] * workspace.oshape[0];
  data.alpha = 0.5;
  
  ssc_processing_pipeline( &workspace, &processBlock, &data );

  ssc_disp_times( &workspace, true );
   
  ssc_destroy_workspace(workspace);
  
  ssc_destroy_h5info(H5);
  
  return EXIT_SUCCESS;
}
