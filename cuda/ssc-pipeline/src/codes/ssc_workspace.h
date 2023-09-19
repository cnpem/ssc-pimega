#ifndef SSC_WORKSPACE_H
#define SSC_WORKSPACE_H

#include "ssc_types.h"

__host__ void ssc_destroy_workspace(ssc_volume workspace);

__host__ void ssc_create_workspace(ssc_volume *workspace,
				   ssc_h5 *H5,
				   ssc_gpus *gpus,
				   int *oshape,
				   int blockSize,
				   char *outputPath,
				   char *uuid,
				   bool printTiming,
				   bool save,
				   int Init,
				   int Final);

#endif // SSC_WORKSPACE_H

