#ifndef SSC_H5_H
#define SSC_H5_H

#include "../codes/ssc_workspace.h"
#include "../codes/ssc_types.h"


__host__ void ssc_create_h5info(ssc_h5 *H5,
				char *rank,
				char *path,
				char *datasetName,
				char *volOrder);

__host__ void ssc_destroy_h5info(ssc_h5 H5);

__host__ void ssc_read_block_xy(ssc_volume *workspace);

__host__ void ssc_read_block_xz(ssc_volume *workspace);


#endif // SSC_H5_H
