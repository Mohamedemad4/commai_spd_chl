# commai_spd_chl
this some code I used for the speed chl from commai. i need to clean this.

### Solution is in 
[`STACKED_CNN_2FPS_Block_approach_45bs-HE_norm-op_flow_dense-RGB-KITTI.ipynb`](https://github.com/Mohamedemad4/commai_spd_chl/blob/master/notebooks/STACKED_CNN_2FPS_Block_approach_45bs-HE_norm-op_flow_dense-RGB-KITTI.ipynb) with 53 epochs
that gets us **13.62 MSE** (the lower the better)
- Commai's internal solution got a 7.4
- [This guy](https://github.com/ryanchesler/comma-speed-challenge) got 5.4 

### Generating data_new/
- use the 2 kitti scripts they should handle everything including splicing KITTI data with data/
