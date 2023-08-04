## re-implementation of iMeshSegNet, for AI research use only

## structure of dataset directory
    -- dataset
        -- 3D_scans_per_patient_obj_files_b1
            -- OEJBIPTC
                -- OEJBIPTC_lower.obj
                -- OEJBIPTC_upper.obj
            ... ...

            -- ground-truth_labels_instances_b1
                -- OEJBIPTC
                    -- OEJBIPTC_lower.json
                    -- OEJBIPTC_upper.json
            ... ...

        -- 3D_scans_per_patient_obj_files_b2
        ... ...

## h5 dataset generating
`python preprocessing.py`

## training
`python train_lit.py`

## visualization
`python mesh_viewer.py`

## inference and visualization
`python test_inf.py`

## Contribution
Contribute for the codes apart from those under `./losses_and_metrics` and `./model`, and `mesh_dataset.py`, `train.py`.

## Implement Detail
- Dataset Preparing: Put the dataset under `./dataset`
- Run `preprocessing.py` to generate .hdf5 dataset, to be more detailed:
  - **Downsampling**: `do_downsample()` will parallelly downsample all the mesh files within a list with their labels to output .vtk files into `./dataset/3D_scans_ds`. During this step, those samples who're already downsampled, distinguished by their sample name, won't be re-processed.
  - **Augmentation**: `do_augmentation()` will parallelly perform flipping and random combination of random rescaling, random translation and random rotation on all the samples in `./dataset/3D_scans_ds`
  - **Split Dataset**: `make_filelist_final()` will read samples from `./dataset/3D_scans_ds` split all the files into training set and validation set, output the training set filelist, the validation set filelist and the whole filelist into `./dataset/FileLists`
  - **Dataset Generating**: `vtk2h5` will call `gen_metadata()` to extract features from .vtk files and output a .hdf5 file, which can be sent to hpc for training once its ready. Can use cmd to compress the size of .hdf5 `h5repack -v -f GZIP=9 ./dataset/3D_scans_h5/upper.hdf5 ./dataset/3D_scans_h5/compressed_upper.hdf5`, but compressed file will take more time to read data.
- Run `train_lit.py` to start training once the config file and dataset is ready. For Hpc training use `qsub run_script.txt`.
  - It will first try to load the latest checkpoint according to the path specified in config file, restore its param dict and epoch. If fail to load that checkpoint, it will train from scratch.
  - During training procedure, it will save the checkpoint both with the best loss and best dice score by default.
- Run `test_inf.py` to predict and visualize the selected sample.
  - The inference pipeline can be customized in `test_inf()`.
  - Use `infer()` to load the model and do inference. Set the parameter `refine=True` to perform graph cut after prediction. Set `with_raw_output=True` to return the raw predict tensor from the model.
  - Call `high_resolution_restore()` to use KNN to project the prediction back to its original resolution. Can also try SVM other than KNN. This function will need the raw tensor mentioned above.