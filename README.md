# Deep Learning Pipeline for Preprocessing and Segmenting Cardiac Magnetic Resonance of Single Ventricle Patients from an Image Registry

This repository contains the code for the paper:

**Tina Yao, Nicole St. et al.** Deep Learning Pipeline for Preprocessing and Segmenting Cardiac Magnetic Resonance of Single Ventricle Patients from an Image Registry [[preprint]](https://arxiv.org/abs/2303.11676)

## Models
To download pretrained models, please download them from this [Google Drive link](https://www.example.com](https://drive.google.com/drive/folders/1df2Cf-bUgBG3KeMkaaUTp-ZE8sS-tGfK?usp=drive_link)https://drive.google.com/drive/folders/1df2Cf-bUgBG3KeMkaaUTp-ZE8sS-tGfK?usp=drive_link)

We have also added our implementation of the UNet3+ model.[1]


## Folder Structure
The pipeline is designed to work with a specific folder structure, as described below:
```
• data_path
    ○ patient1
        ‣ scan1
            ⁃ dicoms.dcm
        ‣ scan2
            ⁃ dicoms.dcm
        ‣ ...
    ○ patient2
        ‣ scan1
            ⁃ dicoms.dcm
        ‣ scan2
            ⁃ dicoms.dcm
        ‣ ...
    ○ ...
    
• results_path
    ○ segs  (segmented gifs)
        ‣ patient1.gif
        ‣ patient2.gif
    ○ volume_curves
        ‣ patient1.jpg
        ‣ patient2.jpg
    ○ sys_dia_plot (segmented images of systole and diastole)
        ‣ patient1.png
        ‣ patient2.png
    ○ ...
```

## Pipeline
Once you have the correct folder structure, to use the pipeline, you only need to use the following code:
`p = Pipeline(patient, data_path='data', results_path='results')`

Where 'patient' is the name of the folder holding a given patient's scans.

This line of code will create the pipeline object, p.

Once the pipeline object is instantiated, it will automatically process the scans for the specified patient and generate the desired results. The segmented GIFs will be stored in the segs folder, the volume curves will be saved as JPEG images in the volume_curves folder, and the segmented images of systole and diastole will be stored as PNG files in the sys_dia_plot folder.

Example implementation is also provided in example.py

## Pipeline Attributes
The pipeline will have the following attributes.
| Attribute Name                               | Data Type        | Description                                                                                                                                                                   |
|----------------------------------------------|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| patient                                      | string           | Patient Name                                                                                                                                                                  |
| series_list                                  | list             | List of Study Series Names                                                                                                                                                    |
| dicom_info (contains info on series level: ) | pandas dataframe | The information of all the dicom files stored as a dataframe with headers: sliceLocation, thickness, triggertime, N_timesteps, orientation, position, pixelspacing, and phase |
| N_timesteps                                  | int              | Number of phases in the Short-Axis cine stack                                                                                                                                 |
| N_slices                                     | int              | Number of 2D slices in the Short-Axis cine stack                                                                                                                              |
| stack_df_list                                | list             | List of pandas dataframes where each dataframe is a separate CMR stack (not necessarily a cine stack)                                                                         |
| sax_probs                                    | list             | list of the mean probabilities that each stack in stack_df_list is a Short-Axis stack. Where the mean is taken from 5 images in a stack                                       |
| sax_df                                       | pandas DataFrame | Dataframe containing the information of the identified Short-Axis cine stack                                                                                                  |
| single_dicom                                 | dicom            | A dicom taken from the Short-Axis stack, which is useful for obtaining information about the series that the stack belongs to.                                                |
| voxel_size                                   | float            | The size of the voxel in the Short-Axis stack                                                                                                                                 |
| image                                        | numpy array      | The Short-Axis stack represented as a numpy array                                                                                                                             |
| cropped_image                                | numpy array      | The cropped Short-Axis stack images represented as a numpy array                                                                                                              |
| crop_box                                     | list             | x_min, x_max, y_min and y_max representing the coordinates around the heart                                                                                                   |
| segged_image                                 | numpy array      | The cropped and segmented Short-Axis stack images represented as a numpy array                                                                                                |
| masks                                        | numpy array      | The segmentation masks of the Short-Axis stack (resized from cropped to original size)                                                                                        |
| dia_idx                                      | int              | Index of the diastolic index                                                                                                                                                  |
| sys_idx                                      | int              | Index of the systolic frame                                                                                                                                                   |
| volume                                       | list             | List of calculated volumes over the cardiac cycle                                                                                                                             |
| mass                                         | float            | Mass of the ventricles                                                                                                                                                        |
| esv                                          | float            | End-Systolic Volume                                                                                                                                                           |
| edv                                          | float            | End-Diastolic Volume                                                                                                                                                          |
| sv                                           | float            | Stroke Volume                                                                                                                                                                 |
| ef                                           | float            | Ejection Fraction                                                                                                                                                             |


So if you wanted to access the calculated volume of the patient exam you would use `p.volume`


### Reference
[1]  Huang H, Lin L, Tong R, Hu H, Zhang Q, Iwamoto Y, et al. UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation [Internet]. arXiv; 2020 [cited 2023 Feb 14]. Available from: http://arxiv.org/abs/2004.08790
