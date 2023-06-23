# Single Ventricle Segmentation Pipeline

This repository contains the code for the paper:

**Tina Yao, Nicole St. et al.** Deep Learning Pipeline for Preprocessing and Segmenting Cardiac Magnetic Resonance of Single Ventricle Patients from an Image Registry.[[preprint]](https://arxiv.org/abs/2303.11676)


## Code
The pipeline is designed to work with a specific folder structure, as described below:

* data_path
    * patient1
        * scan1
            * dicoms.dcm
        * scan2
            * dicoms.dcm
        * ...
    * patient2
        * scan1
            * dicoms.dcm
        * scan2
            * dicoms.dcm
        * ...
    * ...
    
* results_path
    * segs  (segmented gifs)
        * patient1.gif
        * patient2.gif
    * volume_curves
        * patient1.jpg
        * patient2.jpg
    * sys_dia_plot (segmented images of systole and diastole)
        * patient1.png
        * patient2.png
    * ...


The code provides a convenient way to process patient data and generate various results. To use the code, follow the steps below:

Make sure you have the necessary machine learning models stored in the "models" folder. By default, the pipeline expects the models to be located in the correct relative path from the pipeline.py file. If you choose to move the models to a different location, you can modify the model paths in pipeline.py at the top of the file.

Instantiate the pipeline object by adding the following line of code to your script:
`p = Pipeline(patient, data_path='data', results_path='results')`

The patient argument should be the name of the patient for whom you want to process the scans. The data_path and results_path arguments specify the paths to the input data and the location where the results should be stored, respectively.

The pipeline consists of several functions, each performing a specific task. Each function is commented to provide a clear understanding of its purpose and functionality. You can review these comments to understand the code's inner workings.

Once the pipeline object is instantiated, it will automatically process the scans for the specified patient and generate the desired results. The segmented GIFs will be stored in the segs folder, the volume curves will be saved as JPEG images in the volume_curves folder, and the segmented images of systole and diastole will be stored as PNG files in the sys_dia_plot folder.

Please note that if the code is implemented correctly and the necessary models are available.

### Models
To download pretrained models, please download them from this [Google Drive link](https://www.example.com](https://drive.google.com/drive/folders/1df2Cf-bUgBG3KeMkaaUTp-ZE8sS-tGfK?usp=drive_link)https://drive.google.com/drive/folders/1df2Cf-bUgBG3KeMkaaUTp-ZE8sS-tGfK?usp=drive_link)
