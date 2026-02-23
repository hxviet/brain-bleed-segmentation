# Multiclass Intracranial Hemorrhage Segmentation in CT Scans


This repository contains the implementation and results for a multiclass segmentation model for Intracranial Hemorrhage (ICH). We enhance the standard 3D nnU-Net model through optimized data preprocessing and ensemble learning.



## Project Overview


Intracranial hemorrhage (ICH) is a critical medical condition involving bleeding within the skull that requires rapid diagnosis. Treatment depends heavily on the specific type, location, and volume of the bleed. This project focuses on the multiclass segmentation of five ICH categories:

* **EDH**: Epidural Hemorrhage

* **SDH**: Subdural Hemorrhage

* **SAH**: Subarachnoid Hemorrhage

* **IPH**: Intraparenchymal Hemorrhage

* **IVH**: Intraventricular Hemorrhage


### Key Features

* **Backbone**: Built on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), a self-configuring framework for U-Net-based medical image segmentation.

* **Dataset**: Utilizes the public [Brain Hemorrhage Segmentation Dataset (BHSD)](https://github.com/White65534/BHSD), comprising 3D CT head scans. Uses 296 scans for model training/validation and 96 scans for model testing.

* **CT Windowing Optimization**: Implements custom data preprocessing schemes inspired by clinical practices to enhance the visibility of rarer hemorrhage types such as EDH and SDH.

* **Ensemble Learning**: Combines multiple U-Net models to improve performance across all classes.



## Methodology


### 1. Data Preprocessing

3D CT volumes are preprocessed in three steps: windowing, normalization, and resampling.

#### Windowing

Raw HU values in CT scans span a very wide range, typically at least [-1000, 1000], so they need to be truncated to a narrow range to focus the neural network on the radiodensity ranges most relevant to hemorrhages and brain tissue. nnU-Net preprocesing by default uses the window from the 0.5th to the 99.5th percentile of foreground voxels, which is [10, 81] Hounsfield units (HU) for our dataset. This corresponds to the ["brain window"](https://radiopaedia.org/articles/windowing-ct) in radiology). We introduce two additional ["subdural windows"](https://radiopaedia.org/articles/windowing-ct) to experiment with:

* **Subdural Window A**: [-15, 115] HU
* **Subdural Window B**: [-100, 200] HU

#### Normalization and resampling

These steps follow nnU-Net's default settings.


### 2. Model Training

We compared 2D and 3D nnU-Net architectures on the validation set. The 3D nnU-Net was selected as the backbone because it outperformed the 2D variant in terms of macro-average Dice Similarity Coefficient (DSC).

* **2D nnU-Net**: input patch size 512x512, batch size 12

* **3D nnU-Net**: input patch size 256x256x28, batch size 2

All other training settings use nnU-Net's defaults.


### 3. Ensemble Strategy

The final model is an ensemble that averages the probability distributions from three distinct 3D nnU-Net models trained with different CT windowing settings (Brain Window, Subdural Window A, and Subdural Window B).



## Results Highlights


* **Preprocessing Impact**: Customized CT windowing improved the DSC of nnU-Net by up to 3.63% for EDH and 5.66% for SDH over default settings on the test set.


* **Architecture**: 3D nnU-Net achieved a macro-average DSC of 30.40% on the validation set, significantly outperforming the 2D model (22.37%).


* **Ensemble Learning**: An ensemble of three 3D models achieved the highest validation macro-average DSC of 34.68%.



## Repository Structure


* `ICH_Segmentation.ipynb`: Complete pipeline including dataset preparation, model training, evaluation, and inference.


* `data_utils.py`: Utility functions for creating segmentation masks and handling NIfTI files.


* `eval_utils.py`: Utility functions for computing segmentation metrics and confusion matrices


* `Presentation.pdf`: More detailed project summary.



## Usage


1. Open `ICH_Segmentation.ipynb` in Google Colab (badge link included in the notebook).


2. Configure the following Colab Secrets:

   * `gdrive_nnUNet_raw_path`, `gdrive_nnUNet_preprocessed_path`, and `gdrive_nnUNet_results_path`: Google Drive locations (starting with `/content/drive`) (or any persistent storage locations mounted on Colab disk) at which you want to store raw data, preprocessed data, and results for nnU-Net. (Learn more about [nnU-Net's dataset format](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md))
   
   * `gdrive_BHSD_name`: must be `DatasetXXX_<whatever>`, where each `X` is a digit (e.g., `Dataset011_BHSD`)


3. Connect to a runtime with a GPU (A100 recommended) and run the notebook sections sequentially.




## Acknowledgments

This research was conducted as part of my thesis at Hanoi University of Science and Technology. I extend my gratitude to M.S. Đỗ Tuấn Anh and Dr. Nguyễn Đức Anh for their guidance.




## References


1. Wu et al. *"BHSD: A 3D Multi-class Brain Hemorrhage Segmentation Dataset"*. MLMI. 2023. https://link.springer.com/chapter/10.1007/978-3-031-45673-2_15


2. Isensee et al. *"nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation"*. Nature Methods. 2020. https://www.nature.com/articles/s41592-020-01008-z


3. Murphy et al. *"Windowing (CT)"*. Radiopaedia. 2025. https://radiopaedia.org/articles/52108