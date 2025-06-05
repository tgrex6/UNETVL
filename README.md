# UNETVL
Official repository for UNetVL: Enhancing 3D Medical Image Segmentation with Chebyshev KAN Powered Vision-LSTM.

| ![vision-lstm](https://github.com/tgrex6/UNETVL/blob/main/figures/vision-lstm.jpg) |
|:--:|
| Left: Overall architecture of UNETVL; Right: The internal structure of the ViL blocks. |

Paper: https://arxiv.org/pdf/2501.07017

## Installation
1. Create a new conda environment

```shell
conda create -n unetvl python=3.11 -y
conda activate unetvl
```
2. Install [Pytorch](https://pytorch.org/get-started/locally/), v2.1.2 or later required
3. Download code
```shell
git clone https://github.com/tgrex6/UNETVL.git
```
4. Install requirements
```shell
cd UNETVL\UNetVL\nnUNet
pip install -e .
```

## Dataset Preparation
UNetVL is built upon the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework. Please follow this [guideline](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) to prepare the dataset.

Three folders should be created: nnUNet_raw, nnUNet_preprocessed, and nnUNet_results.

### Setting environment variables:
Windows:
```bash
set nnUNet_raw=.../nnUNet_raw
set nnUNet_preprocessed=.../nnUNet_preprocessed
set nnUNet_results=.../nnUNet_result

## For permanent change you should add these to environmnent variables
```

 

Linux:
```bash
export nnUNet_raw=".../nnUNet_raw"
export nnUNet_preprocessed=".../nnUNet_preprocessed"
export nnUNet_results=".../nnUNet_results"

## For permanent change you should add these to ~/.bashrc 
```


### Convert data format:
```bash
nnUNetv2_convert_MSD_dataset -i ...\nnUNet_raw\...
```

### Preprocess and plan generation:
```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```
Replace the paths with your own.

## Model Training 
```bash
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainer_unetr_lstm --lstm True --no_kan False

## --lstm True: Use Vision-LSTM; False: Use Vision-Transformer
## --no_kan True: Use MLP; False: Use Chebyshev-KAN
```

## Inference
```bash
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainer_unetr_lstm --lstm True --no_kan False --val --npz

nnUNetv2_find_best_configuration DATASET_ID -c 3d_fullres -f 0 1 2 3 4 -tr nnUNetTrainer_unetr_lstm --lstm True --no_kan False

nnUNetv2_predict -d DATASET_ID -i INPUT_FOLDER -o OUTPUT_FOLDER -tr nnUNetTrainer_unetr_lstm -c 3d_fullres -p nnUNetPlans --lstm True --no_kan False
```
## Greatly appreciate the tremendous effort for the following projects!

## Citation
![nnUNet](https://github.com/MIC-DKFZ/nnUNet)
![vision-lstm](https://github.com/NX-AI/vision-lstm)

```
@INPROCEEDINGS{10980966,
  author={Guo, Xuhui and Dam, Tanmoy and Dhamdhere, Rohan and Modanwal, Gourav and Madabhushi, Anant},
  booktitle={2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI)}, 
  title={UNETVL: Enhancing 3D Medical Image Segmentation with Chebyshev KAN Powered Vision-LSTM}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Computer vision;Three-dimensional displays;Semantic segmentation;Scalability;Memory management;Information processing;Chebyshev approximation;Transformers;Convolutional neural networks;Biomedical imaging;Vision-LSTM;Chebyshev KAN;UNETVL;Semantic Segmentation;3D Medical Image Segmentation},
  doi={10.1109/ISBI60581.2025.10980966}}
```



