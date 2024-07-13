# CGCN: Context Graph Convolutional Network for Few-Shot Temporal Action Localization

## Introduction

This is a PyTorch implementation for our paper "`CGCN: Context Graph Convolutional Network for Few-Shot Temporal Action Localization`".

![network](framework.png?raw=true)

## Dependencies
* Python == 3.7
* Pytorch==1.1.0 or 1.3.0
* CUDA==10.0.130
* CUDNN==7.5.1_0
* GCC >= 4.9
* pip install git+https://github.com/luizgh/visdom_logger.git

## Installation
1. Create conda environment
    ```shell script
    conda env create -f env.yml
    source activate gtad
    ```
2. Install `Align1D2.2.0`
    ```shell script
    cd gtad_lib
    python setup.py install
    ```
3. Test `Align1D2.2.0`
    ```shell script
    python align.py
    ```
4. Post-processing : Download the CUHK classifier from this [link](https://drive.google.com/file/d/1--d6V5xeVWznO0cPI_47f5wWGL8RO6P0/view?usp=sharing) and place it in "data" folder
   


## License

This project is licensed under the Apache-2.0 License.

## Acknowledgements

This codebase is built upon [`fewQAT`](https://github.com/sauradip/fewshotQAT).

