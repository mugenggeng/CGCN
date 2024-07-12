# CGCN: Context Graph Convolutional Network for Few-Shot Temporal Action Localization

## Introduction

This is a PyTorch implementation for our paper "`Explicit Long-term Gated Compressor for Online Action Detection and Anticipation`".

![network](demo/Framework1.png?raw=true)

## Environment

- The code is developed with CUDA 10.2, ***Python >= 3.7.7***, ***PyTorch >= 1.7.1***

    0. [Optional but recommended] create a new conda environment.
        ```
        conda create -n mat python=3.7.7
        ```
        And activate the environment.
        ```
        conda activate mat
        ```

    1. Install the requirements
        ```
        pip install -r requirements.txt
        ```

## Data Preparation

### Pre-extracted Feature

You can directly download the pre-extracted feature (.zip) from the UTBox links provided by [`TeSTra`](https://github.com/zhaoyue-zephyrus/TeSTra#pre-extracted-feature).


### (Alternative) Prepare dataset from scratch

You can also try to prepare the datasets from scratch by yourself. 

#### THUMOS14 and TVSeries

For THUMOS14 and TVSeries, please refer to [`LSTR`](https://github.com/amazon-research/long-short-term-transformer#data-preparation).

#### EK100

For EK100, please find more details at [`RULSTM`](https://github.com/fpv-iplab/rulstm).

### Data Structure

1. If you want to use our [dataloaders](src/rekognition_online_action_detection/datasets), please make sure to put the files as the following structure:

   * THUMOS'14 dataset:
       ```
       $YOUR_PATH_TO_THUMOS_DATASET
       ├── rgb_kinetics_resnet50/
       |   ├── video_validation_0000051.npy (of size L x 2048)
       │   ├── ...
       ├── flow_kinetics_bninception/
       |   ├── video_validation_0000051.npy (of size L x 1024)
       |   ├── ...
       ├── target_perframe/
       |   ├── video_validation_0000051.npy (of size L x 22)
       |   ├── ...
       ```
   
   
   
   
   
   * EK100 dataset:
       ```
          $YOUR_PATH_TO_EK_DATASET
          ├── rgb_kinetics_bninception/
          |   ├── P01_01.npy (of size L x 1024)
          │   ├── ...
          ├── flow_kinetics_bninception/
          |   ├── P01_01.npy (of size L x 1024)
          |   ├── ...
          ├── target_perframe/
          |   ├── P01_01.npy (of size L x 3807)
          |   ├── ...
          ├── noun_perframe/
          |   ├── P01_01.npy (of size L x 301)
          |   ├── ...
          ├── verb_perframe/
          |   ├── P01_01.npy (of size L x 98)
          |   ├── ...
       ```
   
2. Create softlinks of datasets:

    ```
    cd memory-and-anticipation-transformer
    ln -s $YOUR_PATH_TO_THUMOS_DATASET data/THUMOS
    ln -s $YOUR_PATH_TO_EK_DATASET data/EK100
    ```

## Training

The commands are as follows.

```
cd GatedOAD
# Training from scratch
python tools/train_net.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES
# Finetuning from a pretrained model
python tools/train_net.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES \
    MODEL.CHECKPOINT $PATH_TO_CHECKPOINT
```

## Online Inference

There are *two kinds* of evaluation methods in our code.

* First, you can use the config `SOLVER.PHASES "['train', 'test']"` during training. This process devides each test video into non-overlapping samples, and makes prediction on the all the frames in the short-term memory as if they were the latest frame. Note that this evaluation result is ***not*** the final performance, since (1) for most of the frames, their short-term memory is not fully utlized and (2) for simplicity, samples in the boundaries are mostly ignored.

    ```
    cd memory-and-anticipation-transformer
    # Inference along with training
    python tools/train_net.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES \
        SOLVER.PHASES "['train', 'test']"
    ```

* Second, you could run the online inference in `batch mode`. Note that this evaluation result matches the numbers reported in the paper. On the other hand, this mode can run faster when you use a large batch size, and we recomand to use it for performance benchmarking.

    ```
    cd GatedOAD
    # Online inference in batch mode
    python tools/test_net.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES \
        MODEL.CHECKPOINT $PATH_TO_CHECKPOINT MODEL.LSTR.INFERENCE_MODE batch
    ```
    
## Main Results and checkpoints

### THUMOS14

|       method      | feature   |  mAP (%)  |                             config                                                |   checkpoint   |
|  :--------------: |  :-------------:  |  :-----:  |  :-----------------------------------------------------------------------------:  |  :----------:  |
|  GatedOAD           |    Kinetics    |   72.9    | [yaml](configs/THUMOS/MAT/gatedoad_long_256_work_8_kinetics_1x.yaml)      | [Download](https://drive.google.com/file/d/1FFjpaeSFRRgpUodcvBIIcTIzsdag0iVF/view?usp=drive_link) |

### EK100

|  method  |    feature    |  verb (overall)  |  noun (overall)  |  action (overall)  |  config  |                                checkpoint                                |
|  :----:  |  :-------------:  |  :------------:  |  :------------:  |  :--------------:  |  :----:  |  :--------------------------------------------------------------------:  |
|  GatedOAD  |  RGB+FLOW  |      35.0      |       41.3       |        19.8        |  [yaml](configs/EK100/MAT/gatedoad_long_64_work_5_kinetics_1x.yaml) | [Download](https://drive.google.com/file/d/1qVz1EuIZ7pUKRjn2Udq6h82xwDrHDePY/view?usp=drive_link) |

## Citations

If you are using the data/code/model provided here in a publication, please cite our paper:
```

```
## License

This project is licensed under the Apache-2.0 License.

## Acknowledgements

This codebase is built upon [`MAT`](https://github.com/Echo0125/Memory-and-Anticipation-Transformer).

The code snippet for evaluation on EK100 is borrowed from [`TeSTra`](https://github.com/zhaoyue-zephyrus/TeSTra).
