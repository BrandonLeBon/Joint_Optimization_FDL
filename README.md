# Joint Optimization FDL

This repository is the pytorch implementation of the method proposed in the paper: [*B. Le Bon, M. Le Pendu, C. Guillemot. "Joint Unrolled Fourier Dispary Layers and view synthesis optimization for light field reconstruction from few-shots focal stacks"*].

# Usage

The purpose of this code is to jointly optimize an unroll Fourier Disparity Layers optimization with a view synthesis Deep Convolutionnal Neural Network (DCNN) in order to reconstruct a light field from focal stack images as measurements. 

## Training

### Preparation

Before launching the training, you need to prepare the following files:
* Training dataset and validation dataset files listing the path to the corresponding dataset folders. For more information on the format, refers to the dataset folder *LF_example/* and the *LF_datasets_example.txt* file
* A yaml configuration file to set up the training parameters. *Config/JointOptimizationFDLShift.yaml* is an example of a configuration file.


### Command line

The following command line is an example of how to launch the training:
    
    
    python main.py --training_dataset training_datasets.txt --validation_dataset validation_datasets.txt --config Configs/JointOptimizationFDLShift.yaml --model_name my_model_name --mode train
    

The model *my_model_name* will be saved in the *Models/* directory. In order to use the pre-training strategy described in the paper to fine-tune a pre-trained model and adding the coordinates channels in the view synthesis network input, please use the --pretrained argument as follows:
    
    
    python main.py --training_dataset training_datasets.txt --validation_dataset validation_datasets.txt --config Configs/JointOptimizationFDLShiftPretrained.yaml --model_name my_pretrained_model_name --mode train --pretrained


## Testing

### Preparation

Before launching the testing, you need to prepare the following files:
* A testing dataset file listing the path to the corresponding dataset folders. For more information on the format, refers to the dataset folder *LF_example* and the *LF_datasets_example.txt* file
* A yaml configuration file to set up the testing parameters. *Config/JointOptimizationFDLShift.yaml* is an example of a configuration file.
* A trained model located in the *Models/* repertory.

### Command line

The following command line is an example of how to launch the testing to reconstruct a light field:

    python main.py --testing_dataset testing_datasets.txt --config Configs/JointOptimizationFDLShift.yaml --model_name my_model_name --mode test --save_directory save_directory_folder

If you want to use a model which was trained using the pre-training strategy described in the paper to fine-tune a pre-trained model and adding the coordinates channels in the view synthesis network input, please use the --pretrained argument as follows:

    python main.py --testing_dataset testing_datasets.txt --config Configs/JointOptimizationFDLShiftPretrained.yaml --model_name my_model_name --mode test --save_directory save_directory_folder --pretrained
    
The model *my_model_name* in the *Models/* directory will be used, and the results will be saved in the *save_directory_folder* folder.