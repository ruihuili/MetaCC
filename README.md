# MetaCodingBench

This repository provides a benchmarking framework using the channel coding application. We build the codebase based upon learn2learn library.  

We currently have implementation of 13 algorithms (12 meta-learner + 1 non-meta ERM baseline called vanilla): 
MAML, MAML + FOMAML, Reptile, Meta-SGD, Meta-KFO, ANIL, MetaCurvature, CAVIA, BOIL, ProtoNets, MetaBaseline, FEAT, ERM (vanilla).  

# Data Generation
The file utils/gen_channel_data.py contains functions used for generating true messages and encoded messages based on a given channel model and the corresponding params. It provides APIs to the commpy package which implements the convolutional encoder etc. In our current research project, we focus on learning the channel decoder while keeping the encoder fixed. 

Currently, we use pre-generated dataset instead of generating data on-the-fly in order to reduce training time. The dataset will be shared in a different repo, along with the dataset documentation. To run the program in this repo, a users should put this dataset under the same directory as the repo (or create a symbolic link accordingly).  

We use different "main" scripts for differnet learners. For example, one should use maml_coding.py for using MAML algorithm. This is likely to be changed in the near future.

## How to run the code here  
Create a new environment and install ``learn2learn`` library:  
```
conda create -n l2l python=3.6
pip install learn2learn
pip install tqdm
cd standard
```
Get the pre-gen-ed dataset and put under current directory run with compulsory argument name_of_args_json_file which specify an input .json file  

```
python maml_coding.py --name_of_args_json_file configs/set_nd_15ts_5cls/awgn_mid_higher.json  
```

## Details
We use a number of Json files under config/set_nd_15ts_5cl/* to specify specific training parameters for the channel noise. A number of utility tools e.g. parser can be found unter the sub-dir "/utils/". We currently use 4 layer CNN for all algorithms. 


# Main Contributors 
Rui Li (Rui.Li@samsung.com) 
Ondrej Bohdal (Ondrej.Bohdal@ed.ac.uk)  
