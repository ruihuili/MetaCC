# MetaCodingBench

This repository provides a benchmarking framework using the channel coding application. We build the codebase based upon learn2learn library.  

We currently have implementation of 13 algorithms (12 meta-learner + 1 non-meta ERM baseline called vanilla): 
MAML, MAML + FOMAML, Reptile, Meta-SGD, Meta-KFO, ANIL, MetaCurvature, CAVIA, BOIL, ProtoNets, MetaBaseline, FEAT, ERM (vanilla).  

The file gen_channel_data.py contains functions used for generating true messages and encoded messages based on a given channel model and the corresponding params. It provides APIs to the commpy package which implements the convolutional encoder etc. In our current research project, we focus on learning the channel decoder while keeping the encoder fixed. 

Currently, we use pre-generated dataset instead of generating data on-the-fly in order to reduce training time. The dataset will be shared in a different repo, and users should put this dataset dir under standard/withWarpGrad directory (or create a symbolic link accordingly).  

We use different "main" scripts for differnet learners.

NB: most of the experiments are done using standard sub-dir. The other sub-dir withWarpGrad is used only for warpgrad in task augmentation.   

## How to run the code here  
Create a new environment and install ``learn2learn`` library:  
```
conda create -n l2l python=3.6
pip install learn2learn
pip install tqdm
cd standard
```
Get the pre-gen-ed dataset and put under ./standard/ run with compulsory argument name_of_args_json_file which specify an input .json file  

```
python maml_coding.py --name_of_args_json_file configs/set_nd_15ts_5cls/awgn_mid_higher.json  
```

## Details
We use a number of Json files under config/set_nd_15ts_5cl/* to specify lists of arguments that are required for a range of experiment settings. We also use parser_util.py to keep control of universal arguments applicable to all experiments.   

