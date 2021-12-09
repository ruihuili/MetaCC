Download the dataset [[here](https://drive.google.com/drive/folders/1QTTPxyylBJntAru2b_KZ39tNZZMl_WfG?usp=sharing)]

This is a README doc for the channel dataset we publish along MetaCC project: https://github.com/ruihuili/MetaCC.git 

This dataset contains the synthetic & real channel data. 

# Synthetic Data
Synthetic data can be found in `\train\` and `\test\` sub-folders.

Particularly, for `\train\`, there are a number of `\*.npz` files for the data and `\*.json` files for the corresponding description of the channel settings. 

File names indicate the channel types and distribution type i.e. `$ChannelType_$DistributionType_data.npz` 

## Parameter setting
The specific range from which the channel parameters are uniformly sampled, for each channel type:

Channel Setting (filename)                | AWGN SNR  | Bursty  SNR SNR_B  | Memory SNR \alpha        |  Multipath SNR \beta 
------------|------------------------------------------|-------|----|---------------------          
Focused (narrow)           | [-0.5, 0.5]  | [5.5, 6.5] [-15, -13] |[-0.5, 0.5] [0.45, 0.55] | [-0.5, 0.5] [0.45, 0.55] 
Expanded  (wide)       | [-5, 5]  |  [1, 11] [-19, -9] | [-5, 5] [0.1, 0.9] | [-5, 5] [0.1, 0.9] 
Low (mid_low)         | -  |[ -2.5, 3.5] [ -23, -17] | - | - |
High (mid_high)       | -  |[ 8.5, 13.5 ] [ -11, -5] | - | - |


For the `\test\` sub-folder there are `test_data.npy` and `test_data.json` where the latter describes the range of channels contained in the former. 




# Real Data

`DataIn_64.mat` and `DataOut_64.mat` contains the groud truth messages and the received messages, respectively.
 
## Real data collection/testbed setup

The wireless testbed setup consists of two separate N200 USRPs operating as the transmitter and the receiver using antennas to communicate over air. The USRPs are connected to the system through the ethernet medium. We use MATLAB 2021 to preprocess and post-process the data while we use GNURadio to communicate with the USRPs. We derive the frame structure and the modulation parameters from the WiFi standard 802.11a. The transmit signals are arranged in frames with a preamble followed by data. The preamble consists of a short training sequence (STS) and a long training sequence (LTS). The encoded bits are mapped into 64-QAM symbols on the transmitter side and then modulated onto the subcarriers of the OFDM symbol along with the guard interval. The symbols also carry the pilot carriers in specific locations to aid in channel estimation and phase offset correction. These symbols are then converted to the time domain, appended by a cyclic prefix, and sent to the USRPs for transmission. Upon receiving the signal at the other USRP, we use the STS, and the LTS preambles for synchronization and frequency offset corrections.  We remove the added cyclic prefix and convert the signal into frequency domain through an FFT. Lastly, we use the pilot carriers for channel equalization followed by demodulation to get the corresponding LLRs. The SNR of the transmission is managed by changing the transmit and receive power gains in order to achieve a requisite error performance mandated by the training procedure.
