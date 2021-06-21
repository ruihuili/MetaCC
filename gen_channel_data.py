import numpy as np
import commpy.channelcoding.convcode as cc
import commpy.channelcoding
import commpy.channelcoding.turbo as turbo
import commpy.channelcoding.interleavers as RandInterlv
from commpy.utilities import *

import math
# import matplotlib
#matplotlib.use('pdf')
# import matplotlib.pyplot as plt

import time
import pickle


# =============================================================================
# Generating pairs of (noisy codewords, message bit sequence)
# =============================================================================


def generate_viterbi_batch(batch_size=100, block_len=200, code_rate = 2, batch_criteria = {}, seed = 0):

    noise_type = batch_criteria["noise_type"]
    SNR = batch_criteria["snr"]
    rng = np.random.RandomState(seed)
    

    # print("[generate_viterbi_batch] block_len, code_rate", block_len, code_rate)
    trellis1 = cc.Trellis(np.array([2]), np.array([[7,5]]))
    trellis2 = cc.Trellis(np.array([2]), np.array([[7,5]]))
    #print('trellis: cc.Trellis(np.array([2]), np.array([[7,5]]))') # G(D) corresponding to the convolutional encoder

    tic = time.time()

    ### TEST EXAMPLES

    # Initialize Test Examples/
    noisy_codewords = np.zeros([1,batch_size,block_len,2])
    true_messages = np.zeros([1,batch_size,block_len,1])

    iterations_number = batch_size

    #for idx in range(SNR_points):
    nb_errors = np.zeros([iterations_number,1])


    tic = time.time()

    noise_sigmas = 10**(-SNR*1.0/20)

    mb_test_collect = np.zeros([iterations_number,block_len])

    interleaver = RandInterlv.RandInterlv(block_len,0)

    message_bits = rng.randint(0, 2, block_len)
#            mb_test_collect[iterations,:] = message_bits
    [sys, par1, par2] = turbo.turbo_encode(message_bits, trellis1, trellis2, interleaver)

    # print("[debug] noise type ", noise_type, " noise_sigmas ", noise_sigmas,
    # "vv", vv, "radar_power", radar_pow, "radar_prob", radar_prob)
    for iterations in range(iterations_number):
        noise_seed1 = rng.randint(1, 999999)
        noise_seed2 = rng.randint(1, 999999)
        noise_seed3 = rng.randint(1, 999999)
        # print("seeds ",  noise_seed1, noise_seed2)
        sys_r = corrupt_signal(input_signal = sys, noise_type = noise_type, sigma = noise_sigmas, \
            metrics = batch_criteria, seed = noise_seed1)
        par1_r = corrupt_signal(input_signal = par1, noise_type = noise_type, sigma = noise_sigmas, \
            metrics = batch_criteria, seed = noise_seed2)
        par2_r = corrupt_signal(input_signal = par2, noise_type = noise_type, sigma = noise_sigmas ,\
            metrics = batch_criteria, seed = noise_seed3)
        # print("sys_r ", sys_r, flush=True)
        # print("par1_r", par1_r, flush=True)
        # ADD Training Examples
        noisy_codewords[0,iterations,:,:] = np.concatenate([sys_r.reshape(block_len,1),par1_r.reshape(block_len,1)],axis=1)

        # Message sequence
        true_messages[0,iterations,:,:] = message_bits.reshape(block_len,1)

    noisy_codewords = noisy_codewords.reshape(batch_size,block_len,code_rate)
    true_messages = true_messages.reshape(batch_size,block_len)
 #   target_true_messages  = mb_test_collect.reshape([mb_test_collect.shape[0],mb_test_collect.shape[1],1])

    toc = time.time()

    #print('time to generate test examples:', toc-tic)

    return (noisy_codewords, true_messages)

 

def corrupt_signal(input_signal, noise_type, sigma, metrics = {}, seed = 0, debugging = False):
    data_shape = input_signal.shape  # input_signal has to be a numpy array.
    rng = np.random.RandomState(seed)
    
    if noise_type == 'awgn':
        noise = sigma * rng.standard_normal(data_shape) # Define noise
        corrupted_signal = 2.0*input_signal-1.0 + noise

    elif noise_type == 't':
        vv = metrics["vv"]
        if debugging: print("creating [t] sig with prob ", vv, " sigma ", sigma)
        noise = sigma * math.sqrt((vv-2)/vv) *rng.standard_t(vv, size = data_shape)
        corrupted_signal = 2.0*input_signal-1.0 + noise

    elif noise_type == 'awrad':
        radar_prob = metrics["prob"]
        radar_power = metrics["pow"]
        if debugging: print("creating radar sig with prob ", radar_prob , "and radar_power", radar_power)
        bpsk_signal = 2.0*input_signal-1.0 + sigma * rng.standard_normal(data_shape)
        # add_pos     = np.random.choice([-1.0, 0.0, 1.0], data_shape, p=[radar_prob/2, 1 - radar_prob, radar_prob/2])
        # add_poscomp = np.ones(data_shape) - abs(add_pos)
        # corrupted_signal = bpsk_signal * add_poscomp + np.random.normal(radar_power, 1.0,size = data_shape ) * add_pos

        add_pos     = rng.choice([0.0, 1.0], data_shape, p=[1 - radar_prob, radar_prob])

        corrupted_signal = bpsk_signal  + rng.normal(0 , radar_power,size = data_shape ) * add_pos

        # noise = sigma * np.random.standard_normal(data_shape) + \
        #         np.random.normal(radar_power, 1.0,size = data_shape ) * np.random.choice([-1.0, 0.0, 1.0], data_shape, p=[radar_prob/2, 1 - radar_prob, radar_prob/2])
        #
        # corrupted_signal = 2.0*input_signal-1.0  + noise

    # elif noise_type == 'radar':
    #     noise = np.random.normal(radar_power, 1.0,size = data_shape ) * np.random.choice([-1.0, 0.0, 1.0], data_shape, p=[radar_prob/2, 1 - radar_prob, radar_prob/2])
    #     corrupted_signal = 2.0*input_signal-1.0 + noise

    elif noise_type == "multipath":
        d = rng.randint(1, data_shape)[0] # random location 
        weight = metrics["weight"]
        if debugging: print("creating multipath with d ", d, "and weight", weight, " sigma ", sigma)
        awgn_noise = sigma * rng.standard_normal(data_shape) 
        bpsk_sig =  2.0* input_signal-1.0

        time_delayed_sig = np.concatenate((np.zeros((d,)), bpsk_sig)) 
        if debugging: 
            print("weighted time_delayed_sig",  weight*time_delayed_sig[:-d])
            print("awgn", awgn_noise)
            print("bpsk_sig", bpsk_sig)
        multipath_noise = weight*time_delayed_sig[:-d]
        corrupted_signal = bpsk_sig +  multipath_noise  + awgn_noise 

    elif noise_type == "multipath_multitab":

        weights = metrics["weights"]
        d = rng.randint(data_shape, size = len(weights)) # random locations
        if debugging: print("creating ", len(d), " tab multipath with d ", d, "and weights", weights, " sigma ", sigma)
        awgn_noise = sigma * rng.standard_normal(data_shape) 
        bpsk_sig =  2.0* input_signal-1.0
        multipath_noise = np.zeros(np.shape(input_signal))

        for i in range(len(d)):
            time_delayed_sig = np.concatenate((np.zeros((d[i],)), bpsk_sig)) 
            # print("tab ", i, "weighted time_delayed_sig",  weights[i]*time_delayed_sig[:-d[i]])

            multipath_noise = multipath_noise + weights[i]*time_delayed_sig[:-d[i]]
        corrupted_signal = bpsk_sig +  multipath_noise  + awgn_noise 

        # print("inupt_signal", input_signal, "\n bpsk_sig", bpsk_sig)
        # print("awgn noise ", awgn_noise)
        # print("multipath_noise", multipath_noise)
        # print("corrupted sig", corrupted_signal)

    elif noise_type == "memory":
        alpha = metrics["alpha"]
        awgn_noise = sigma * rng.standard_normal(data_shape) 
        memory_noise = awgn_noise
        for i in np.arange(1, len(input_signal)):
            memory_noise[i] = np.sqrt(1-alpha**2) *memory_noise[i] + alpha* memory_noise[i-1]
        corrupted_signal = 2.0* input_signal-1.0 +  memory_noise

    elif noise_type == "bursty":
        p = metrics["p"] if "p" in metrics.keys() else 0.3
        snrb = metrics["snrb"] if "snrb" in metrics.keys() else 1.2 
        sigma_bursty = 10**(-snrb*1.0/20)
        if debugging: print("sigb", sigma_bursty)
        awgn = sigma * rng.standard_normal(data_shape)
        bursty_awgn = sigma_bursty * rng.standard_normal(data_shape)
        bursty_indicator = (rng.random(len(input_signal)) <= p)
        corrupted_signal = 2.0*input_signal-1.0 + awgn + bursty_awgn * bursty_indicator

    elif noise_type == "bsc":
        p = metrics["p"]
        corrupted_signal = input_signal.copy()
        flip_locs = (rng.random(len(input_signal)) <= p)
        corrupted_signal[flip_locs] = 1 ^ corrupted_signal[flip_locs]
        # print("flip locs", flip_locs )
        # print("corrupted sig", corrupted_signal)
        
    elif noise_type == "bec":
        p = metrics["p"]
        corrupted_signal = input_signal.copy()
        corrupted_signal[rng.random(len(corrupted_signal)) <= p] = -1

    else:
        print("Undefiled channel ")
        import sys
        sys.exit()

    return corrupted_signal

"""
    if noise_type == "awgn":
        print("[debug] noise type awgn")
        for iterations in range(iterations_number):
            noise = noise_sigmas*np.random.standard_normal(sys.shape) # Generate noise
            sys_r = (2*sys-1) + noise # Modulation plus noise
            noise = noise_sigmas*np.random.standard_normal(par1.shape) # Generate noise
            par1_r = (2*par1-1) + noise # Modulation plus noise
            noise = noise_sigmas*np.random.standard_normal(par2.shape) # Generate noise
            par2_r = (2*par2-1) + noise # Modulation plus noise

            sys_symbols = sys_r
            non_sys_symbols_1 = par1_r
            non_sys_symbols_2 = par2_r

            # ADD Training Examples
            noisy_codewords[0,iterations,:,:] = np.concatenate([sys_r.reshape(block_len,1),par1_r.reshape(block_len,1)],axis=1)

            # Message sequence
            true_messages[0,iterations,:,:] = message_bits.reshape(block_len,1)

    # T distribution: y=x+z, where z∼T(ν,σ2)

    elif noise_type == "t":
        print("[debug] noise type t")
        for iterations in range(iterations_number):
            noise_t = 3
            noise = noise_t + noise_sigmas*np.random.standard_normal(sys.shape) # Generate noise
            sys_r = (2*sys-1) + noise # Modulation plus noise
            noise = noise_t + noise_sigmas*np.random.standard_normal(par1.shape) # Generate noise
            par1_r = (2*par1-1) + noise # Modulation plus noise
            noise = noise_t + noise_sigmas*np.random.standard_normal(par2.shape) # Generate noise
            par2_r = (2*par2-1) + noise # Modulation plus noise

            sys_symbols = sys_r
            non_sys_symbols_1 = par1_r
            non_sys_symbols_2 = par2_r

            # ADD Training Examples
            noisy_codewords[0,iterations,:,:] = np.concatenate([sys_r.reshape(block_len,1),par1_r.reshape(block_len,1)],axis=1)

            # Message sequence
            true_messages[0,iterations,:,:] = message_bits.reshape(block_len,1)
    
    elif noise_type == "isi":
        print("[debug] noise type isi")
        for iterations in range(iterations_number):
            interf = np.zeros(np.shape(sys))
            interf[1:] = sys[:-1]
            noise = noise_sigmas*np.random.standard_normal(sys.shape) # Generate noise
            sys_r = (2*sys-1) + noise + interf# Modulation plus noise

            interf = np.zeros(np.shape(par1_r))
            interf[1:] = par1_r[:-1]
            noise = noise_sigmas*np.random.standard_normal(par1.shape) # Generate noise
            par1_r = (2*par1-1) + noise + interf# Modulation plus noise
            
            interf = np.zeros(np.shape(par2_r))
            interf[1:] = par2_r[:-1]
            noise = noise_sigmas*np.random.standard_normal(par2.shape) # Generate noise
            par2_r = (2*par2-1) + noise +interf # Modulation plus noise

            sys_symbols = sys_r
            non_sys_symbols_1 = par1_r
            non_sys_symbols_2 = par2_r

            # ADD Training Examples
            noisy_codewords[0,iterations,:,:] = np.concatenate([sys_r.reshape(block_len,1),par1_r.reshape(block_len,1)],axis=1)

            # Message sequence
            true_messages[0,iterations,:,:] = message_bits.reshape(block_len,1)


    noisy_codewords = noisy_codewords.reshape(batch_size,block_len,code_rate)
    true_messages = true_messages.reshape(batch_size,block_len,1)
 #   target_true_messages  = mb_test_collect.reshape([mb_test_collect.shape[0],mb_test_collect.shape[1],1])

    toc = time.time()

    #print('time to generate test examples:', toc-tic)

    return (noisy_codewords, true_messages)
    """
