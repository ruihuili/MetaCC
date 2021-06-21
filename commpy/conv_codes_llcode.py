__author__ = 'abc'
'''
Evaluate convolutional code benchmark.
'''
from utils import corrupt_signal, get_test_sigmas

import sys
import numpy as np
import time


import channelcoding.convcode as cc
from utilities import hamming_dist
import multiprocessing as mp

def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-num_block', type=int, default=10)
    parser.add_argument('-block_len', type=int, default=100)
    parser.add_argument('-delay', type=int, default=2)
    parser.add_argument('-tb', type=int, default=15)
    parser.add_argument('-fair', type=int, default=1)

    parser.add_argument('-M',  type=int, default=2) # 2
    parser.add_argument('-code_rate',  type=int, default=2)

    parser.add_argument('-enc1',  type=int, default=7)  #7
    parser.add_argument('-enc2',  type=int, default=7)  #5
    parser.add_argument('-enc3',  type=int, default=5)
    parser.add_argument('-enc4',  type=int, default=7)

    parser.add_argument('-feedback',  type=int, default=7) #7

    parser.add_argument('-num_cpu', type=int, default=1)

    parser.add_argument('-snr_test_start', type=float, default=-1.0)
    parser.add_argument('-snr_test_end', type=float, default=8.0)
    parser.add_argument('-snr_points', type=int, default=10)

    parser.add_argument('-dec_type', choices = ['awgn', 't3', 't5'], default='awgn')

    parser.add_argument('-noise_type',        choices = ['awgn', 't-dist','radar', 'fading',
                                                         'radar_saturate', 'radar_erasure'], default='awgn')
    parser.add_argument('-radar_power',       type=float, default=20.0)
    parser.add_argument('-radar_prob',        type=float, default=0.05)
    parser.add_argument('-radar_denoise_thd', type=float, default=10.0)
    parser.add_argument('-v',                 type=float,   default=3.0)


    parser.add_argument('-id', type=str, default=str(np.random.random())[2:8])

    args = parser.parse_args()

    print args
    print '[ID]', args.id
    return args


if __name__ == '__main__':
    args = get_args()

    ##########################################
    # Setting Up Codec
    ##########################################
    M = np.array([args.M]) # Number of delay elements in the convolutional encoder
    if args.code_rate == 2:
        generator_matrix = np.array([[args.enc1, args.enc2]])
    elif args.code_rate == 3:
        generator_matrix = np.array([[args.enc1, args.enc2, args.enc3]])

    elif args.code_rate == 4:
        generator_matrix = np.array([[args.enc1, args.enc2, args.enc3, args.enc4]])
    else:
        print 'Not supported!'
        sys.exit()
    feedback = args.feedback

    print '[testing] Convolutional Code Encoder: G: ', generator_matrix,'Feedback: ', feedback,  'M: ', M

    trellis1 = cc.Trellis(M, generator_matrix,feedback=feedback)  # Create trellis data structure

    SNRS, test_sigmas = get_test_sigmas(args.snr_test_start, args.snr_test_end, args.snr_points)

    tic = time.time()
    tb_depth = args.tb

    def turbo_compute((idx, x)):
        '''
        Compute Turbo Decoding in 1 iterations for one SNR point.
        '''
        np.random.seed()
        message_bits = np.random.randint(0, 2, args.block_len)

        coded_bits = cc.conv_encode(message_bits, trellis1)
        received  = corrupt_signal(coded_bits, noise_type =args.noise_type, sigma = test_sigmas[idx],
                                   vv =args.v, radar_power = args.radar_power, radar_prob = args.radar_prob,
                                   denoise_thd = args.radar_denoise_thd)

        # make fair comparison between (100, 204) convolutional code and (100,200) RNN decoder, set the additional bit to 0
        if args.fair == 1:
            received[-2*int(M):] = 0.0

        if args.dec_type == 't3':
            decoded_bits = cc.viterbi_decode(received.astype(float), trellis1, tb_depth, decoding_type='tdist3')
        elif args.dec_type == 't5':
            decoded_bits = cc.viterbi_decode(received.astype(float), trellis1, tb_depth, decoding_type='tdist5')
        else:
            decoded_bits = cc.viterbi_decode(received.astype(float), trellis1, tb_depth, decoding_type='unquantized')

        decoded_bits = decoded_bits[:-int(M)]
        num_bit_errors = hamming_dist(message_bits[args.block_len - args.delay - 1], decoded_bits[args.block_len - args.delay - 1])


        error_vec = [hamming_dist(message_bits[idx],decoded_bits[idx]) for idx in range(len(message_bits))]

        return error_vec

    commpy_res_ber_delay0 = []
    commpy_res_ber_delay1 = []
    commpy_res_ber_delay2 = []
    commpy_res_ber_delay5 = []
    commpy_res_ber_delay10= []
    commpy_res_ber_delay15= []
    commpy_res_ber_delay20= []
    commpy_res_ber_delay50= []

    nb_errors          = np.zeros(test_sigmas.shape)
    map_nb_errors      = np.zeros(test_sigmas.shape)
    nb_block_no_errors = np.zeros(test_sigmas.shape)

    for idx in range(len(test_sigmas)):
        start_time = time.time()

        pool = mp.Pool(processes=args.num_cpu)
        results = pool.map(turbo_compute, [(idx,x) for x in range(args.num_block)])

        results = np.array(results)
        res = np.mean(results, axis=0)

        print '[testing]SNR: ' , SNRS[idx]
        print '[testing]BER: ', res

        commpy_res_ber_delay0.append(float(res[-1]))
        commpy_res_ber_delay1.append(float(res[-2]))
        commpy_res_ber_delay2.append(float(res[-3]))
        commpy_res_ber_delay5.append(float(res[-6]))
        commpy_res_ber_delay10.append(float(res[-11]))
        commpy_res_ber_delay15.append(float(res[-16]))
        commpy_res_ber_delay20.append(float(res[-21]))
        commpy_res_ber_delay50.append(float(res[-51]))

        end_time = time.time()
        print '[testing] This SNR runnig time is', str(end_time-start_time)


    print '[Result]SNR: ', SNRS
    print '[Result]BER at delay 0', commpy_res_ber_delay0
    print '[Result]BER at delay 1', commpy_res_ber_delay1
    print '[Result]BER at delay 2', commpy_res_ber_delay2
    print '[Result]BER at delay 5', commpy_res_ber_delay5
    print '[Result]BER at delay 10', commpy_res_ber_delay10
    print '[Result]BER at delay 20', commpy_res_ber_delay20
    print '[Result]BER at delay 50', commpy_res_ber_delay50

    toc = time.time()

    print '[Result]Total Running time:', toc-tic
