#! /usr/bin/env python

import numpy as np
from python_speech_features import *
import  scipy.io.wavfile
import dtw_detector

class DtwAsr(object):

    def __init__(self,path,num,suffix):

        self.path = path
        self.num = num
        self.suffix = suffix
        self.win = 0.02
        self.inc = 0.01
        self.amp1 = 10.0
        self.amp2 = 2.0
        self.zrc1 = 10
        self.zrc2 = 5
        self.maxSilence = 8
        self.minLen= 15
        self.numc=12
        self.nfilt=24
        self.preemh=0.9375


    def en_frame(self,signal,fs):

        win = int(fs*self.win)
        inc = int(fs*self.inc)
        signal_len = len(signal)
        if signal_len <= win:
            nf = 1
        else:
            nf = int(np.ceil((1.0 * signal_len - win + inc) / inc))
        pad_len = int((nf-1) * inc + win)
        zeros = np.zeros((pad_len-signal_len,))
        pad_signal = np.concatenate((signal, zeros))
        indices = np.tile(np.arange(0, win), (nf, 1))+np.tile(np.arange(0, nf*inc, inc), (win, 1)).T
        indices = np.array(indices,dtype=np.int32)
        frames = pad_signal[indices]
        return  frames

    def vad(self,signal,fs):

        win = int(fs*self.win)
        inc = int(fs*self.inc)
        signal = np.array(signal, dtype = np.float16)
        signal=signal / np.max(np.abs(signal))

        status = 0
        count = 0
        silence = 0

        tmp1 = self.en_frame(signal[:-1], fs)
        tmp2 = self.en_frame(signal[1:], fs)

        signs = np.multiply(tmp1, tmp2) < 0
        diffs = (tmp1 - tmp2) > 0.02
        zrc = np.sum(np.multiply(signs, diffs), axis=1)
        amp = np.sum(np.abs(self.en_frame(signal, fs)), axis=1)
        amp1 = max(self.amp1, np.mean(amp[:10])*3.0)
        amp2 = max(self.amp2, np.mean(amp[:10])*1.5)

        for i in range(len(zrc)):

            if status == 0 or status == 1:
                if amp[i] > amp1:
                    x1 = max(i-count-1, 0)
                    status = 2
                    silence = 0
                    count += 1
                elif amp[i] > amp2 or zrc[i] > self.zrc2:
                    count +=1
                    status = 1
                else:
                    count = 0
                    status = 0
            elif status == 2:
                if amp[i] > amp2 or zrc[i] > self.zrc2:
                    count += 1
                else:
                    silence += 1
                    if silence < self.maxSilence:
                        count += 1
                    elif count < self.minLen:
                        status = 0
                        count = 0
                        silence = 0
                    else:
                        status =3
            elif status == 3:
                break;

        count = count - silence/2;
        x2 = int(x1 + count)
        return x1, x2

    def get_mfcc(self, data, fs):

        wave_mfcc=mfcc(data,fs,winlen=self.win,winstep=self.inc,numcep=self.numc,nfilt=self.nfilt,
                       lowfreq=0,highfreq=None,preemph=self.preemh,nfft=512)
        '''
        signal - 需要用来计算特征的音频信号，应该是一个N*1的数组
        samplerate - 我们用来工作的信号的采样率
        winlen - 分析窗口的长度，按秒计，默认0.025s(25ms)
        winstep - 连续窗口之间的步长，按秒计，默认0.01s（10ms）
        numcep - 倒频谱返回的数量，默认13
        nfilt - 滤波器组的滤波器数量，默认26
        nfft - FFT的大小，默认512
        lowfreq - 梅尔滤波器的最低边缘，单位赫兹，默认为0
        highfreq - 梅尔滤波器的最高边缘，单位赫兹，默认为采样率/2
        preemph - 应用预加重过滤器和预加重过滤器的系数，0表示没有过滤器，默认0.97        
        '''
        d_wave_mfcc=delta(wave_mfcc,2)
        feature=np.hstack((wave_mfcc,d_wave_mfcc))

        return feature

    def get_feature(self, signal, fs):

        x1, x2 = self.vad(signal, fs)
        print(x1,x2)
        signal = signal/32768
        sigMfcc = self.get_mfcc(signal,fs)

        tmp2 = sigMfcc[x1:x2]
        feature = np.array(tmp2)
        return feature


    def read_wavefile(self, path):

        fs, signal = scipy.io.wavfile.read(path)

        return fs,signal






if __name__ == "__main__":

   total_mfcc_1 = []
   total_mfcc_2 = []
   path=r"D:\voice_regnition\recording"

   dtw_asr = DtwAsr(path,12,suffix=12)

   for i in range(1,8):
       spath=path + str(i) + ".wav"
       fs,sigal=dtw_asr.read_wavefile(spath)
       feature = dtw_asr.get_feature(sigal,fs)
       total_mfcc_1.append(feature)
       total_mfcc_2.append(feature)

   dtw_detector = dtw_detector.DtwDetector()

   dist=np.zeros((7,7))

   for i in range(7):
       for j in range(7):
           dist[i,j]=dtw_detector.dtw_count(total_mfcc_1[i],total_mfcc_1[j])

   print(dist)







