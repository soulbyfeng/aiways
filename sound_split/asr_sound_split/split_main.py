#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
用于测试整个一套语音识别系统的程序
语音模型 + 语言模型
"""
import platform as plat
import numpy as np
from SpeechModel251 import ModelSpeech2

from keras import backend as K
import time

datapath = ''
modelpath = './model_speech'

system_type = plat.system() # 由于不同的系统的文件路径表示不一样，需要进行判断
if(system_type == 'Windows'):
	datapath = 'D:\\语音数据集'
	modelpath = modelpath + '\\'
elif(system_type == 'Linux'):
	datapath = 'dataset'
	modelpath = modelpath + '/'
else:
	print('*[Message] Unknown System\n')
	datapath = 'dataset'
	modelpath = modelpath + '/'

ms2 = ModelSpeech2(datapath)
ms2.LoadModel(modelpath + 'speech_model251_e_0_step_507500.model')
k=0

# ---------参数说明---------------
# sound_file  音频文件目录
# text_file    文本文件目录
# save_file     保存目录    目前默认保存在./sound_split_result/{text_file}/下
def exec_sound_split(sound_file,text_file,save_file):
    from keras import backend as K
    K.clear_session()
# while 1:
    first = time.time()
    print(time.time())
    ms2 = ModelSpeech2(datapath)
    ms2.LoadModel(modelpath + 'speech_model251_e_0_step_507500.model')
    # sound_file='/home/wanggf/Github/SoundSplitPro/sound_data/3.'+str(i+1)+'.m4a'
    # text_file='/home/wanggf/Github/SoundSplitPro/sound_data/3.'+str(i+1)+'.txt'
    ms2.RecognizeSpeech_FromFile(sound_file,text_file)
    # r = ms2.RecognizeSpeech_FromFile('/home/wanggf/01.wav')
    # print('*[提示] 语音识别结果：\n',r)
    end=time.time()
    print('语音分割一共耗时：')
    print(end-first)
#
if __name__=="__main__":
    sound_file='./voice_test/lina_voice/voice_samples_s303.m4a'
    text_file='./voice_text/voice_samples_s303.txt'
    save_file=''
    exec_sound_split(sound_file,text_file,save_file)

    # sound_file = './voice_test/lina_voice/voice_samples_s302.m4a'
    # text_file = './voice_text/voice_samples_s302.txt'
    # save_file = ''
    # exec_sound_split(sound_file, text_file, save_file)













