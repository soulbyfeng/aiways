#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
"""
import platform as plat
import os
import time

from general_function.file_wav import *
from general_function.file_dict import *
from general_function.gen_func import *

# LSTM_CNN
import keras as kr
import numpy as np
import random
import librosa

from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Dropout, Input, Reshape, BatchNormalization  # , Flatten
from keras.layers import Lambda, TimeDistributed, Activation, Conv2D, MaxPooling2D  # , Merge
from keras import backend as K
from keras.optimizers import SGD, Adadelta, Adam

from readdata24 import DataSpeech


abspath = ''
ModelName = '251'
sr=16000
def load_wav(path):
  return librosa.core.load(path, sr=sr)[0]
def save_wav(filename,wave_arr):
    librosa.output.write_wav(filename, wave_arr, sr)
# NUM_GPU = 2
datapath='./dict.txt'

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
list_symbol_dic = GetSymbolList(datapath)  # 获取拼音列表
class ModelSpeech2():  # 语音模型类
    def __init__(self, datapath):
        '''
        初始化
        默认输出的拼音的表示大小是1424，即1423个拼音+1个空白块
        '''
        MS_OUTPUT_SIZE = 1424
        self.MS_OUTPUT_SIZE = MS_OUTPUT_SIZE  # 神经网络最终输出的每一个字符向量维度的大小
        # self.BATCH_SIZE = BATCH_SIZE # 一次训练的batch
        self.label_max_string_length = 64
        self.AUDIO_LENGTH = 1600
        self.AUDIO_FEATURE_LENGTH = 200
        self._model, self.base_model = self.CreateModel()

        self.datapath = datapath
        self.slash = ''
        system_type = plat.system()  # 由于不同的系统的文件路径表示不一样，需要进行判断
        if (system_type == 'Windows'):
            self.slash = '\\'  # 反斜杠
        elif (system_type == 'Linux'):
            self.slash = '/'  # 正斜杠
        else:
            print('*[Message] Unknown System\n')
            self.slash = '/'  # 正斜杠
        if (self.slash != self.datapath[-1]):  # 在目录路径末尾增加斜杠
            self.datapath = self.datapath + self.slash

    def CreateModel(self):
        '''
        定义CNN/LSTM/CTC模型，使用函数式模型
        输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）
        隐藏层：卷积池化层，卷积核大小为3x3，池化窗口大小为2
        隐藏层：全连接层
        输出层：全连接层，神经元数量为self.MS_OUTPUT_SIZE，使用softmax作为激活函数，
        CTC层：使用CTC的loss作为损失函数，实现连接性时序多输出

        '''

        input_data = Input(name='the_input', shape=(self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1))

        layer_h1 = Conv2D(32, (3, 3), use_bias=False, activation='relu', padding='same',
                          kernel_initializer='he_normal')(input_data)  # 卷积层
        layer_h1 = Dropout(0.05)(layer_h1)
        layer_h2 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h1)  # 卷积层
        layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2)  # 池化层
        # layer_h3 = Dropout(0.2)(layer_h2) # 随机中断部分神经网络连接，防止过拟合
        layer_h3 = Dropout(0.05)(layer_h3)
        layer_h4 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h3)  # 卷积层
        layer_h4 = Dropout(0.1)(layer_h4)
        layer_h5 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h4)  # 卷积层
        layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5)  # 池化层

        layer_h6 = Dropout(0.1)(layer_h6)
        layer_h7 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                          kernel_initializer='he_normal')(layer_h6)  # 卷积层
        layer_h7 = Dropout(0.15)(layer_h7)
        layer_h8 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                          kernel_initializer='he_normal')(layer_h7)  # 卷积层
        layer_h9 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h8)  # 池化层

        layer_h9 = Dropout(0.15)(layer_h9)
        layer_h10 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h9)  # 卷积层
        layer_h10 = Dropout(0.2)(layer_h10)
        layer_h11 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h10)  # 卷积层
        layer_h12 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h11)  # 池化层

        layer_h12 = Dropout(0.2)(layer_h12)
        layer_h13 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h12)  # 卷积层
        layer_h13 = Dropout(0.2)(layer_h13)
        layer_h14 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h13)  # 卷积层
        layer_h15 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h14)  # 池化层

        # test=Model(inputs = input_data, outputs = layer_h12)
        # test.summary()

        layer_h16 = Reshape((200, 3200))(layer_h15)  # Reshape层
        # layer_h5 = LSTM(256, activation='relu', use_bias=True, return_sequences=True)(layer_h4) # LSTM层
        # layer_h6 = Dropout(0.2)(layer_h5) # 随机中断部分神经网络连接，防止过拟合
        layer_h16 = Dropout(0.3)(layer_h16)
        layer_h17 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h16)  # 全连接层
        layer_h17 = Dropout(0.3)(layer_h17)
        layer_h18 = Dense(self.MS_OUTPUT_SIZE, use_bias=True, kernel_initializer='he_normal')(layer_h17)  # 全连接层

        y_pred = Activation('softmax', name='Activation0')(layer_h18)
        model_data = Model(inputs=input_data, outputs=y_pred)
        # model_data.summary()

        labels = Input(name='the_labels', shape=[self.label_max_string_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer

        # layer_out = Lambda(ctc_lambda_func,output_shape=(self.MS_OUTPUT_SIZE, ), name='ctc')([y_pred, labels, input_length, label_length])#(layer_h6) # CTC
        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')(
            [y_pred, labels, input_length, label_length])

        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

        # model.summary()

        # clipnorm seems to speeds up convergence
        # sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        # opt = Adadelta(lr = 0.01, rho = 0.95, epsilon = 1e-06)
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, epsilon=10e-8)
        # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)

        # captures output of softmax so we can decode the output during visualization
        test_func = K.function([input_data], [y_pred])

        # print('[*提示] 创建模型成功，模型编译成功')
        print('[*Info] Create Model Successful, Compiles Model Successful. ')
        return model, model_data

    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args

        y_pred = y_pred[:, :, :]
        # y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def TrainModel(self, datapath, epoch=2, save_step=1000, batch_size=32,
                   filename=abspath + 'model_speech/m' + ModelName + '/speech_model' + ModelName):
        '''
        训练模型
        参数：
            datapath: 数据保存的路径
            epoch: 迭代轮数
            save_step: 每多少步保存一次模型
            filename: 默认保存文件名，不含文件后缀名
        '''
        data = DataSpeech(datapath, 'train')

        num_data = data.GetDataNum()  # 获取数据的数量

        yielddatas = data.data_genetator(batch_size, self.AUDIO_LENGTH)

        for epoch in range(epoch):  # 迭代轮数
            print('[running] train epoch %d .' % epoch)
            n_step = 0  # 迭代数据数
            while True:
                try:
                    print('[message] epoch %d . Have train datas %d+' % (epoch, n_step * save_step))
                    # data_genetator是一个生成器函数

                    # self._model.fit_generator(yielddatas, save_step, nb_worker=2)
                    self._model.fit_generator(yielddatas, save_step)
                    n_step += 1
                except StopIteration:
                    print('[error] generator error. please check data format.')
                    break

                self.SaveModel(comment='_e_' + str(epoch) + '_step_' + str(n_step * save_step))
                self.TestModel(self.datapath, str_dataset='train', data_count=4)
                self.TestModel(self.datapath, str_dataset='dev', data_count=4)

    def LoadModel(self, filename=abspath + 'model_speech/m' + ModelName + '/speech_model' + ModelName + '.model'):
        '''
        加载模型参数
        '''
        self._model.load_weights(filename)
        self.base_model.load_weights(filename + '.base')
        # print(filename)
        # self._model=load_model(filename+'.h5')
        # self.base_model=load_model(filename+'.base.h5')

    def SaveModel(self, filename=abspath + 'model_speech/m' + ModelName + '/speech_model' + ModelName, comment=''):
        '''
        保存模型参数
        '''
        self._model.save_weights(filename + comment + '.model')
        self.base_model.save_weights(filename + comment + '.model.base')
        # 需要安装 hdf5 模块
        self._model.save(filename + comment + '.h5')
        self.base_model.save(filename + comment + '.base.h5')
        f = open('step' + ModelName + '.txt', 'w')
        f.write(filename + comment)
        f.close()

    def TestModel(self, datapath='', str_dataset='dev', data_count=32, out_report=False, show_ratio=True,
                  io_step_print=10, io_step_file=10):
        '''
        测试检验模型效果

        io_step_print
            为了减少测试时标准输出的io开销，可以通过调整这个参数来实现

        io_step_file
            为了减少测试时文件读写的io开销，可以通过调整这个参数来实现

        '''
        data = DataSpeech(self.datapath, str_dataset)
        # data.LoadDataList(str_dataset)
        num_data = data.GetDataNum()  # 获取数据的数量
        if (data_count <= 0 or data_count > num_data):  # 当data_count为小于等于0或者大于测试数据量的值时，则使用全部数据来测试
            data_count = num_data

        try:
            ran_num = random.randint(0, num_data - 1)  # 获取一个随机数

            words_num = 0
            word_error_num = 0

            nowtime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
            if (out_report == True):
                txt_obj = open('Test_Report_' + str_dataset + '_' + nowtime + '.txt', 'w', encoding='UTF-8')  # 打开文件并读入

            txt = '测试报告\n模型编号 ' + ModelName + '\n\n'
            for i in range(data_count):
                data_input, data_labels = data.GetData((ran_num + i) % num_data)  # 从随机数开始连续向后取一定数量数据

                # 数据格式出错处理 开始
                # 当输入的wav文件长度过长时自动跳过该文件，转而使用下一个wav文件来运行
                num_bias = 0
                while (data_input.shape[0] > self.AUDIO_LENGTH):
                    print('*[Error]', 'wave data lenghth of num', (ran_num + i) % num_data, 'is too long.',
                          '\n A Exception raise when test Speech Model.')
                    num_bias += 1
                    data_input, data_labels = data.GetData((ran_num + i + num_bias) % num_data)  # 从随机数开始连续向后取一定数量数据
                # 数据格式出错处理 结束

                pre = self.Predict(data_input, data_input.shape[0] // 8)

                words_n = data_labels.shape[0]  # 获取每个句子的字数
                words_num += words_n  # 把句子的总字数加上
                edit_distance = GetEditDistance(data_labels, pre)  # 获取编辑距离
                if (edit_distance <= words_n):  # 当编辑距离小于等于句子字数时
                    word_error_num += edit_distance  # 使用编辑距离作为错误字数
                else:  # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字
                    word_error_num += words_n  # 就直接加句子本来的总字数就好了

                if ((i % io_step_print == 0 or i == data_count - 1) and show_ratio == True):
                    # print('测试进度：',i,'/',data_count)
                    print('Test Count: ', i, '/', data_count)

                if (out_report == True):
                    if (i % io_step_file == 0 or i == data_count - 1):
                        txt_obj.write(txt)
                        txt = ''

                    txt += str(i) + '\n'
                    txt += 'True:\t' + str(data_labels) + '\n'
                    txt += 'Pred:\t' + str(pre) + '\n'
                    txt += '\n'

            # print('*[测试结果] 语音识别 ' + str_dataset + ' 集语音单字错误率：', word_error_num / words_num * 100, '%')
            print('*[Test Result] Speech Recognition ' + str_dataset + ' set word error ratio: ',
                  word_error_num / words_num * 100, '%')
            if (out_report == True):
                txt += '*[测试结果] 语音识别 ' + str_dataset + ' 集语音单字错误率： ' + str(word_error_num / words_num * 100) + ' %'
                txt_obj.write(txt)
                txt = ''
                txt_obj.close()

        except StopIteration:
            print('[Error] Model Test Error. please check data format.')

    def Predict(self, data_input, input_len):
        '''
        预测结果
        返回语音识别后的拼音符号列表
        '''


        batch_size = 1

        in_len = np.zeros((batch_size), dtype=np.int32)
        in_len[0] = input_len
        x_in = np.zeros((batch_size, 1600, self.AUDIO_FEATURE_LENGTH, 1), dtype=np.float)
        for i in range(batch_size):
            x_in[i, 0:len(data_input)] = data_input

        base_pred = self.base_model.predict(x=x_in)


        # print('base_pred:\n', base_pred)

        # y_p = base_pred
        # for j in range(200):
        #	mean = np.sum(y_p[0][j]) / y_p[0][j].shape[0]
        #	print('max y_p:',np.max(y_p[0][j]),'min y_p:',np.min(y_p[0][j]),'mean y_p:',mean,'mid y_p:',y_p[0][j][100])
        #	print('argmin:',np.argmin(y_p[0][j]),'argmax:',np.argmax(y_p[0][j]))
        #	count=0
        #	for i in range(y_p[0][j].shape[0]):
        #		if(y_p[0][j][i] < mean):
        #			count += 1
        #	print('count:',count)
        # pdb.set_trace()
        base_pred = base_pred[:, :, :]
        # base_pred =base_pred[:, 2:, :]
        # print(base_pred)
        # print(in_len)

        # r = K.ctc_decode(base_pred, in_len)
        # print('r', r)
        temp = base_pred.reshape(200, 1424)
        temp_arr = temp.argmax(1)
        arr_result=[]
        for k in range(200):
            # print(k)
            if temp_arr[k]!=1423:
                arr_result.append(temp_arr[k])
        np_arr=np.array(arr_result)
        # print(np_arr)

        # pdb.set_trace()
        # decode = K.function([base_pred, in_len], [r[0][0]])
        # r1 = K.get_value(r[0][0])
        # #
        # #
        # # print('r1', r1)
        # #
        # # # r2 = K.get_value(r[1])
        # # # print(r2)
        # #
        # # # del self.base_model
        # r1 = r1[0]
        # return r1
        return np_arr

    def RecognizeSpeech(self, wavsignal, fs):
        '''
        最终做语音识别用的函数，识别一个wav序列的语音
        不过这里现在还有bug
        '''

        # data = self.data
        # data = DataSpeech('E:\\语音数据集')
        # data.LoadDataList('dev')
        # 获取输入特征
        # data_input = GetMfccFeature(wavsignal, fs)
        # t0=time.time()

        # print(wavsignal)
        data_input = GetFrequencyFeature3(wavsignal, fs)
        # t1=time.time()
        # print('time cost:',t1-t0)

        input_length = len(data_input)
        # print(input_length)
        input_length = input_length // 8
        # print(input_length)

        data_input = np.array(data_input, dtype=np.float)
        # print(data_input,data_input.shape)
        data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
        # t2=time.time()
        r1 = self.Predict(data_input, input_length)
        # t3=time.time()
        # print('time cost:',t3-t2)
        r_str = []
        for i in r1:
            r_str.append(list_symbol_dic[i])

        return r_str


    def RecognizeSpeech_FromFile(self, filename1,text_file):
        '''
        最终做语音识别用的函数，识别指定文件名的语音
        '''
        # wavsignal, fs = read_wav_data(filename2)
        wavsignal = load_wav(filename1)
        fs=16000
        wavsignal = wavsignal * 32768
        wavsignal.shape=1,-1
        cut_num=3000
        num=wavsignal.shape[1]//cut_num
        print(wavsignal.shape[1])
        # from source_data.file_util import generate_word_list,generate_pinyin_list
        # word_list=generate_word_list()
        # sentencelen=[]
        # sentencepinyin=[]
        # for words in word_list:
        #     sentencelen.append(len(words))
        #     # print(len(words))
        #     sentencepinyin.append(generate_pinyin_list(words))
        # # sentencelen=[18,25,22]
        i=0
        k=0
        r_r=[]
        # =====================================================
        base = 0
        cut_num = 160000
        num = wavsignal.shape[1] // cut_num
        print(num)
        while 1:
            if i > 8:
                break
            if k<num:
                temp=wavsignal[:,base:base+cut_num]
                base=base+cut_num
                r = self.RecognizeSpeech(temp, fs)
                r_r.append(r)
                print(r)
            else:
                if wavsignal.shape[1]-base<400:
                    break
                else:
                    temp=wavsignal[:,base:]
                    r = self.RecognizeSpeech(temp, fs)
                    r_r.append(r)
                    print(r)
                break
            k=k+1
        step=0
        print('语音识别的数组是：',r_r)
        # ====================================================================================================
        r_f_r=[]
        for temp in r_r:
            for t_t in temp:
                r_f_r.append(t_t[:-1])
        #
        print('识别语音长度是===',len(r_f_r))
        s_f_s=[]
        print('识别的语音是===',r_f_r)
        # for s_f in  sentencepinyin:
        #     for pin in s_f:
        #         s_f_s.append(pin)
        from split import get_index, generate_word_list_from_text,generate_pinyin_list,generate_pinyin,reset,\
            get_str,get_str2,levenshtein,mkdir,get_first_right_index,generate_pinyin_from_text_array,generate_word_list
        # text_file='/media/wanggf/Data/ASR_T/data/5.8.txt'

        # word_list_have_biaodian,word_list_no_biaodao=generate_word_list_from_text(text_file)
        # print(word_list_no_biaodao)

        # #############################
        # 自动断句，输入的是文本文件
        # sentencepinyin,_=generate_pinyin(text_file)
        # #############################
        # 手动断句输入文本文件
        print('==-============文本文件名',text_file)
        text_arr=generate_word_list(text_file)
        print(text_arr)
        sentencepinyin, _=generate_pinyin_from_text_array(text_arr)
        print(sentencepinyin)
        reset()

        print('识别语音长度是===', len(sentencepinyin))
        _,generate_yuyin_arr = get_index(r_f_r, sentencepinyin)
        print('原有语音是===', sentencepinyin)
        print('生成语音是===',generate_yuyin_arr)
        generate_yuyin_arr_final=[]
        print('识别语音长度是===', len(r_f_r))

        # ############################
        # 对生成的语音根据位置做微调
        for k in range(len(sentencepinyin)):
            print('当前下标',k)
            # print(sentencepinyin[k])
            # print(generate_yuyin_arr[k])
            if len(generate_yuyin_arr[k])==len(sentencepinyin[k]):
                generate_yuyin_arr_final.append(generate_yuyin_arr[k])
                pass
            else:
                # for tt in generate_yuyin_arr[k]:
                #     for kk in generate_yuyin_arr[k]:
                #         if abs(tt-kk)<=3:
                #             break
                source_index,rec_index= get_first_right_index(sentencepinyin[k],generate_yuyin_arr[k])
                # print(len(sentencepinyin[k]))
                generate_yuyin_arr_final.append(generate_yuyin_arr[k])
                # print(generate_yuyin_arr_final)
                # print('====================')
                # print('正确的源文件下标：',source_index)
                # print(len(generate_yuyin_arr[k]))
                # print('正确的识别下标：',rec_index)
                # 正确的源文件下表等于需要识别的下表，说明没有错误
                if source_index==rec_index:
                    pass
                # 正确的源文件下表大于需要识别的下表，说明断开的地方太靠后
                elif source_index > rec_index:
                    dis=source_index-rec_index
                    dis=-dis
                    # print(dis)
                    # print(generate_yuyin_arr[k-1])
                    generate_yuyin_arr_final[k-1]=generate_yuyin_arr[k-1][:dis]
                    # print('修改后：')
                    # print(generate_yuyin_arr[k - 1])

                    temp=[]
                    # print(generate_yuyin_arr[k])
                    for i in range(abs(dis)):
                        temp.append(generate_yuyin_arr[k-1][i-2])
                    for key in generate_yuyin_arr[k]:
                        temp.append(key)
                    # print('当前数组',generate_yuyin_arr_final)
                    # print('修改后：',temp)
                    generate_yuyin_arr_final[k]=temp

                # 推到后面去
                # 推到前面去
                # 正确的源文件下表小于需要识别的下表，说明断开的地方太靠前
                else:
                    dis = abs(source_index - rec_index)
                    dis = -dis
                    print(generate_yuyin_arr[k])
                    generate_yuyin_arr_final[k] = generate_yuyin_arr[k][-dis:]
                    # print('修改后：', temp)
                    # print(generate_yuyin_arr[k])
                    temp = []

                    print(generate_yuyin_arr[k-1])
                    if k>0:
                        for key in generate_yuyin_arr[k-1]:
                            temp.append(key)
                        for i in range(abs(dis)):
                            temp.append(generate_yuyin_arr[k][i])
                        generate_yuyin_arr_final[k-1] = temp
                        # print('修改后：', temp)
                        # print(generate_yuyin_arr[k - 1])

        print('修改前的语音：',generate_yuyin_arr)
        print('修改后的语音：',generate_yuyin_arr_final)

        # #################
        # 获取微调后的拼音 generate_yuyin_arr_final
        generate_yuyin_arr=generate_yuyin_arr_final
        generate_arr=[]
        for temp in generate_yuyin_arr:
            for t_t in temp:
                generate_arr.append(t_t[:])
        stand_yuyin=[]
        for k in generate_yuyin_arr:
            stand_yuyin.append(get_str(k))
        # print('生成语音长度是===', len(generate_arr))
        # print(generate_arr)
        # print('=========开始准确识别==========')
        base=0
        cut=3333
        k=0
        i=0
        result_index=[]
        min_lenv=50
        # 开始最后跟读分割
        while 1:
            # 最后一次
            if i==len(stand_yuyin)-1:
                temp = wavsignal[:, base:]
                r = self.RecognizeSpeech(temp, fs)
                print('生成的第{}句话是{}'.format(i, get_str2(r)))
                print('原有语音是===', get_str(sentencepinyin[i]))
                break
            temp = wavsignal[:, base:base + cut * (k + 1)]
            temp_last = wavsignal[:, base:base + cut * k ]
            temp_next=wavsignal[:,base:base+cut*(k+2)]
            if k==0:
                temp_last=temp
            r = self.RecognizeSpeech(temp, fs)
            r_last = self.RecognizeSpeech(temp_last, fs)
            r_next=self.RecognizeSpeech(temp_next,fs)
            rec_str=get_str2(r)
            rec_str_last=get_str2(r_last)
            rec_str_next = get_str2(r_next)
            # print(get_str2(r))
            # print(base)
            # print(base + cut * (k + 1))
            len_cur=levenshtein(rec_str,stand_yuyin[i])
            if len_cur<min_lenv:
                min_lenv=len_cur
            len_last=levenshtein(rec_str_last,stand_yuyin[i])
            len_next = levenshtein(rec_str_next, stand_yuyin[i])
            # 最理想状况
            # print('==========================')
            # print(len_cur)
            # print(len_last)
            # print(len_next)
            # print('标准结果',stand_yuyin[i])
            # print('当前结果',rec_str)
            # print('前一个结果',rec_str_last )
            # print('后一个结果', rec_str_next)
            # print(len(rec_str_last) )
            # print( len(get_str(stand_yuyin[i])))
            if get_str2(r) == stand_yuyin[i]:
                print('和标准结果一致获得的结果')
                print('生成的第{}句话是{}'.format(i+1, get_str2(r)))
                print('原有语音是===', get_str(sentencepinyin[i]))
                result_index.append(base + cut * (k + 1))
                base = base + cut * (k + 1)
                i = i + 1
                k = 0
                min_lenv = 50
            # 当前和前一个一样，继续向前遍历
            if len_cur==len_last:
                k+=1
                continue
            #     当前和前一个不一样，如果编辑距离减小，说明下一个是该句的 ，如果编辑距离变大，说明下一个不是该句的，
            else    :
                # if len_cur>len_last and len_cur>min_lenv and  abs(len(rec_str_last)-len(get_str(stand_yuyin[i])))<=8:
                #     result_index.append(base + cut * k)
                #     print('生成的第{}句话是{}'.format(i + 1, rec_str_last))
                #     print('原有语音是===', get_str(sentencepinyin[i]))
                #     # print('第{}句话对比：{}'.format(i, rec_str))
                #     base = base + cut * k
                #     k = 0
                #     i += 1
                #     min_lenv=50
                #     continue
                # and abs(len(rec_str_last) - len(get_str(stand_yuyin[i]))) <= 2
                if len_cur>len_last and len_next>len_last and abs(len(rec_str_last) - len(get_str(stand_yuyin[i]))) <= 5:
                    result_index.append(base + cut * k)
                    print('生成的第{}句话是{}'.format(i + 1, rec_str_last))
                    print('原有语音是===', get_str(sentencepinyin[i]))
                    # print('第{}句话对比：{}'.format(i, rec_str))
                    base = base + cut * k
                    k = 0
                    i += 1
                    min_lenv = 50
                    continue

                # 根据文本长度设定误差阈值
                temp_len=len(rec_str_last)
                stand_pinyin_mid=0
                if temp_len <50:
                    stand_pinyin_mid=3
                else:
                    stand_pinyin_mid=(temp_len-50)/10+3
                #     根据当前语音片段和上次语音片段的差值和结合语音本身的文本长度判断是否要继续往下遍历
                if len_cur>len_last and abs(len(rec_str_last) - len(get_str(stand_yuyin[i]))) <= stand_pinyin_mid:
                    # print('通过比对获得结果')
                    result_index.append(base + cut * k )
                    print('生成的第{}句话是{}'.format(i+1, rec_str_last))
                    print('原有语音是===', get_str(sentencepinyin[i]))
                    # print('第{}句话对比：{}'.format(i, rec_str))
                    base=base+cut*k
                    k=0
                    i+=1
                    min_lenv = 50
                    continue
                #     其他情况继续往下遍历
                else:
                    k+=1
                    continue
        for k in result_index:
            print(k)

        #     #####################
        # 端点检测
        base=0
        audio_arr=[]
        audio = librosa.load(filename1, sr=16000)[0]
        for index in result_index :
            print('index is =======',index)
            audio_arr.append(audio[base:index])
            base=index
        if  len (result_index)==1:
            pass
        else:
            audio_arr.append(result_index[result_index[-1]:])
        base_dir = './sound_split_result/' + get_str(text_file[:-1])
        mkdir(base_dir)
        from  voice_seg.main_seg5 import  get_segpoint
        i=0
        result_index_final=[]
        for k in range(len(result_index)):
            segpoint=get_segpoint(audio_arr[k],audio_arr[k+1])
            print(k)
            print(segpoint)
            result_index_final.append(result_index[k]-8000+segpoint)
            # file_name = base_dir + '/' + str(i+1) + '.wav'
            # base_len=len(audio_arr[k])
            # librosa.output.write_wav(file_name, , 16000)
            i+=1
        base = 0
        i=100
        for index in result_index_final:
            #     print(base)
            file_name = base_dir + '/' + str(i) + '.wav'

            librosa.output.write_wav(file_name, audio[base:index], 16000)
            #     audio[base:index].export('/media/wanggf/Data/ASR_T/xunfei/sound_delimer/1/' + str(i) + '.wav', format='wav')
            # file_word_dic[file_name] = wordsource[i]
            base = index
            i += 1
        # print(result[-1])
        file_name = base_dir + '/' + str(i) + '.wav'
        librosa.output.write_wav(file_name, audio[result_index_final[-1]:], 16000)
        r_f_r=[]



#         #############     开始根据下标保存
#         file_word_dic = {}
#         # base_dir='./data2/sound_split_5.8.2/'
#         base_dir='./spund_split_data/'+get_str(text_file[-7:-1])
#         mkdir(base_dir)
#         i = 0
#         audio = librosa.load(filename1, sr=16000)[0]
#         base = 0
#         temp_index=0
#         # from vad2 import  get_no_voice_index
#         # for index in result_index:
#         #     file_name = base_dir + str(i) + '.wav'
#         #
#         #     temp_index = get_no_voice_index(audio[base:index])
#         #     # 如果没检测到静音：
#         #     if temp_index==0:
#         #         print('没事别到静音')
#         #         print(str(base) + ':' + str(index))
#         #         librosa.output.write_wav(file_name, audio[base:index], 16000)
#         #         base=index
#         #         i+=1
#         #     else:
#         #         print(str(base) + ':' + str(index))
#         #         print(str(base)+':'+str(temp_index+base))
#         #         print(temp_index)
#         #         librosa.output.write_wav(file_name, audio[base:temp_index+base], 16000)
#         #         base=temp_index+base
#         #         i+=1
#         # file_name = base_dir + str(i) + '.wav'
#         # librosa.output.write_wav(file_name, audio[temp_index :], 16000)
# ###################################################
#         base=0
#         for index in result_index:
#             #     print(base)
#             file_name = base_dir +'/'+ str(i) + '.wav'
#
#
#             librosa.output.write_wav(file_name, audio[base :index ], 16000)
#             #     audio[base:index].export('/media/wanggf/Data/ASR_T/xunfei/sound_delimer/1/' + str(i) + '.wav', format='wav')
#             # file_word_dic[file_name] = wordsource[i]
#             base = index
#             i += 1
#         # print(result[-1])
#         file_name = base_dir +'/'+ str(i) + '.wav'
#         librosa.output.write_wav(file_name, audio[result_index[-1] :], 16000)
#         # file_word_dic[file_name] = wordsource[i]
#         # audio[result[-1]:].export('/media/wanggf/Data/ASR_T/xunfei/sound_delimer/1/' + str(i) + '.wav', format='wav')
#         return file_word_dic


#################################################################################
        # i = 0
        # k = 0
        # num = wavsignal.shape[1] // cut_num
        # base = 1000
        # front = False
        # temp_front_len = 0
        # cut_num=1000
        # while 1:
        #     if i>7:
        #         break
        #     distance = 0
        #     if i==0:
        #         distance = index[i]
        #     else:
        #         distance = index[i] - index[i - 1]
        #     # step=(distance-5)//3 *16000
        #     # step=0
        #     temp = wavsignal[:, base:base +step+ cut_num * (k + 1)]
        #     # print(base)
        #     # print(base+cut_num*(k+1))
        #     r = self.RecognizeSpeech(temp, fs)
        #     # print(r)
        #     if k > num:
        #         break
        #     k = k + 1
        #
        #
        #     # if i==0:
        #     #
        #     # else :
        #
        #     # print(distance)
        #     # if r and len(r)>=2:
        #     #     print('i is ====={}'.format(i))
        #     #     print('r is ====={}'.format(r))
        #     #     print('识别到1的拼音===',r[-1][:-1])
        #     #     print('需要识别1的拼音====',r_f_r[index[i] - 1])
        #     #     print('识别到最后 2的拼音===', r[-2][:-1])
        #     #     print('需要识别最后2的拼音====', r_f_r[index[i] - 2])
        #     #     print('distance is ======={}'.format(distance))
        #     #     print(len(r))
        #     # if len(r)-distance>1:
        #     #     filename2 = r"./data3/temp" + str(i) + ".wav"
        #     #     save_wav(filename2, wavsignal_float[base:base + step + cut_num * (k + 1)])
        #     #     print(base + cut_num * (k + 1) + step)
        #     #     base = base + cut_num * (k + 1) + step
        #     #     r_r.append(r)
        #     #     k=0
        #     #     i=i+1
        #     # 如果最后一个的前1个是正确的，保存信息
        #     # if abs(len(r) - distance) <= 2 and r[-2][:-1] == r_f_r[index[i] - 2]:
        #     #     front=True
        #     #     temp_front_len=len(r)
        #     #     再向后面跑一个到最后的结论，打印
        #     if front and len(r)==temp_front_len+1:
        #         filename2 = r"./data3/temp/" + str(i) + ".wav"
        #         save_wav(filename2, wavsignal_float[base:base + step + cut_num * (k + 1)])
        #         print(base + cut_num * (k + 1) + step)
        #         base = base + cut_num * (k + 1) + step
        #         k = 0
        #         # print(base)
        #         # temp.shape = 1, -1
        #         print(r)
        #         print('已经保存第{}个文件'.format(i + 1))
        #         # filename=r"./data2/temp"+str(i)+".wav"
        #         r_r.append(r)
        #         # print(filename)
        #         # f = wave.open(filename, "wb")
        #         # #     # 配置声道数、量化位数和取样频率
        #         # f.setnchannels(1)
        #         i += 1
        #         front=False
        #         temp_front_len=0
        #     #     如果最后一个是正确的
        #     if abs(len(r) - distance)<=1 and r[-1][:-1]==r_f_r[index[i]-1]:
        #         # print(r)
        #         # print(r[-1][:-1])
        #         # print(r_f_r[index[i] - 1])
        #         #         #
        #         #         # 开始某个位置音符、中间随机某个位置音符、末尾随机位置拼音相似
        #         # global base
        #         # 保存文件
        #         filename2 = r"./data3/temp/" + str(i) + ".wav"
        #         save_wav(filename2, wavsignal_float[base:base +step+ cut_num * (k + 1)])
        #         print(base+cut_num * (k + 1)+step)
        #         base = base + cut_num * (k + 1)+step
        #         k = 0
        #         # print(base)
        #         # temp.shape = 1, -1
        #         print(r)
        #         print('已经保存第{}个文件'.format(i+1))
        #         # filename=r"./data2/temp"+str(i)+".wav"
        #         r_r.append(r)
        #         # print(filename)
        #         # f = wave.open(filename, "wb")
        #         # #     # 配置声道数、量化位数和取样频率
        #         # f.setnchannels(1)
        #         i += 1
        #         # f.setsampwidth(2)
        #         # f.setframerate(16000)
        #         # 将wav_data转换为二进制数据写入文件
        #         # f.writeframes(temp.tostring())
        #         # f.close()

        # return r_f_r
        pass

    @property
    def model(self):
        '''
        返回keras model
        '''
        return self._model


if (__name__ == '__main__'):

    # import tensorflow as tf
    # from keras.backend.tensorflow_backend import set_session
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 进行配置，使用95%的GPU
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.95
    # config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
    # set_session(tf.Session(config=config))

    datapath = abspath + ''
    modelpath = abspath + 'model_speech'

    if (not os.path.exists(modelpath)):  # 判断保存模型的目录是否存在
        os.makedirs(modelpath)  # 如果不存在，就新建一个，避免之后保存模型的时候炸掉

    system_type = plat.system()  # 由于不同的系统的文件路径表示不一样，需要进行判断
    if (system_type == 'Windows'):
        datapath = 'E:\\语音数据集'
        modelpath = modelpath + '\\'
    elif (system_type == 'Linux'):
        datapath = abspath + 'dataset'
        modelpath = modelpath + '/'
    else:
        print('*[Message] Unknown System\n')
        datapath = 'dataset'
        modelpath = modelpath + '/'

    # ms = ModelSpeech2(datapath)
    #
    # # ms.LoadModel(modelpath + 'm251/speech_model251_e_0_step_100000.model')
    # ms.TrainModel(datapath, epoch=50, batch_size=16, save_step=500)

# t1=time.time()
# ms.TestModel(datapath, str_dataset='train', data_count = 128, out_report = True)
# ms.TestModel(datapath, str_dataset='dev', data_count = 128, out_report = True)
# ms.TestModel(datapath, str_dataset='test', data_count = 128, out_report = True)
# t2=time.time()
# print('Test Model Time Cost:',t2-t1,'s')
# r = ms.RecognizeSpeech_FromFile('E:\\语音数据集\\ST-CMDS-20170001_1-OS\\20170001P00241I0053.wav')
# r = ms.RecognizeSpeech_FromFile('E:\\语音数据集\\ST-CMDS-20170001_1-OS\\20170001P00020I0087.wav')
# r = ms.RecognizeSpeech_FromFile('E:\\语音数据集\\wav\\train\\A11\\A11_167.WAV')
# r = ms.RecognizeSpeech_FromFile('E:\\语音数据集\\wav\\test\\D4\\D4_750.wav')
# print('*[提示] 语音识别结果：\n',r)
