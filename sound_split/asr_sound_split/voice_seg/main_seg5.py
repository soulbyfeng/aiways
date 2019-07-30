#! /usr/bin/python

# Voice Activity Detection (VAD) tool.
# use the vad_help() function for instructions.
# Navid Shokouhi December 2012.

# Updated: May 2017 for Speaker Recognition collaboration.

from voice_seg.audio_tools import *
import numpy as np
import matplotlib.pyplot as plt
import librosa
import itertools

##Function definitions:
def vad_help():
    """Voice Activity Detection (VAD) tool.
	
	Navid Shokouhi May 2017.
    """
    print("Usage:")
    print("python unsupervised_vad.py")

#### Display tools
def plot_this(s,title=''):
    """
     
    """
    import pylab
    s = s.squeeze()
    if s.ndim ==1:
        pylab.plot(s)
    else:
        pylab.imshow(s,aspect='auto')
        pylab.title(title)
    pylab.show()

def plot_these(s1,s2):
    import pylab
    try:
        # If values are numpy arrays
        pylab.plot(s1/max(abs(s1)),color='red')
        pylab.plot(s2/max(abs(s2)),color='blue')
    except:
        # Values are lists
        pylab.plot(s1,color='red')
        pylab.plot(s2,color='blue')
    pylab.legend()
    pylab.show()


#### Energy tools
def zero_mean(xframes):
    """
        remove mean of framed signal
        return zero-mean frames.
        """
    m = np.mean(xframes,axis=1)
    xframes = xframes - np.tile(m,(xframes.shape[1],1)).T
    return xframes

def compute_nrg(xframes):
    # calculate per frame energy
    n_frames = xframes.shape[1]
    return np.diagonal(np.dot(xframes,xframes.T))/float(n_frames)

def compute_log_nrg(xframes):
    # calculate per frame energy in log
    n_frames = xframes.shape[1]
    raw_nrgs = np.log(compute_nrg(xframes+1e-5))/float(n_frames)
    return (raw_nrgs - np.mean(raw_nrgs))/(np.sqrt(np.var(raw_nrgs)))

def power_spectrum(xframes):
    """
        x: input signal, each row is one frame
        """
    X = np.fft.fft(xframes,axis=1)
    X = np.abs(X[:,:X.shape[1]/2])**2
    return np.sqrt(X)



def nrg_vad(xframes,percent_thr,nrg_thr=0.,context=5):
    """
        Picks frames with high energy as determined by a 
        user defined threshold.
        
        This function also uses a 'context' parameter to
        resolve the fluctuative nature of thresholding. 
        context is an integer value determining the number
        of neighboring frames that should be used to decide
        if a frame is voiced.
        
        The log-energy values are subject to mean and var
        normalization to simplify the picking the right threshold. 
        In this framework, the default threshold is 0.0
        """
    xframes = zero_mean(xframes)
    n_frames = xframes.shape[1]
    
    # Compute per frame energies:
    xnrgs = compute_log_nrg(xframes)
    xvad = np.zeros((n_frames,1))
    for i in range(n_frames):
        start = max(i-context,0)
        end = min(i+context,n_frames-1)
        n_above_thr = np.sum(xnrgs[start:end]>nrg_thr)
        n_total = end-start+1
        xvad[i] = 1.*((float(n_above_thr)/n_total) > percent_thr)
    return xvad

def float2binary(voice_list):
    binary_voice=[0 for i in range(len(voice_list))]
    for i in range(len(voice_list)):
        if voice_list[i]>-0.5:
            binary_voice[i]=1
        else:
            binary_voice[i]=0
        #print('----i=',i)
    #print('-------binary_voice=',binary_voice)
    return binary_voice

def seg_fine(voice,fs):
    #fs=sr
    #fs=16000
    s=voice
    win_len = int(fs*0.025)
    hop_len = int(fs*0.010)
    sframes = enframe(s,win_len,hop_len) # rows: frame index, cols: each frame
    print('------------len of sframes---------\n',len(sframes))
    #print('------------compute_log_nrg(sframes)=',compute_log_nrg(sframes))
    # print('------------len of compute_log_nrg(sframes)=',len(compute_log_nrg(sframes)))
    rnum=float2binary(compute_log_nrg(sframes))
    print('------------rnum------------\n',rnum)

    #l = [(k, len(list(g))) for k, g in itertools.groupby('TTFTTTFFFFTFFTT')]
    l = [(k, len(list(g))) for k, g in itertools.groupby(rnum)]
    print('-----过滤前的数组------',l)
    
    #l_filter=[[] for i in range(len(l))]
    point_pos=0
    for k in range(len(l)):
       point_pos=point_pos+l[k][1]
       # print('---point_pos=',point_pos)
       # 1的个数少于5 修整
       if (l[k][1]<=5) and (l[k][0]==1):
          tst=[0 for s in range(l[k][1])]
          # print('tst=',tst)
          # print(point_pos)
          # rnum[point_pos-2:point_pos+l[k][1]]=tst
          rnum[point_pos - l[k][1]:point_pos ] = tst
          # print('----rnum3=',rnum)
          # print(len(rnum))
       else:
          pass
          #print('----rnum4=',rnum)
    
    # print('----rnum5=',rnum)
    # 重新遍历
    l_filter = [(k, len(list(g))) for k, g in itertools.groupby(rnum)]
    print('-----过滤后的数组是-------------',l_filter)
    ##########################################################3
    #找到最大间隔 和最大间隔对应的下标
    max_index=0
    max_number=0
    for k in range(len(l_filter)):
        if l_filter[k][0]==1:
            continue
        else:
            # 取出下标和最大值比较
            if (l_filter[k][1]>=max_number):
                max_number=l_filter[k][1]
                max_index=k

    # print('最大的静音下标是：',max_index)
    # print('最大的静音区间是：',max_number)
    # 取出最大间隔的index
    k=0
    sepoint=0
    while 1 :
        if k==max_index:
            sepoint=sepoint+l_filter[k][1]/2
            break
        else:
            sepoint=sepoint+l_filter[k][1]
        k+=1

    sepoint=int(sepoint)*160
    # print(sepoint)

    # ######################## james代码############################
    # l_record=[[] for i in range(len(l_filter))]
    # for k in range(len(l_filter)):
    #    print(k)
    #    #  统计0的下标
    #    # 如果是1 就标为0
    #    # 如果为0  就记录下标位置
    #    if l_filter[k][0]==1:
    #      l_record[k]=0
    #    else:
    #      l_record[k]=l_filter[k][1]
    #    print('-----l[k][0]=',l_filter[k][0])
    #    print('-----l[k][1]=',l_filter[k][1])
    # print('--l_record=',l_record)
    # # 获取0下标的最大值
    # v_location=l_record.index(max(l_record))
    # print('--v_location=',v_location)
    #
    # sep_loc=0
    # print(l)
    # for j in range(v_location):
    #    sep_loc=sep_loc+l[j][1]
    #    print('--sep_loc1=',sep_loc)
    # sep_loc=sep_loc+int(l_filter[v_location][1]/2)
    # print('--sep_loc=',sep_loc)
    # sepoint=160*sep_loc
    # #############################################################

    # plt.figure(2)
    # plt.plot(rnum)
    # plt.show()

    return sepoint
def get_segpoint(y,y2):
    wave_all = np.concatenate((y, y2))
    len_y = len(y)
    voice = wave_all[len_y - 8000:len_y + 8000]
    sepoint = seg_fine(voice, 16000)
    return sepoint

if __name__=='__main__':
    input_path1='./data/3.wav'
    input_path2='./data/4.wav'
    y, sr = librosa.load(input_path1, sr=None)
    y2, sr2 = librosa.load(input_path2, sr=None)
    wave_all=np.concatenate((y,y2))
    print(len(y))
    print(len(y2))
    len_y=len(y)
    len_y2=len(y2)
    # voice=wave_all[len_y-8000:len_y+8000]
    #
    sepoint=get_segpoint(y,y2)
    print('=============\nseg point is ',sepoint)
    #
    good_v1=wave_all[:len_y-8000+sepoint]
    good_v2=wave_all[len_y-8000+sepoint+1:]
    librosa.output.write_wav(input_path1[:-4]+'_new.wav', good_v1,16000)
    librosa.output.write_wav(input_path2[:-4]+'_new.wav', good_v2,16000)
    #

    '''
    statistics_zeros2(rnum)
    plt.figure(1)
    plt.plot(compute_log_nrg(sframes))
    plt.show()
    '''
    
    #plot_this(compute_log_nrg(sframes))    
    # percent_high_nrg is the VAD context ratio. It helps smooth the
    # output VAD decisions. Higher values are more strict.
    # percent_high_nrg = 0.5
    
    #vad = nrg_vad(sframes,percent_high_nrg)
    # plot_these(deframe(vad,win_len,hop_len),s)

'''
    #fs=sr
    fs=16000
    s=voice
    win_len = int(fs*0.025)
    hop_len = int(fs*0.010)
    sframes = enframe(s,win_len,hop_len) # rows: frame index, cols: each frame
    print('------------len of sframes=',len(sframes))
    #print('------------compute_log_nrg(sframes)=',compute_log_nrg(sframes))
    print('------------len of compute_log_nrg(sframes)=',len(compute_log_nrg(sframes)))
    rnum=float2binary(compute_log_nrg(sframes))
    print('------------rnum=',rnum)

    #l = [(k, len(list(g))) for k, g in itertools.groupby('TTFTTTFFFFTFFTT')]
    l = [(k, len(list(g))) for k, g in itertools.groupby(rnum)]
    print('-----l=',l)
    
    #l_filter=[[] for i in range(len(l))]
    point_pos=0
    for k in range(len(l)):
       point_pos=point_pos+l[k][1]
       print('---point_pos=',point_pos)
       if (l[k][1]<=5) and (l[k][0]==1):
          tst=[0 for s in range(l[k][1])]
          print('tst=',tst)
          rnum[point_pos-2:point_pos+l[k][1]]=tst
          #print('----rnum3=',rnum)
       else:
          pass
          #print('----rnum4=',rnum)
    
    print('----rnum5=',rnum)
    l_filter = [(k, len(list(g))) for k, g in itertools.groupby(rnum)]
    print('-----l_filter=',l_filter)
    
    l_record=[[] for i in range(len(l_filter))]
    for k in range(len(l_filter)):
       if l_filter[k][0]==1:
         l_record[k]=0
       else:
         l_record[k]=l_filter[k][1]
       print('-----l[k][0]=',l_filter[k][0])
       print('-----l[k][1]=',l_filter[k][1])
    print('--l_record=',l_record)
    v_location=l_record.index(max(l_record))
    print('--v_location=',v_location)

    sep_loc=0
    for j in range(v_location):
       sep_loc=sep_loc+l[j][1]
       print('--sep_loc1=',sep_loc)
    sep_loc=sep_loc+int(l_filter[v_location][1]/2)
    print('--sep_loc=',sep_loc)
    sepoint=160*sep_loc
'''




