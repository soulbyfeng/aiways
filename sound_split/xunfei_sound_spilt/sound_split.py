import json
import pinyin
import numpy
import heapq
import pydub
import librosa
import re
def generate_pinyin_list(words):
    pinyin_str = pinyin.get(words, format='numerical', delimiter=" ")
    temp_list = pinyin_str.strip().split(' ')[:]
    final_list = []
    for temp in temp_list:
        if len(temp)==1:
            final_list.append(temp)
        else:
            final_list.append(temp[:-1])
    return final_list
def cut_sent(para):
    para = re.sub('([，;。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([，;。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")
def del_biaodian(line):
    return re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！“”，：。？、~@#￥%……&*（）]+".encode('utf-8').decode('utf-8'), "".encode('utf-8').decode('utf-8'),line)
def is_legal_text(text):
    # text='我们的班主任小星老师是一位漂亮姑娘你好你是jsshfjkafgsajkgfakjshf，她笑诉我们，这座发动机比是咝咝咝咝咝咝i死死，人们管它叫“上帝的喷灯”。我们站在它巨大的阴影中，感受着大地转来的振动。小星老师让我们带上氧气面罩，随着光度和温度都在剧增，面罩的颜色渐渐变深。'
    result_biaodiao=cut_sent(text)
    # 无标点的list
    # print(result_biaodiao)
    for k in result_biaodiao:
        if len(k)>24:
            print('==========你输入的文本========\n{}\n已经超过设定的24个字'.format(text))
            print('具体文本：{}'.format(k))
            return False
    return True
def generate_word_list_from_text(text):
    # text='在人们眼前，还有一个无穷无尽的广阔领域，就像撒旦在高山上向救世主显示所有的那些世上的王国。对于那些在一生中永远感到饥渴的人，渴望着征服的人，人生就是这样：专注于获取更多的领地，专注于更宽阔的视野。军事远征诱惑着他们，而权力就是他们的乐趣。他们愿望就是使他们能更多地占据男人的头脑和女人的心。他们利用岁月，因而岁月并不使他们厌倦。'
    # 有标点的list
    # print('需要分割的音频信息：\n',text)
    result_biaodiao=cut_sent(text)
    # 无标点的list
    # print(result_biaodiao)
    word_list=[]
    base=0
    index=1
    is_first=True
    last_tem=''
    # 根据字数划分
    while 1:
        # print(base)
        if index>len(result_biaodiao)+1:
           break
        tem= get_str(result_biaodiao[base:index])
        # 第一次就保存当前信息
        if  is_first:
            last_tem=get_str(result_biaodiao[base:index])
        #     其他保存前一次信息
        else:
            last_tem = get_str(result_biaodiao[base:index-1])
        # print(result_biaodiao[base:index])
        # print(tem)
        # print(len(tem))
        # 判断当前的长度是否符合要求
        # 是，直接划分
        # if is_first and len(tem)>26:
        #     word_list.append(last_tem)
        #     base = index
        #     is_first = True
        if len(tem)<26:
            index+=1
            is_first=False
        else:
            word_list.append(last_tem)
            if is_first:
                base=index
            else:
                base=index-1
            is_first = True
    # if base!=len(result_biaodiao)-1:
    word_list.append(get_str(result_biaodiao[base:]))
    # print(word_list)
    # 打印需要识别的句子：
    i=1
    for k  in word_list:
        # print('文本分割第{}个句子：{}'.format(i,k))
        i=i+1
    # 去标点获取最终结果
    word_list_final = []
    for k in word_list:
        temp = del_biaodian(k)
        word_list_final.append(temp)
    #   最终的结果
    # print(word_list_final)
    return  word_list_final,word_list



    # for k in range(len(text)):

def generate_word_list():
    word_list = []
    file = '/media/wanggf/Data/ASR_T/source_data/word'
    with open(file,'r') as r:
        for line in r :
            # print(line.strip())
            word_list.append(line.strip())
    return word_list
def generate_biaodian_dict():
    biaodian_dict={}
    file='./biaodian.txt'
    with open(file,'r') as f:
        for line in f:
            line=line.strip()
            biaodian_dict[line]=line
    return biaodian_dict

fuhao=['，','!', ' ','。']
# 识别的语音
word_array=[]
# 识别的语音对应的字典
word_dic={}

# 语音下标对应的开始位置
word_index_begin_dic={}

pinyin_list=[]
biaodian_dict=generate_biaodian_dict()
def reset():
    global word_array ,word_dic,word_index_begin_dic,pinyin_list
    # 识别的语音
    word_array = []
    # 识别的语音对应的字典
    word_dic = {}

    # 语音下标对应的开始位置
    word_index_begin_dic = {}
    pinyin_list = []
def generate_pinyin(file_text):
    w_list,w_source=generate_word_list_from_text(file_text)
    # w_list=generate_word_list()
    for w in w_list:
        pinyin_list.append(generate_pinyin_list(w))
    return pinyin_list,w_source
def generate_pinyin_from_text_array(text_arr):
    for w in text_arr:
        pinyin_list.append(generate_pinyin_list(w))
    return pinyin_list,text_arr
def read_json(n_d):
    # with open(file_json,'r') as rf:
    #     n_d = json.load(rf)
    # n_d=js
    # print(n_d)

    real_data = n_d["data"]
    i=0
    for res in eval(real_data):
        # print('=====')
        # print(res)
        # print(res['onebest'])
        base_begin=int(res['bg'])
        base_end=int(res['ed'])
        resultList=res['wordsResultList']
        for k in resultList:
            # print(k)
            words=k['wordsName']
            words_bg=int(k['wordBg'])
            words_end=int(k['wordEd'])
            temp_dic={'begin':base_begin+words_bg*10,'end':base_begin+words_end*10,'index':i}
            if words in biaodian_dict:
                continue
            word_index_begin_dic[i]=  base_begin+words_bg*10
            i=i+1
            word_array.append(words)
            word_dic[words]=temp_dic
    # print(word_array)
    # print('====一共的字数====：',len(word_array))
        # print(resultList)
        # print(resultList[0])
        # print(resultList[1])
        # print(res.decode('utf-8'))
def levenshtein(source, target):
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = numpy.array(tuple(source))
    target = numpy.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = numpy.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = numpy.minimum(
            current_row[1:],
            numpy.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = numpy.minimum(
            current_row[1:],
            current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]

def get_str(arr):
    str=''
    for k in arr:
        str+=k
    return str
def find_min_index(nums):
    min_num_index_list = map(nums.index, heapq.nsmallest(3, nums))

# def get_result_text_arr(json_data, text):
#     read_json(json_data)
#     _,word_source =generate_pinyin(text)
# ........................
def get_result_text_arr(json_data,text_arr):

    read_json(json_data)
    #_,word_source =generate_pinyin(file_text)
    _,word_source=generate_pinyin_from_text_array(text_arr)
    # pinyin对照list
    pinyin_list_base = []
    for line in pinyin_list:
        tem_line = ''
        for i in line:
            tem_line += i
        # print(tem_line)
        pinyin_list_base.append(tem_line)
    # 识别的汉字对应的拼音list
    # print(pinyin_list_base)
    word_pinyin_array = []
    for k in word_array:
        temp_word = generate_pinyin_list(k)
        tem_line = ''
        for i in temp_word:
            tem_line += i
        # print(tem_line)
        word_pinyin_array.append(tem_line)
    print(word_pinyin_array)
    base_step = 4
    is_end = False
    is_start = True
    index = 0
    i = 0
    base = 0
    final_index = []
    # while not is_end:
    # 找编辑距离最小且长度最长的下标
    base = 0
    while 1:
        #         # 需要对比的拼音
        if i >= len(pinyin_list_base):
            break
        pinyin_base = pinyin_list_base[i]
        distance_arr = []
        step = 0
        # print('第{}句话：'.format(i))
        # print(pinyin_base)
        # print('====需要跑的次数', len(word_pinyin_array) - base)
        k = 0
        while True:
            if k > len(word_pinyin_array) - base:
                break
            word_temp = word_pinyin_array[base:base + step]
            word_temp = get_str(word_temp)
            # print('当前句子',word_temp)
            dis = levenshtein(pinyin_base, word_temp)
            # print(dis)
            distance_arr.append(dis)
            step += 1
            k = k + 1
        # print(distance_arr)
        # print('长度是',len(distance_arr))
        # 最小元素的下标
        min_num_index = distance_arr.index(min(distance_arr))
        print('识别的汉字：',get_str(word_array[base:base + min_num_index]))
        print('识别的拼音：', get_str(word_pinyin_array[base:base + min_num_index]))
        # print('最小下标是', min_num_index + base)
        final_index.append(base + min_num_index)
        base = min_num_index + base
        i += 1
        # for key in distance_arr:
        #     print(k)
    print(final_index)
    # 根据下标划分句子 得到最终句子划分序列
    start = 0
    result = []
    k = 0
    for index in final_index:
        if k > len(final_index):
            break
        # print(word_array[start:index])
        # print(index)
        # print(temp)
        for key in word_index_begin_dic:
            # print(key)
            # 下标一样
            # print(word_dic.get(key)['index'])
            if key == index:
                # print(word_index_begin_dic.get(key))
                result.append(word_index_begin_dic.get(key))
        k += 1
    print(result)
    return result,word_source
def mkdir(path):
    import os
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
    # 如果目录不存在则创建该目录
        os.makedirs(path)
        return True
    else:
    # 如果目录存在则不创建，并提示目录已存在
        print(path,'目录已存在')
        return False


if __name__ == '__main__':
    generate_word_list_from_text()
    # is_legal_text()
    pass
    # for k in range(16):
    #     file2='/media/wanggf/Data/ASR_T/data/paragraphs/'+str(k+1)+'.wav'
    #     file_json='/media/wanggf/Data/ASR_T/xunfei/json_file/'+str(k+1)+'.json'
    #     result=get_result(file_json)
    #     base = 0
    #     audio=librosa.load(file2,sr=16000)[0]
    #     dir_base='/media/wanggf/Data/ASR_T/xunfei/sound/'+str(k+1)+'/'
    #     mkdir(dir_base)
    # # print(audio[0].shape)
    # # audio = pydub.AudioSegment.from_wav(file)
    #     i = 0
    #     for index in result:
    # #     print(base)
    #         librosa.output.write_wav(dir_base + str(i)
    #                              + '.wav',audio[base*16:index*16],16000)
    # #     audio[base:index].export('/media/wanggf/Data/ASR_T/xunfei/sound_delimer/1/' + str(i) + '.wav', format='wav')
    #         base = index
    #         i += 1
    # # print(result[-1])
    #     librosa.output.write_wav(dir_base + str(i)
    #                          + '.wav', audio[result[-1] * 16:], 16000)
    # # audio[result[-1]:].export('/media/wanggf/Data/ASR_T/xunfei/sound_delimer/1/' + str(i) + '.wav', format='wav')
    #     reset()







