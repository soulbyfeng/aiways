# -*- coding: utf-8 -*-
# 
#   author: yanmeng2
# 
# 非实时转写调用demo

import base64
import hashlib
import hmac
import json
import os
import time
import librosa

import requests
from sound_split import get_result_text_arr,reset,generate_word_list_from_text
# from sound_split.sound_split import get_result_text_arr,reset,generate_word_list_from_text
# from sound_split_0519.sound_split  import get_result,reset,generate_word_list_from_text
lfasr_host = 'http://raasr.xfyun.cn/api'

# 请求的接口名
api_prepare = '/prepare'
api_upload = '/upload'
api_merge = '/merge'
api_get_progress = '/getProgress'
api_get_result = '/getResult'
# 文件分片大下52k
file_piece_sice = 10485760

# ——————————————————转写可配置参数————————————————
# 参数可在官网界面（https://doc.xfyun.cn/rest_api/%E8%AF%AD%E9%9F%B3%E8%BD%AC%E5%86%99.html）查看，根据需求可自行在gene_params方法里添加修改
# 转写类型
lfasr_type = 0
# 是否开启分词
has_participle = 'false'
has_seperate = 'true'
# 多候选词个数
max_alternatives = 0
# 子用户标识
suid = ''


class SliceIdGenerator:
    """slice id生成器"""

    def __init__(self):
        self.__ch = 'aaaaaaaaa`'

    def getNextSliceId(self):
        ch = self.__ch
        j = len(ch) - 1
        while j >= 0:
            cj = ch[j]
            if cj != 'z':
                ch = ch[:j] + chr(ord(cj) + 1) + ch[j + 1:]
                break
            else:
                ch = ch[:j] + 'a' + ch[j + 1:]
                j = j - 1
        self.__ch = ch
        return self.__ch


class RequestApi(object):
    def __init__(self, appid, secret_key, upload_file_path):
        self.appid = appid
        self.secret_key = secret_key
        self.upload_file_path = upload_file_path

    # 根据不同的apiname生成不同的参数,本示例中未使用全部参数您可在官网(https://doc.xfyun.cn/rest_api/%E8%AF%AD%E9%9F%B3%E8%BD%AC%E5%86%99.html)查看后选择适合业务场景的进行更换
    def gene_params(self, apiname, taskid=None, slice_id=None):
        appid = self.appid
        secret_key = self.secret_key
        upload_file_path = self.upload_file_path
        ts = str(int(time.time()))
        m2 = hashlib.md5()
        m2.update((appid + ts).encode('utf-8'))
        md5 = m2.hexdigest()
        md5 = bytes(md5, encoding='utf-8')
        # 以secret_key为key, 上面的md5为msg， 使用hashlib.sha1加密结果为signa
        signa = hmac.new(secret_key.encode('utf-8'), md5, hashlib.sha1).digest()
        signa = base64.b64encode(signa)
        signa = str(signa, 'utf-8')
        file_len = os.path.getsize(upload_file_path)
        file_name = os.path.basename(upload_file_path)
        param_dict = {}

        param_dict['max_alternatives'] = 5
        param_dict['has_participle'] = 'true'

        if apiname == api_prepare:
            # slice_num是指分片数量，如果您使用的音频都是较短音频也可以不分片，直接将slice_num指定为1即可
            slice_num = int(file_len / file_piece_sice) + (0 if (file_len % file_piece_sice == 0) else 1)
            param_dict['app_id'] = appid
            param_dict['signa'] = signa
            param_dict['ts'] = ts
            param_dict['file_len'] = str(file_len)
            param_dict['file_name'] = file_name
            param_dict['slice_num'] = str(slice_num)
        elif apiname == api_upload:
            param_dict['app_id'] = appid
            param_dict['signa'] = signa
            param_dict['ts'] = ts
            param_dict['task_id'] = taskid
            param_dict['slice_id'] = slice_id
        elif apiname == api_merge:
            param_dict['app_id'] = appid
            param_dict['signa'] = signa
            param_dict['ts'] = ts
            param_dict['task_id'] = taskid
            param_dict['file_name'] = file_name
        elif apiname == api_get_progress or apiname == api_get_result:
            param_dict['app_id'] = appid
            param_dict['signa'] = signa
            param_dict['ts'] = ts
            param_dict['task_id'] = taskid
        return param_dict

    # 请求和结果解析，结果中各个字段的含义可参考：https://doc.xfyun.cn/rest_api/%E8%AF%AD%E9%9F%B3%E8%BD%AC%E5%86%99.html
    def gene_request(self, apiname, data, files=None, headers=None):
        response = requests.post(lfasr_host + apiname, data=data, files=files, headers=headers)
        result = json.loads(response.text)
        if result["ok"] == 0:
            # print("{} success:".format(apiname) + str(result))
            return result
        else:
            # print("{} error:".format(apiname) + str(result))
            exit(0)
            return result

    # 预处理
    def prepare_request(self):
        return self.gene_request(apiname=api_prepare,
                                 data=self.gene_params(api_prepare))

    # 上传
    def upload_request(self, taskid, upload_file_path):
        file_object = open(upload_file_path, 'rb')
        try:
            index = 1
            sig = SliceIdGenerator()
            while True:
                content = file_object.read(file_piece_sice)
                if not content or len(content) == 0:
                    break
                files = {
                    "filename": self.gene_params(api_upload).get("slice_id"),
                    "content": content
                }
                response = self.gene_request(api_upload,
                                             data=self.gene_params(api_upload, taskid=taskid,
                                                                   slice_id=sig.getNextSliceId()),
                                             files=files)
                if response.get('ok') != 0:
                    # 上传分片失败
                    # print('upload slice fail, response: ' + str(response))
                    return False
                # print('upload slice ' + str(index) + ' success')
                index += 1
        finally:
            'file index:' + str(file_object.tell())
            file_object.close()
        return True

    # 合并
    def merge_request(self, taskid):
        return self.gene_request(api_merge, data=self.gene_params(api_merge, taskid=taskid))

    # 获取进度
    def get_progress_request(self, taskid):
        return self.gene_request(api_get_progress, data=self.gene_params(api_get_progress, taskid=taskid))

    # 获取结果
    def get_result_request(self, taskid):
        return self.gene_request(api_get_result, data=self.gene_params(api_get_result, taskid=taskid))

    def all_api_request(self,text_arr):
        # 1. 预处理
        pre_result = self.prepare_request()
        start=time.time()
        print('开始调用迅飞语音识别api，请耐心等待，耗时约30秒')
        taskid = pre_result["data"]
        # 2 . 分片上传
        self.upload_request(taskid=taskid, upload_file_path=self.upload_file_path)
        # 3 . 文件合并
        self.merge_request(taskid=taskid)
        # 4 . 获取任务进度
        while True:
            # 每隔20秒获取一次任务进度
            progress = self.get_progress_request(taskid)
            progress_dic = progress
            if progress_dic['err_no'] != 0 and progress_dic['err_no'] != 26605:
                # print('task error: ' + progress_dic['failed'])
                return
            else:
                data = progress_dic['data']
                task_status = json.loads(data)
                if task_status['status'] == 9:
                    # print('task ' + taskid + ' finished')
                    break
                # print('The task ' + taskid + ' is in processing, task status: ' + str(data))

            # 每次获取进度间隔20S
            time.sleep(20)
        # 5 . 获取结果
        end=time.time()
        print('=====迅飞语音识别结束=====\n。耗时：{}'.format(end-start))
        dict = self.get_result_request(taskid=taskid)
        # print('======开始保存文件======',file)
        # with open(file,'w') as wf:
        #     json.dump(dict,wf,ensure_ascii=False)
        result,word_source=get_result_text_arr(dict,text_arr)
        reset()
        return result,word_source
    # 输入文本和音频整段目录
    # 返回音频分割目录，文本分割序列
    # def exec_split(self,text,sound_file):
    #     sound_name=sound_file.replace('/','')
    #     base_dir = '/home/wanggf/Github/sound_split/xunfei/sound_split_file/' + sound_name + '/'
    #     api = RequestApi(appid="5cdd6758", secret_key="80b4634b795fc7f2acc0e71d0bfc7945", upload_file_path=sound_file)
    #     result_index, word_source = api.all_api_request(text)
    #     api.save_file(base_dir, result_index, sound_file,word_source)

    def generate_result_dic(self,base_dir,temp,audio_file,wordsource):
        def mkdir( path):
            import os
            path = path.strip()
            path = path.rstrip("\\")
            isExists = os.path.exists(path)
            if not isExists:
                # 如果目录不存在则创建该目录
                os.makedirs(path)
                return True
            else:
                # 如果目录存在则不创建，并提示目录已存在
                print(path, '目录已存在')
                return False
        file_word_dic={}
        mkdir(base_dir)
        i=0
        audio = librosa.load(audio_file, sr=16000)[0]
        base=0
        for index in temp:
    #     print(base)
            file_name=base_dir + str(i)+ '.wav'
            librosa.output.write_wav(file_name,audio[base*16:index*16],16000)
    #     audio[base:index].export('/media/wanggf/Data/ASR_T/xunfei/sound_delimer/1/' + str(i) + '.wav', format='wav')
            file_word_dic[file_name]=wordsource[i]
            base = index
            i += 1
    # print(result[-1])
        file_name = base_dir + str(i) + '.wav'
        librosa.output.write_wav(file_name, audio[temp[-1] * 16:], 16000)
        file_word_dic[file_name] = wordsource[i]
    # audio[result[-1]:].export('/media/wanggf/Data/ASR_T/xunfei/sound_delimer/1/' + str(i) + '.wav', format='wav')
        reset()
        return file_word_dic
    ####################################################################
 # 输入文本和音频整段目录
# 返回音频分割目录=====>文本分割内容的dic
# def exec_split(text,sound_file):
#         sound_name=sound_file.replace('/','')
#         base_dir = '/home/wanggf/Github/sound_split/xunfei/sound_split_file/' + sound_name + '/'
#         api = RequestApi(appid="5cdd6758", secret_key="80b4634b795fc7f2acc0e71d0bfc7945", upload_file_path=sound_file)
#         result_index, word_source = api.all_api_request(text)
#         result=api.generate_result_dic(base_dir, result_index, sound_file,word_source)
#         return  result
#
# ####################################################################
# 输入文本文件 音频文件目录
# 输出音频分割下标，音频总长度
def exec_split(text_file,sound_file):
    api = RequestApi(appid="5cdd6758", secret_key="80b4634b795fc7f2acc0e71d0bfc7945", upload_file_path=sound_file)
    text_arr=[]
    with open( text_file ,'r') as r :
        for line in r :
            text_arr.append(line.strip())
    print(text_arr)
    result_index, word_source = api.all_api_request(text_arr)
    max_len=len(librosa.load(sound_file, sr=16000)[0])
    return result_index,max_len



#         sound_name=sound_file.replace('/','')
#         base_dir = '/home/wanggf/Github/sound_split/xunfei/sound_split_file/' + sound_name + '/'
#         api = RequestApi(appid="5cdd6758", secret_key="80b4634b795fc7f2acc0e71d0bfc7945", upload_file_path=sound_file)
#         result_index, word_source = api.all_api_request(text)
#         result=api.generate_result_dic(base_dir, result_index, sound_file,word_source)
#         return  result
# 注意：如果出现requests模块报错："NoneType" object has no attribute 'read', 请尝试将requests模块更新到2.20.0或以上版本(本demo测试版本为2.20.0)
# 输入讯飞开放平台的appid，secret_key和待转写的文件路径
if __name__ == '__main__':
    # read_file = '/media/wanggf/Data/ASR_T/data/paragraphs/2.wav'
    # api = RequestApi(appid="5cdd6758", secret_key="80b4634b795fc7f2acc0e71d0bfc7945", upload_file_path=read_file)
    # save_file='a.txt'
    # api.all_api_request(save_file)
    # for i in range(16):
    #     print('========开始第{}个文件的生成======'.format(i+1))
    #     read_file='/media/wanggf/Data/ASR_T/data/paragraphs/'+str(i+1)+'.wav'
    #     api = RequestApi(appid="5cdd6758", secret_key="80b4634b795fc7f2acc0e71d0bfc7945", upload_file_path=read_file)
    # #     save_file='/media/wanggf/Data/ASR_T/xunfei/json_file/'+str(i+1)+'.json'
    #     result_index=api.all_api_request()
    #     base_dir='/media/wanggf/Data/ASR_T/xunfei/sound5/'+str(i+1)+'/'
    #     api.save_file(base_dir,result_index,read_file)
    # read_file='./sound_source/5.4.m4a'
    # index=read_file.replace('.wav','').replace('./sound_source/','')
    # # print(temp)
    # text='在人们眼前，还有一个无穷无尽的广阔领域，就像撒旦在高山上向救世主显示所有的那些世上的王国。对于那些在一生中永远感到饥渴的人，渴望着征服的人，人生就是这样：专注于获取更多的领地，专注于更宽阔的视野。军事远征诱惑着他们，而权力就是他们的乐趣。他们愿望就是使他们能更多地占据男人的头脑和女人的心。他们是不知足的，强有力的。他们利用岁月，因而岁月并不使他们厌倦。'
    # text='哪有一帆风顺的人生，人生的确很累，看你如何品味；世界不完美，生活也就是难免有缺憾。幸福是一种对照，因为流过泪，所以笑得更甜美；因为经历过痛，所以更珍惜幸福；收起抱怨的心，让心灵的镜子照向光明，让黑暗躲到角落里，给自己一个微笑。'
    # text='正确的充电方式可以延长电池的使用寿命，建议车主们多采用慢充方式，这样可以延长电池的使用年限，电池的最佳充电温度是20度左右，千万不要在高温暴晒下给电池充电，这样会造成电池的损害；而低温下充电，虽然不会造成电池的损害，但是会使充电时间变得更长，所以日常充电需要注意环境温度哦。'
    # base_dir='/home/wanggf/Github/sound_split/xunfei/sound_split_file/'+index+'/'
    # api = RequestApi(appid="5cdd6758", secret_key="80b4634b795fc7f2acc0e71d0bfc7945", upload_file_path=read_file)
    # result_index,word_source = api.all_api_request(text)
    # api.save_file(base_dir, result_index, read_file)
    # print(word_source)
    #     save_file='/media/wanggf/Data/ASR_T/xunfei/json_file/'+str(i+1)+'.json'
    # text='一个人如果对自己的事业充满热爱，并确定了自己的工作愿望，就会自发地尽自己最大的努力去工作。如果一个人的一生当中没有任何目标，那他就会迷失自己。成功的人之所以成功，是因为将有限的精力专注到一个领域，每天做好一件事，日积月累成就伟业。'
    # text='如果失去梦想，人类将会怎样？不要怀有渺小的梦想，它们无法打动人心，一块砖没有什么用，一堆砖也没有什么用，如果你心中没有一个造房子的梦想，拥有天下所有的砖头也是一堆废物；但如果你只有造房子的梦想，而没有造房子的砖头，梦想也没法实现。'
    # text='别把人生想的太难，给自己一份乐观。走过生命的逆旅，人世沧桑，谁都会彷徨，会忧伤，会有苦雨寒箫的幽怨，也会有月落乌啼的悲凉，但有限的生命，不允许我们挥霍那份属于人生的酸甜苦辣，经历了寒风阴霾的苦砺，才会破茧在阳光明媚的日子。'
    # sound_file='/home/wanggf/Github/sound_split/sound_split_0519/sound_source/4.4.m4a'
    # _,text_arr=generate_word_list_from_text(text)
    # print(text_arr)
    sound_file='/home/wanggf/Github/sound_split/sound_split_0519/sound_source/3.1.m4a'
    text_file='/home/wanggf/Github/sound_split/sound_split_0519/text.txt'
    result_arr,maxlen=exec_split(text_file,sound_file)
    print('=======需要切开的位置=======',result_arr)
    print('======音频长度=====',maxlen)

    # sound_file='/home/wanggf/Github/sound_split/xunfei/sound_source/4.1.m4a'
    # result=exec_split(text,sound_file)
    # print(result)
