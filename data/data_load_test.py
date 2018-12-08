import wave
import numpy as np
import os
#import matplotlib.pyplot as plt

def read_wav_data(filename):
    '''
    读取一个wav文件，返回声音信号的时域谱矩阵和播放时间
    '''
    wav = wave.open(filename, "rb")  # 打开一个wav格式的声音文件流
    num_frame = wav.getnframes()  # 获取帧数
    num_channel = wav.getnchannels()  # 获取声道数
    framerate = wav.getframerate()  # 获取帧速率
    num_sample_width = wav.getsampwidth()  # 获取实例的比特宽度，即每一帧的字节数
    str_data = wav.readframes(num_frame)  # 读取全部的帧
    wav.close()  # 关闭流
    wave_data = np.fromstring(str_data, dtype=np.short)  # 将声音文件数据转换为数组矩阵形式
    wave_data.shape = -1, num_channel  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    wave_data = wave_data.T  # 将矩阵转置
    # wave_data = wave_data
    return wave_data, framerate


# def wav_show(wave_data, fs):  # 显示出来声音波形
    #time = np.arange(0, len(wave_data)) * (1.0/fs)  # 计算声音的播放时间，单位为秒
    # 画声音波形
    #plt.plot(time, wave_data)
    #plt.show()


if __name__ == '__main__':
    '''
    wave_data, fs = read_wav_data("VCC2TM1/10001.wav")
    print(wave_data[0])
    print(type(wave_data[0]))
    print(wave_data[0].shape)
    #wav_show(wave_data[0], fs)
    # wav_show(wave_data[1], fs)  # 如果是双声道则保留这一行，否则删掉这一行
    # process all data
    '''
    path = "VCC2SF1/"         #文件夹路径
    files = os.listdir(path)
    length = len(files)
    print(length)
    train_len = int(0.8*length)      # 计算train valid test 数据集条目数量
    valid_len = int(0.1*length)
    test_len = length - train_len - valid_len
    temp = []
    smallest_number = 0              # 记录最小值
    for i in range(train_len):
        wave_data, fs = read_wav_data(path+files[i])
        temp.append(wave_data[0])
        smallest_number = np.min([smallest_number, np.min(wave_data[0])])
    train_data = np.array(temp)      # list 转 array
    temp = []
    
    for i in range(valid_len):
        wave_data, fs = read_wav_data(path+files[i + train_len])
        temp.append(wave_data[0])
        smallest_number = np.min([smallest_number, np.min(wave_data[0])])
    valid_data = np.array(temp)
    temp = []

    for i in range(test_len):
        wave_data, fs = read_wav_data(path+files[i + train_len + valid_len])
        temp.append(wave_data[0])
        smallest_number = np.min([smallest_number, np.min(wave_data[0])])
    test_data = np.array(temp)

    train_data = train_data - smallest_number
    valid_data = valid_data - smallest_number
    test_data = test_data - smallest_number

    np.save(path+"train_data.npy",train_data)
    np.save(path+"valid_data.npy",valid_data)
    np.save(path+"test_data.npy",test_data)


