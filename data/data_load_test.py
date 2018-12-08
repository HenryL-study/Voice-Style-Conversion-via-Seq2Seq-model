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
    path1 = "VCC2SF1/"            #文件夹路径
    path2 = "VCC2SM1/"
    save_path = "processed_data"  #save 路径
    files_1 = os.listdir(path1)
    files_2 = os.listdir(path2)
    length = len(files_1)
    #print(length)
    train_len = int(0.8*length)      # 计算train valid test 数据集条目数量
    valid_len = int(0.1*length)
    test_len = length - train_len - valid_len
    temp_1 = []
    temp_2 = []
    smallest_number = 0              # 记录最小值
    largest_number = 0               # 记录最大值
    for i in range(train_len):       # 记录 train set
        wave_data_1, fs = read_wav_data(path1+files_1[i])
        wave_data_2, fs = read_wav_data(path2+files_2[i])
        temp_1.append(wave_data_1[0])
        temp_2.append(wave_data_2[0])
        smallest_number = np.min([smallest_number, np.min(wave_data_1[0]), np.min(wave_data_2[0])])
        largest_number = np.max([largest_number, np.max(wave_data_1[0]), np.max(wave_data_2[0])])
    train_data_1 = np.array(temp_1)      # list 转 array
    train_data_2 = np.array(temp_2)
    temp_1 = []
    temp_2 = []
    
    for i in range(valid_len):        # 记录 valid set
        wave_data_1, fs = read_wav_data(path1+files_1[i + train_len])
        wave_data_2, fs = read_wav_data(path2+files_2[i + train_len])
        temp_1.append(wave_data_1[0])
        temp_2.append(wave_data_2[0])
        smallest_number = np.min([smallest_number, np.min(wave_data_1[0]), np.min(wave_data_2[0])])
        largest_number = np.max([largest_number, np.max(wave_data_1[0]), np.max(wave_data_2[0])])
    valid_data_1 = np.array(temp_1)
    valid_data_2 = np.array(temp_2)
    temp_1 = []
    temp_2 = []

    for i in range(test_len):         # 记录 test set
        wave_data_1, fs = read_wav_data(path1+files_1[i + train_len + valid_len])
        wave_data_2, fs = read_wav_data(path2+files_2[i + train_len + valid_len])
        temp_1.append(wave_data_1[0])
        temp_2.append(wave_data_2[0])
        smallest_number = np.min([smallest_number, np.min(wave_data_1[0]), np.min(wave_data_2[0])])
        largest_number = np.max([largest_number, np.max(wave_data_1[0]), np.max(wave_data_2[0])])
    test_data_1 = np.array(temp_1)
    test_data_2 = np.array(temp_2)

    class_number = largest_number - smallest_number + 1
    print("class number : ")
    print(class_number)
    train_data_1 = train_data_1 - smallest_number
    train_data_2 = train_data_2 - smallest_number
    valid_data_1 = valid_data_1 - smallest_number
    valid_data_2 = valid_data_2 - smallest_number
    test_data_1 = test_data_1 - smallest_number
    test_data_2 = test_data_2 - smallest_number

    np.save("train_data_1.npy",train_data_1)
    np.save("train_data_2.npy",train_data_2)
    np.save("valid_data_1.npy",valid_data_1)
    np.save("valid_data_2.npy",valid_data_2)
    np.save("test_data_1.npy",test_data_1)
    np.save("test_data_2.npy",test_data_2)


