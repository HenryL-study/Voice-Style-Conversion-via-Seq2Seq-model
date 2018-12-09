import numpy as np
import os

if __name__ == '__main__':
    path1 = "parameters/VCC2SF1/"   #文件夹路径
    path2 = "parameters/VCC2SM1/"
    #save_path = "processed_data"  #save 路径
    files_1 = os.listdir(path1)
    files_2 = os.listdir(path2)
    length = 81
    #print(length)
    train_len = int(0.8*length)      # 计算train valid test 数据集条目数量
    valid_len = int(0.1*length)
    test_len = length - train_len - valid_len
    temp_1 = []
    temp_2 = []

    for i in range(train_len):       # 记录 train set
        wave_data_1 = np.load(path1 + "coded_sp/" + str(i)+ '.npy')
        wave_data_2 = np.load(path2 + "coded_sp/" + str(i)+ '.npy')
        temp_1.append(wave_data_1)
        temp_2.append(wave_data_2)
    train_data_1 = np.array(temp_1)      # list 转 array
    train_data_2 = np.array(temp_2)
    temp_1 = []
    temp_2 = []

    for i in range(valid_len):        # 记录 valid set
        wave_data_1 = np.load(path1 + "coded_sp/" + str(i + train_len)+'.npy')
        wave_data_2 = np.load(path2 + "coded_sp/" + str(i + train_len)+'.npy')
        temp_1.append(wave_data_1)
        temp_2.append(wave_data_2)
    valid_data_1 = np.array(temp_1)
    valid_data_2 = np.array(temp_2)
    temp_1 = []
    temp_2 = []

    temp_ap1 = []
    temp_ap2 = []
    temp_f01 = []
    temp_f02 = []
    for i in range(test_len):         # 记录 test set
        wave_data_1 = np.load(path1 + "coded_sp/" + str(i + train_len + valid_len)+'.npy')
        wave_data_2 = np.load(path2 + "coded_sp/" + str(i + train_len + valid_len)+'.npy')
        # test ap and f0
        ap1 = np.load(path1 + "ap/" + str(i + train_len + valid_len)+'.npy')
        ap2 = np.load(path2 + "ap/" + str(i + train_len + valid_len)+'.npy')
        f01 = np.load(path1 + "f0/" + str(i + train_len + valid_len)+'.npy')
        f02 = np.load(path2 + "f0/" + str(i + train_len + valid_len)+'.npy')
        temp_1.append(wave_data_1)
        temp_2.append(wave_data_2)
        temp_ap1.append(ap1)
        temp_ap2.append(ap2)
        temp_f01.append(f01)
        temp_f02.append(f02)
    test_data_1 = np.array(temp_1)
    test_data_2 = np.array(temp_2)
    test_ap_1 = np.array(temp_ap1)
    test_ap_2 = np.array(temp_ap2)
    test_f0_1 = np.array(temp_f01)
    test_f0_2 = np.array(temp_f02)


    np.save("generate/train_data_1.npy",train_data_1)
    np.save("generate/train_data_2.npy",train_data_2)
    np.save("generate/valid_data_1.npy",valid_data_1)
    np.save("generate/valid_data_2.npy",valid_data_2)
    np.save("generate/test_data_1.npy",test_data_1)
    np.save("generate/test_data_2.npy",test_data_2)
    # save ap and f0 for test dataset
    np.save("generate/test_ap_1.npy",test_ap_1)
    np.save("generate/test_ap_2.npy",test_ap_2)
    np.save("generate/test_f0_1.npy",test_f0_1)
    np.save("generate/test_f0_2.npy",test_f0_2)





    