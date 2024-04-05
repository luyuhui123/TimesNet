from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot  as plt
import pandas as pd
import more_itertools as mit
warnings.filterwarnings('ignore')


class ErrorWindow():
    def __init__(self, e_s):
        self.e_s = e_s

        self.i_anom = np.array([])  # 窗口中异常点的下标
        self.E_seq = np.array([])  # 连续异常点的首尾坐标
        self.non_anom_max = float('-inf')  # 窗口中非异常点的最大值

        self.sd_lim = 5.0  # z 范围的最大值
        self.sd_threshold = self.sd_lim  # 最佳的 z

        self.mean_e_s = np.mean(self.e_s)
        self.sd_e_s = np.std(self.e_s)
        self.epsilon = self.mean_e_s + self.sd_lim * self.sd_e_s  # 阈值

        self.p = 0.7 # 连续误差序列最大值的波动率

    def find_epsilon(self):
        '''寻找最佳的 z 值'''
        e_s = self.e_s
        max_score = float('-inf')

        # 遍历寻找最佳 z
        for z in np.arange(0.5, self.sd_lim, 0.1):
            # 计算阈值
            print("均值是：",self.mean_e_s)
            print("方差是：",self.sd_e_s)
            print("Z是：",z)
            
            epsilon = self.mean_e_s + (self.sd_e_s * z)
            print("阈值是：",epsilon)
            # 除去异常点后的序列
            pruned_e_s = e_s[e_s < epsilon]

            # 大于阈值的点的下标
            i_anom = np.argwhere(e_s >= epsilon).reshape(-1)
            # # 设置缓冲区
            # buffer = np.arange(1, 2)
            # # 将每个异常点周围的值（缓冲区）添加到异常序列中
            # i_anom = np.concatenate(
            #     (i_anom, np.array([i + buffer for i in i_anom]).flatten(),
            #      np.array([i - buffer for i in i_anom]).flatten()))
            # i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
            i_anom = np.sort(np.unique(i_anom))

            # 如果存在异常点
            if len(i_anom) > 0:
                # 得到连续异常点的下标
                groups = [list(group) for group in mit.consecutive_groups(i_anom)]
                # 得到连续异常点的首尾坐标
                E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]
                # groups: [[1, 2, 3, 4], [6, 7], [9, 10]]
                # E_seq: [(1, 4), (6, 7), (9, 10)]

                # 计算去除异常点前后均值，方差的变化
                mean_delta = (self.mean_e_s - np.mean(pruned_e_s)) / self.mean_e_s
                sd_delta = (self.sd_e_s - np.std(pruned_e_s)) / self.sd_e_s
                # 计算得分
                score = (mean_delta + sd_delta) / (len(E_seq)**2 + len(i_anom))
                # score = (mean_delta + sd_delta)
                if score >= max_score and len(E_seq) < 6 and len(i_anom) < (len(e_s) / 2):
                    max_score = score
                    self.sd_threshold = z
                    self.epsilon = self.mean_e_s + z * self.sd_e_s

    def compare_to_epsilon(self):
        # '''获取当前窗口小于阈值的最大值'''
        e_s = self.e_s
        epsilon = self.epsilon
        print("计算认为的最佳阈值为：",epsilon)
        # 找到异常点的下标
        i_anom = np.argwhere(e_s >= epsilon).reshape(-1)
        print("剪枝前它认为的异常个数：",len(i_anom))
        if len(i_anom) == 0:
            return

        # buffer = np.arange(1, 2)
        # i_anom = np.concatenate((i_anom, np.array([i + buffer for i in i_anom]).flatten(),
        #                          np.array([i - buffer for i in i_anom]).flatten()))
        # i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
        i_anom = np.sort(np.unique(i_anom))

        # 获取当前窗口小于阈值的最大值
        window_indices = np.setdiff1d(np.arange(0, len(e_s)), i_anom)
        non_anom_max = np.max(np.take(e_s, window_indices))

        groups = [list(group) for group in mit.consecutive_groups(i_anom)]
        E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

        self.i_anom = i_anom
        self.E_seq = E_seq
        self.non_anom_max = non_anom_max

    def prune_anoms(self):
        # '''修剪异常点'''
        E_seq = self.E_seq
        e_s = self.e_s
        non_anom_max = self.non_anom_max

        if len(E_seq) == 0:
            return

        # 得到每个连续异常序列中的最大值
        E_seq_max = np.array([max(e_s[e[0]:e[1] + 1]) for e in E_seq])
        E_seq_max_sorted = np.sort(E_seq_max)[::-1]
        # 每个连续异常序列中的最大值 + 非异常点的最大值
        E_seq_max_sorted = np.append(E_seq_max_sorted, [non_anom_max])

        i_to_remove = np.array([])
        for i in range(0, len(E_seq_max_sorted) - 1):
            # 在异常序列中最大误差之间的最小百分比下降
            if (E_seq_max_sorted[i] - E_seq_max_sorted[i+1]) \
                    / E_seq_max_sorted[i] < self.p:
                i_to_remove = np.append(
                    i_to_remove,
                    np.argwhere(E_seq_max == E_seq_max_sorted[i])).astype(int)
            else:
                i_to_remove = np.array([])
        i_to_remove.sort()

        if len(i_to_remove) > 0:
            E_seq = np.delete(E_seq, i_to_remove, axis=0)

        if len(E_seq) == 0:
            self.i_anom = np.array([])
            return

        indices_to_keep = np.concatenate(
            [range(e_seq[0], e_seq[1] + 1) for e_seq in E_seq])
        mask = np.isin(self.i_anom, indices_to_keep)
        self.i_anom = self.i_anom[mask]

class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)
    def detect_anomaly(self,testEnergy,actual, pred, window_size: int):
        # '''异常检测

        # Args:
        # -------
        #     actual(array): 真实值
        #     pred(array): 预测值
        #     window_size(int): 窗口大小

        # Returns:
        # -------
        #     anomaly_list(List): 异常点序号
        # '''
        # e = abs(actual - pred)
        # e = testEnergy
        # smoothing_window = int(window_size * 0.05)
        # e_s = pd.DataFrame(e).ewm(span=smoothing_window).mean().values.flatten()
        e_s = testEnergy

        anomaly_list = np.array([])
        for i in range(len(e_s) // window_size):
            cur = np.array(e_s[i * window_size:(i + 1) * window_size])

            window = ErrorWindow(cur)
            window.find_epsilon()
            window.compare_to_epsilon()

            if len(window.i_anom) == 0:
                continue

            window.prune_anoms()

            if len(window.i_anom) == 0:
                continue

            window.i_anom = np.sort(np.unique(window.i_anom))
            anomaly_list = np.append(anomaly_list,
                                    window.i_anom + i * window_size).astype('int')

        return anomaly_list.tolist()

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        thre_data, thre_loader = self._get_data(flag='thre')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x, None, None, None)
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                # 防止梯度在score张量上反向传播并且把score转到cpu上，再转为Numpy
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []

        # 输入结果与原数据拟合对比
        outData = []
        inputData = []
        for i, (batch_x, batch_y) in enumerate(thre_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x, None, None, None)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)
             #构建输入值数组
            inputs = batch_x.detach().cpu().numpy()
            inputs = np.array([array[:, 0] for array in inputs])
            inputData.append(inputs)
           
            #构建输出值数组
            outputs = outputs.detach().cpu().numpy()
            outputs = np.array([array[:, 0] for array in outputs])
            outData.append(outputs)

            print("input形状",inputs.shape)
            print("output形状",outputs.shape)
      
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(test_energy, 100 - self.args.anomaly_ratio)
        print("原来的方法认为的阈值:", threshold)
        
        #把输入输出数据改成合理的格式
        outData = np.concatenate(outData, axis=0).reshape(-1)
        outData = np.array(outData)

        inputData = np.concatenate(inputData, axis=0).reshape(-1)
        inputData = np.array(inputData)
            
        print("inputData形状",inputData.shape)
        print("outputData形状",outData.shape)

        #计算动态阈值  根据最好的结果返回异常点
        # anomaly_list = self.detect_anomaly(test_energy,inputData,outData,100)
        anomaly_list = self.detect_anomaly(test_energy,inputData,outData,100)
        print("anomaly_list:",anomaly_list)

        x = np.linspace(1, len(test_energy),len(test_energy))
        # y_thresh = [threshold]*len(test_energy)

        # print("pred中有多少个大于阈值: ", np.count_nonzero(test_energy > threshold))
       
        # 某些点的异常分数太高 导致画图不清晰 在此把他们变小
        # test_energy = [x if x < 0.2 else 0.2 for x in test_energy]
        # for i in range(len(test_energy)):
        #     if test_energy[i] > 0.2:
        #           test_energy[i] = 0.2

        plt.figure(figsize=(60, 30)) 
        # 创建子画布
        ax1=plt.subplot(2,2,1)
        ax1.scatter(x, test_energy, label='predict')
        # ax1.scatter(x, y_thresh, label='threshould',s=5)
      

        # 设置x，y轴的标签
        ax1.set_xlabel('time')
        ax1.set_ylabel('AnomalyScore')

        # 设置标签
        ax1.set_title('1 predicted vs threshould')
        # 设置标签的位置
        ax1.legend()

        # (3) evaluation on the test set
        # pred = (test_energy > threshold).astype(int)
        pred = [1 if i in anomaly_list else 0 for i in range(len(test_energy))]
        pred = np.array(pred)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        # gt = (test_labels != 2).astype(int)
        gt = test_labels.astype(int)

        print("pred中有多少个1: ", np.count_nonzero(pred == 1))
        print("gt中有多少个1:     ", np.count_nonzero(gt == 1))
        # indices = [i for i, x in enumerate(gt) if x == 1]

        # print("gt中为1的下标",indices)  # 输出值为1的下标列表
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        # (4) detection adjustment
        # gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)

        
        # 绘制输出值和输入值对比折线图
        ax2=plt.subplot(2,2,2)
       
        # ax2.plot(x[0:100], inputData[0:100], label='inputData',c='blue')
        # ax2.plot(x[0:100], outData[0:100], label='outData',c='orange')
        # ax2.scatter(np.where(pred[0:100] == 1), outData[0:100][pred[0:100]== 1], c='red', label='predAnomaly', marker='^', s=15, zorder=3)

        ax2.plot(x, inputData, label='inputData',c='blue')
        ax2.plot(x, outData, label='outData',c='orange')
        ax2.scatter(np.where(gt == 1), outData[gt== 1], c='green', label='groudtruth', marker='*', s=10, zorder=2)
        ax2.scatter(np.where(pred == 1), outData[pred== 1], c='red', label='predAnomaly', marker='^', s=5, zorder=3)

        # 设置x，y轴的标签

        # 设置x，y轴的标签
        ax2.set_xlabel('time')
        ax2.set_ylabel('I/O data')

        # 设置标签
        ax2.set_title('2 inputData vs outData')
        # 设置标签的位置
        ax2.legend()


        # 绘图观察预测结果
        # 预测值上根据groudtruth打标记
        ax3=plt.subplot(2,2,3)
        ax3.scatter(x, test_energy, label='predict')
        # ax1.plot(x, y_thresh, label='threshould')
        # plt.scatter(x, y_thresh, label='thresh')
        # plt.plot(gt, marker = 'o', ms = 20, mec = 'r', mfc = 'b')
        ax3.scatter(np.where(gt == 1), test_energy[gt== 1], c='red', label='groudtruth', marker='^', s=15, zorder=2)

        # 设置x，y轴的标签
        ax3.set_xlabel('time')
        ax3.set_ylabel('AnomalyScore')

     

        # 设置标签
        ax3.set_title('3 predictScore and groudth')
        # 设置标签的位置
        ax3.legend()

        
        ax4=plt.subplot(2,2,4)
        ax4.scatter(x, test_energy, label='predict')
        ax4.scatter(np.where(pred == 1), test_energy[pred== 1], c='orange', label='predAnomaly', marker='^', s=15, zorder=2)
        # print("异常个数:",pred.count(1))
  

        # 设置x，y轴的标签
        ax4.set_xlabel('时间点')
        ax4.set_ylabel('异常分数值')
     

        # 设置标签
        ax4.set_title('4 predictScore and predAnomaly')
        # 设置标签的位置
        ax4.legend()
        
        # plt.savefig('./pic/AIRT_2006_anomalyResult3_changeValid.png')
        # plt.savefig('./pic/AIRT_2006_anomalyResult.png')
        # plt.savefig('./pic/AIRT_2006_anomalyResult4_withoutAdjustMent.png')
        # plt.savefig('./pic/SMD_RESULT2.png')
        # plt.savefig('./pic/AIRT_2006_testAnomalyResult_withoutTrainTranform.png')
        plt.savefig('./pic/PRES_2006_anomalyResult_DynamicThre.png')
        # plt.savefig('./pic/SST_2006_anomalyResult.png')



        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        f.write('\n')
        f.close()
        return
 