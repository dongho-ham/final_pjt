import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.val_loss2_min = np.inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    """
    Solver for training and testing Anomaly Transformer model
    Created by: Ham Dongho
    2025.12.05

    Args:
        config: configuration dictionary containing hyperparameters and settings
        win_size: size of the input window (25, 50, 100)
        input_c: number of input channels (20, 38)
        output_c: number of output channels (20, 38)
    
    Controls the training and testing process of the Anomaly Transformer model.
    1. Initializes data loaders for training, validation, testing, and thresholding.
    2. Builds the Anomaly Transformer model and optimizer.
    3. Implements training with early stopping and learning rate adjustment.
    4. Implements testing with anomaly score calculation, thresholding, and evaluation metrics.
    5. Saves training history and test results with version management.
    """
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(
        self.data_path, batch_size=self.batch_size, win_size=self.win_size,
        step=self.stride, mode='train', dataset=self.dataset
    )
        self.vali_loader = get_loader_segment(
            self.data_path, batch_size=self.batch_size, win_size=self.win_size,
            step=self.stride, mode='val', dataset=self.dataset
        )
        self.test_loader = get_loader_segment(
            self.data_path, batch_size=self.batch_size, win_size=self.win_size,
            step=self.stride, mode='test', dataset=self.dataset
        )
        self.thre_loader = get_loader_segment(
            self.data_path, batch_size=self.batch_size, win_size=self.win_size,
            step=self.stride, mode='thre', dataset=self.dataset
        )

        self.build_model() # 모델 및 옵티마이저 구축
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # cuda 설정
        self.criterion = nn.MSELoss() # 재구성 손실 함수

    def build_model(self):
        """
        build_model의 Docstring
        
        Builds the Anomaly Transformer model and optimizer.
        Initializes the Anomaly Transformer with specified window size, input channels, and output channels.
        Sets up the Adam optimizer for model training.
        Moves the model to GPU if available.
        """
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        """
        vali의 Docstring
        
        Validates the model on the validation dataset.
        Computes two types of validation losses based on reconstruction and association discrepancies.
        """
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):

        print("======================TRAIN MODE======================")

        # Loss history 저장용 딕셔너리
        history = {
            'train_loss': [],
            'vali_loss1': [],
            'vali_loss2': []
        }

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset) # 조기 종료 객체 생성
        train_steps = len(self.train_loader) # 훈련 스텝 수

        for epoch in range(self.num_epochs): # 에포크 반복
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader): # 배치 반복

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    # local loss 계산
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    # global loss 계산
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                series_loss = series_loss / len(prior) 
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.k * series_loss).item()) # train loss 기록
                # Anomaly Transformer의 Minimax 전략: Anomaly Association Discrepancy 극대화
                loss1 = rec_loss - self.k * series_loss # minimize
                loss2 = rec_loss + self.k * prior_loss # maximize

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.test_loader)

            history['train_loss'].append(float(train_loss))
            history['vali_loss1'].append(float(vali_loss1))
            history['vali_loss2'].append(float(vali_loss2))

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

        import pickle
        with open(os.path.join(path, 'training_history.pkl'), 'wb') as f:
            pickle.dump(history, f)
        print(f"\n✅ Training history saved to {path}/training_history.pkl")

    def test(self):
        self.model.load_state_dict(
            # checkpoint 불러오기
            torch.load( 
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth'))) # 최적화된 가중치 불러오기
        self.model.eval()
        temperature = 50 # 온도 매개변수 설정: anomaly score의 민감도 조절

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the test set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input) # 예측값(output), attention 정보(series, prior)
            # 재구성 손실 계산
            loss = torch.mean(criterion(input, output), dim=-1)
            # Association discrepancy 계산(핵심)
            series_loss = 0.0
            prior_loss = 0.0
            # layer 별로 attention pattern 분석
            for u in range(len(prior)):
                # 정규화된 prior와 series 간의 KL divergence 계산
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            # prior 확률 분포 정규화(모든 값의 합 = 1). 모든 attention weight는 확률 분포여야 함.
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                # u가 0이 아닌 경우 누적
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            # Metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        # Train 데이터만으로 threshold 계산 (percentile 방식, outlier robust)
        # 배터리 데이터는 열화가 후기에 나타나서 combined 방식이 데이터 누수를 일으킬 수 있음
        thresh = np.percentile(train_energy, 100 - self.anormly_ratio)
        print(f"\nThreshold ({100 - self.anormly_ratio}th percentile): {thresh:.6f}")

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            # attention weight 기반의 이상 점수 계산. 이상 time step일수록 높은 점수를 가짐.
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)

        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        anomaly_state = False
        # 후처리: 연속된 이상 구간 보정. 실제 이상은 연속 구간이므로 예측도 연속 구간으로 만듦.
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))

        # 결과 저장
        results = {
            'test_energy': test_energy,
            'pred': pred,
            'gt': gt,
            'threshold': thresh,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f_score': f_score
        }
        
        # Cycle별 anomaly score 계산
        try:
            if hasattr(self.thre_loader.dataset, 'cycle_idx') and self.thre_loader.dataset.cycle_idx is not None:
                # Window별 평균 계산 (stride=1 고려)
                stride = self.stride
                win_size = self.win_size
                n_windows = (len(self.thre_loader.dataset.cycle_idx) - win_size) // stride + 1
                
                # test_energy를 window 단위로 재구성
                test_energy_per_window = []
                for w in range(n_windows):
                    window_start = w * stride  # ← 수정: stride 고려
                    window_end = window_start + win_size
                    # test_energy는 (batch * win_size)로 flatten되어 있음
                    # 각 window의 energy는 [w*win_size : (w+1)*win_size] 범위
                    start_idx = w * win_size
                    end_idx = start_idx + win_size
                    if end_idx <= len(test_energy):
                        test_energy_per_window.append(test_energy[start_idx:end_idx].mean())
                
                test_energy_per_window = np.array(test_energy_per_window)
                
                # 각 window의 대표 cycle (첫 timestep 기준)
                cycle_per_window = []
                for w in range(len(test_energy_per_window)):
                    idx = w * stride  # stride=1
                    if idx < len(self.thre_loader.dataset.cycle_idx):
                        cycle_per_window.append(self.thre_loader.dataset.cycle_idx[idx])
                cycle_per_window = np.array(cycle_per_window)
                
                # Cycle별 평균
                cycle_scores = {}
                for cycle in np.unique(cycle_per_window):
                    mask = cycle_per_window == cycle
                    if mask.sum() > 0:
                        cycle_scores[int(cycle)] = float(test_energy_per_window[mask].mean())
                results['cycle_scores'] = cycle_scores
                
                # Top 10
                top_cycles = sorted(cycle_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                results['top_10_cycles'] = top_cycles
                print("\n--- Top 10 Anomalous Cycles ---")
                for cycle, score in top_cycles:
                    marker = "⭐" if cycle in [42, 598] else ""
                    print(f"Cycle {cycle}: {score:.6f} {marker}")
        except Exception as e:
            print(f"Warning: Could not calculate cycle scores: {e}")
        
        # 저장 (버전 관리)
        import pickle
        from datetime import datetime
        
        save_path = os.path.join(self.model_save_path, 'test_results.pkl')
        
        # 기존 파일 백업
        if os.path.exists(save_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = save_path.replace('.pkl', f'_backup_{timestamp}.pkl')
            os.rename(save_path, backup_path)
            print(f"📦 Previous results backed up to: {backup_path}")
        
        # 저장
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"✅ Test results saved to {save_path}")

        return accuracy, precision, recall, f_score
