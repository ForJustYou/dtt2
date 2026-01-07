from data.data_provider.data_factory import data_provider
from src.exp.exp_basic import Exp_Basic
from src.utils.tools import EarlyStopping, adjust_learning_rate
from src.utils.metrics import MAE, MSE, RMSE, MAPE, MSPE, SMAPE
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from datetime import datetime
import src.utils.log as log


warnings.filterwarnings('ignore')

class exp_MTS_forecasting(Exp_Basic):
    def __init__(self, args):
        super(exp_MTS_forecasting, self).__init__(args)
        data_name = getattr(self.args, "data_path", None)
        if not data_name:
            data_path = getattr(self.args, "data_path", "")
            data_name = os.path.splitext(os.path.basename(str(data_path)))[0] or "data"
        log.init_csv_logger(
            self.args.model,
            data_name,
            params=vars(self.args),
            data_subdir=data_name,
        )
        self.logger = log.Logger

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

    def _select_criterion(self, loss_type):
        if loss_type == 'MAE':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    @staticmethod
    def _calc_metrics(pred, true):
        return {
            "mae": MAE(pred, true),
            "mse": MSE(pred, true),
            "rmse": RMSE(pred, true),
            "mape": MAPE(pred, true),
            "mspe": MSPE(pred, true),
            "smape": SMAPE(pred, true),
        }


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, cycle_index) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                cycle_index = cycle_index.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,cycle_index)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,cycle_index)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,cycle_index)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,cycle_index)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())
        total_loss = np.average(total_loss)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        target_pred = preds[:, -1, -1]
        target_true = trues[:, -1, -1]
        metrics = self._calc_metrics(target_pred, target_true)
        self.model.train()
        return total_loss, metrics


    def train(self, setting):

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=1e-16)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, cycle_index) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                cycle_index = cycle_index.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cycle_index)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cycle_index)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cycle_index)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cycle_index)

                    f_dim = -1 if self.args.features == 'MS' else 0 # Only use the target variable
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    if (i + 1) % 1000 == 0:
                         #记录日志
                        self.logger.log({
                            'iter': i + 1,
                            'train_loss': '{:.4f}'.format(loss.item()),
                            'speed': '{:.4f}'.format(speed),
                        })   

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_metrics = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_metrics = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print("Vali: MAE {0:.7f} MSE {1:.7f} RMSE {2:.7f} MAPE {3:.7f} MSPE {4:.7f} SMAPE {5:.7f}".format(
                vali_metrics['mae'], vali_metrics['mse'], vali_metrics['rmse'],
                vali_metrics['mape'], vali_metrics['mspe'], vali_metrics['smape']))
            print("Test: MAE {0:.7f} MSE {1:.7f} RMSE {2:.7f} MAPE {3:.7f} MSPE {4:.7f} SMAPE {5:.7f}".format(
                test_metrics['mae'], test_metrics['mse'], test_metrics['rmse'],
                test_metrics['mape'], test_metrics['mspe'], test_metrics['smape']))
            # log
            self.logger.log({
                'epoch': epoch + 1,
                'train_loss': '{:.4f}'.format(train_loss),
                'vali_loss': '{:.4f}'.format(vali_loss),
                'test_loss': '{:.4f}'.format(test_loss),
                'vali_mae': '{:.4f}'.format(vali_metrics['mae']),
                'vali_mse': '{:.4f}'.format(vali_metrics['mse']),
                'vali_rmse': '{:.4f}'.format(vali_metrics['rmse']),
                'vali_mape': '{:.4f}'.format(vali_metrics['mape']),
                'vali_mspe': '{:.4f}'.format(vali_metrics['mspe']),
                'vali_smape': '{:.4f}'.format(vali_metrics['smape']),
                'test_mae': '{:.4f}'.format(test_metrics['mae']),
                'test_mse': '{:.4f}'.format(test_metrics['mse']),
                'test_rmse': '{:.4f}'.format(test_metrics['rmse']),
                'test_mape': '{:.4f}'.format(test_metrics['mape']),
                'test_mspe': '{:.4f}'.format(test_metrics['mspe']),
                'test_smape': '{:.4f}'.format(test_metrics['smape']),
                'cost_time': '{:.4f}'.format(time.time() - epoch_time),
                'epoch_time': datetime.now().isoformat(timespec="seconds"),
            })

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print(f"Early stopping on validation score {early_stopping.best_score}")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, cycle_index) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                cycle_index = cycle_index.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cycle_index)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cycle_index)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cycle_index)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cycle_index)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        target_pred = preds[:, -1, -1]
        target_true = trues[:, -1, -1]
        metrics = self._calc_metrics(target_pred, target_true)
        print("target mae:{}".format(metrics['mae']))
        print("target mse:{}".format(metrics['mse']))
        print("target rmse:{}".format(metrics['rmse']))
        print("target mape:{}".format(metrics['mape']))
        print("target mspe:{}".format(metrics['mspe']))
        print("target smape:{}".format(metrics['smape']))

        self.logger.log({
            'epoch': 'test_metric',
            'test_mae': 'mae',
            'test_mse': 'mse',
            'test_rmse': 'rmse',
            'test_mape': 'mape',
            'test_mspe': 'mspe',
            'test_smape': 'smape',
        })
        self.logger.log({
            'epoch': 'test',
            'test_mae': '{:.4f}'.format(metrics['mae']),
            'test_mse': '{:.4f}'.format(metrics['mse']),
            'test_rmse': '{:.4f}'.format(metrics['rmse']),
            'test_mape': '{:.4f}'.format(metrics['mape']),
            'test_mspe': '{:.4f}'.format(metrics['mspe']),
            'test_smape': '{:.4f}'.format(metrics['smape']),
        })

        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}, smape:{}'.format(
            metrics['mae'], metrics['mse'], metrics['rmse'],
            metrics['mape'], metrics['mspe'], metrics['smape']))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
