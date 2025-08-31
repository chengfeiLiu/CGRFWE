import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
import os
import random
import numpy as np
from types import SimpleNamespace
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from dataclasses import dataclass, field
from sklearn.metrics import accuracy_score
from CNNAttention import MultiScaleAttentionNet
from CNNAttentionRes import MultiScaleAttentionNeRes
from Shapelet import ShapeBottleneckModel, DistThresholdSBM
# from Shapelet import ShapeBottleneckModel
from InterpGN import InterpGN, FullyConvNetwork, Transformer, TimesNet, PatchTST, ResNet
from GRU import GRUModel

from tools import EarlyStopping, convert_to_hms, gini_coefficient
from shapelet_util import ClassificationResult
from dataloader import ExcelSeqLoader
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

def get_data(args, flag, batch_size, shuffle_flag=True,scaler=None):
    drop_last = False
     # 如果是训练阶段，创建 scaler 并 fit
    if flag == "train":
        scaler = StandardScaler()

    data_set = ExcelSeqLoader(
        folder_path=args.folder_path,  
        data_columns=args.data_columns,
        label_column=args.label_column,
        seq_len=args.seq_len,
        step=args.step,
        flag=flag,
        scaler=scaler
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
    )

    return data_set, data_loader,scaler
def collate_fn(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels = zip(*data)
    labels = [torch.tensor(label, dtype=torch.long) for label in labels]
    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)

    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = torch.from_numpy(features[i][:end, :])

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, targets, padding_masks

def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
def compute_beta(epoch, max_epoch, schedule='cosine'):
    if schedule == 'cosine':
        beta = 1/2 * (1 + np.cos(np.pi*epoch/max_epoch))
    elif schedule == 'linear':
        beta = 1 - epoch/max_epoch
    else:
        beta = 1
    return beta


def compute_shapelet_score(shapelet_distances, cls_weights, y_pred, y_true):
    score = shapelet_distances @ nn.functional.relu(cls_weights.T) / shapelet_distances.shape[-1]
    score_correct = score[y_pred == y_true]
    class_correct = y_true[y_pred == y_true]
    score_class = score_correct.gather(-1, class_correct.unsqueeze(1))
    return score_class.mean().item()


def get_dnn_model(configs):
    dnn_dict = {
        'FCN': FullyConvNetwork,
        'Transformer': Transformer,
        'TimesNet': TimesNet,
        'PatchTST': PatchTST,
        'ResNet': ResNet,
        'MultiScaleAttentionNet':MultiScaleAttentionNet,
        'MultiScaleAttentionNeRes':MultiScaleAttentionNeRes,
        'GRUModel':GRUModel
    }
    return dnn_dict[configs.dnn_type](configs)


class Experiment(object):
    model_dict = {
        'InterpGN': InterpGN,
        'SBM': ShapeBottleneckModel,
        'LTS': DistThresholdSBM,
        'DNN': get_dnn_model
    }
    def __init__(self, args):
        self.train_data, self.train_loader, scaler  = get_data(args, flag='train', batch_size=64)
        self.test_data, self.test_loader,_ = get_data(args, flag='val', batch_size=64,scaler=scaler)
        self.val_data, self.val_loader,_= get_data(args, flag='test', batch_size=64,scaler=scaler)


        args.seq_len =  args.seq_len
        args.pred_len = 0
        args.enc_in = args.enc_in
        args.num_class = 2
        self.epoch_stop = 0

        # Build Model
        self.args = args
        self.device = torch.device('cuda:0')
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = self._build_model().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=self.args.train_epochs)
        self.checkpoint_dir = "./checkpoints/{}/{}/dnn-{}_seed-{}_k-{}_div-{}_reg-{}_eps-{}_beta-{}_dfunc-{}_cls-{}".format(
            self.args.model,
            self.args.dataset,
            self.args.dnn_type,
            self.args.seed,
            self.args.num_shapelet,
            self.args.lambda_div,
            self.args.lambda_reg,
            self.args.epsilon,
            self.args.beta_schedule,
            self.args.distance_func,
            self.args.sbm_cls
        )
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def print_args(self):
        for arg in vars(self.args):
            print(f"{arg}: {getattr(self.args, arg)}")

    def _build_model(self):
        shapelet_lengths = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
        num_shapelet = [self.args.num_shapelet] * len(shapelet_lengths)

        model = self.model_dict[self.args.model](
            configs=self.args,
            num_shapelet = num_shapelet,
            shapelet_len = shapelet_lengths,
        )
        # model = self.model_dict[self.args.model](
        #     configs=self.args
        # )

        if self.args.multi_gpu:
            model = nn.DataParallel(model)
        return model
        
    def train(self):
        torch.set_float32_matmul_precision('medium')
        checkpoint_dir = self.checkpoint_dir

        time_start = time.time()

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=0)
        train_step = 0
        metrics_log = []  
        for epoch in range(self.args.train_epochs):
            self.model.train()

            train_loss = []
            for i, (batch_x, label, padding_mask) in enumerate(self.train_loader):
                train_step += 1
                batch_x = batch_x.float().to(self.device)
                label = label.long().squeeze(-1).to(self.device)
                padding_mask = padding_mask.float().to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.args.amp):
                    if self.args.model == 'DNN':
                        logits = self.model(batch_x, padding_mask, None, None)
                        loss = nn.functional.cross_entropy(logits, label)
                    else:
                        logits, model_info = self.model(batch_x, padding_mask, None, None)
                        loss = nn.functional.cross_entropy(logits, label) + model_info.loss.mean()
                    if self.args.model in ['InterpGN']:
                        beta = compute_beta(epoch, self.args.train_epochs, self.args.beta_schedule)
                        loss += beta * nn.functional.cross_entropy(model_info.shapelet_preds, label)

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.backward()

                if train_step % self.args.gradient_accumulation_steps == 0:
                    if self.args.gradient_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.gradient_clip)
                    self.optimizer.step()
                    if self.args.pos_weight:
                        self.model.step()
                    self.optimizer.zero_grad()
                train_loss.append(loss.item())

            train_loss = np.mean(train_loss)
            vali_loss, val_accuracy = self.validation()
            time_now = time.time()
            time_remain = (time_now - time_start) * (self.args.train_epochs - epoch) / (epoch + 1)
            metrics_log.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": vali_loss,
                "val_accuracy": val_accuracy,
                "time_elapsed_sec": time_now - time_start,
                "time_remaining_sec": time_remain
            })
            if (epoch + 1) % self.args.log_interval == 0:
                print(f"Epoch {epoch}/{self.args.train_epochs} | Train Loss {train_loss:.4f} | Val Accuracy {val_accuracy:.4f} | Time Remain {convert_to_hms(time_remain)}")
            if self.args.lr_decay:
                self.scheduler.step()

            if epoch >= self.args.min_epochs:
                early_stopping(-val_accuracy, self.model, checkpoint_dir)
                print(f"Epoch {epoch}/{self.args.train_epochs} | Train Loss {train_loss:.4f} | Val Accuracy {val_accuracy:.4f} | Time Remain {convert_to_hms(time_remain)}")
            if early_stopping.early_stop:
                print("Early stopping")
                self.epoch_stop = epoch
                break
            else:
                self.epoch_stop = epoch
            sys.stdout.flush()

        best_model_path = checkpoint_dir + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.metrics_log = metrics_log 
        return self.model

    def validation(self):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(self.val_loader):
                batch_x = batch_x.float().to(self.device)
                label = label.long().squeeze(-1).to(self.device)
                padding_mask = padding_mask.float().to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.args.amp):
                    if self.args.model == 'DNN':
                        logits = self.model(batch_x, padding_mask, None, None)
                        loss = nn.functional.cross_entropy(logits, label, reduction='none')
                    else:
                        logits, model_info = self.model(batch_x, padding_mask, None, None)
                        loss = nn.functional.cross_entropy(logits, label, reduction='none') + model_info.loss.mean()
                total_loss.append(loss.flatten())

                preds.append(logits.cpu())
                trues.append(label.cpu())

        total_loss = torch.cat(total_loss, dim=0).mean().item()

        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        probs = torch.nn.functional.softmax(preds, dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = accuracy_score(predictions, trues)

        self.model.train()
        print("验证集损失：{},准确率:{}".format(total_loss,accuracy))
        return total_loss, accuracy


    def test(self, save_csv=True, result_dir=None):
        if not os.path.isdir(result_dir):
            try:
                os.makedirs(result_dir)
            except:
                pass
        
        @dataclass
        class Buffer:
            x_data: list = field(default_factory=list)
            trues: list = field(default_factory=list)
            preds: list = field(default_factory=list)
            shapelet_preds: list = field(default_factory=list)
            dnn_preds: list = field(default_factory=list)
            p: list = field(default_factory=list)
            d: list = field(default_factory=list)
            eta: list = field(default_factory=list)
            loss: list = field(default_factory=list)

        buffer = Buffer()
        self.model.eval()
        with torch.no_grad():
            buffer_x ,buffer_branch,buffer_channel,buffer_temporal= [],[],[],[]
            for i, (batch_x, label, padding_mask) in enumerate(self.test_loader):
                batch_x = batch_x.float().to(self.device)
                label = label.long().squeeze(-1).to(self.device)
                padding_mask = padding_mask.float().to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.args.amp):
                    if self.args.model == 'DNN':
                        logits = self.model(batch_x, padding_mask, None, None)
                        loss = nn.functional.cross_entropy(logits, label, reduction='none')
                    else:
                        if self.args.dnn_type=="MultiScaleAttentionNet":
                            logits, model_info,attention = self.model(batch_x, padding_mask, None, None, gating_value=self.args.gating_value,return_attentions=True)
                            buffer_x.append(batch_x.cpu())

                            # 保存注意力权重
                            buffer_branch.append(attention['branch'])
                            buffer_channel.append(attention['channel'])
                            buffer_temporal.append(attention['temporal'])
                        else:
                            logits, model_info = self.model(batch_x, padding_mask, None, None, gating_value=self.args.gating_value)
                        loss = nn.functional.cross_entropy(logits, label, reduction='none') + model_info.loss.mean()
                
                buffer.loss.append(loss.flatten())
                buffer.x_data.append(batch_x.cpu())
                buffer.trues.append(label.cpu())
                buffer.preds.append(logits.cpu())
                if self.args.model != 'DNN':
                    buffer.p.append(model_info.p.cpu())
                    buffer.d.append(model_info.d.cpu())
                    buffer.shapelet_preds.append(model_info.shapelet_preds.cpu())
                    if self.args.model == 'InterpGN':
                        buffer.eta.append(model_info.eta.cpu())
                        buffer.dnn_preds.append(model_info.dnn_preds.cpu())
                        
        probs = torch.nn.functional.softmax(torch.cat(buffer.preds, dim=0), dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1)  # (total_samples,) int class index for each sample
        trues = torch.cat(buffer.trues, dim=0).flatten()
        class_report = classification_report(trues, predictions, output_dict=True)  # 含 F1, precision, recall
        auc_score = None
        try:
            if probs.shape[1] == 2:
                auc_score = roc_auc_score(trues, probs[:, 1]) 
            else:
                auc_score = roc_auc_score(trues, probs, multi_class='ovr')  # 多分类
        except Exception as e:
            print("AUC 计算失败:", e)

        cm = confusion_matrix(trues, predictions)

        metrics_df = pd.DataFrame(class_report).transpose()
        metrics_df['AUC'] = auc_score

        cm_df = pd.DataFrame(cm, index=[f"True_{i}" for i in range(cm.shape[0])],
                                columns=[f"Pred_{i}" for i in range(cm.shape[1])])

        metrics_csv_path = f"{result_dir}/classification_metrics_{self.args.dataset}.csv"
        metrics_df.to_csv(metrics_csv_path, index=True)
        print(f"Classification metrics saved at: {metrics_csv_path}")

        cm_csv_path = f"{result_dir}/confusion_matrix_{self.args.dataset}.csv"
        cm_df.to_csv(cm_csv_path, index=True)
        print(f"Confusion matrix saved at: {cm_csv_path}")

        accuracy = accuracy_score(predictions.cpu().numpy(), trues.cpu().numpy())


        cls_result = ClassificationResult(
            x_data=torch.cat(buffer.x_data, dim=0).cpu(),
            trues=trues.cpu(),
            preds=predictions.cpu(),
            loss=torch.cat(buffer.loss, dim=0).mean().item(),
            accuracy=accuracy
        )
        # cls_result.x = torch.cat(buffer_x, dim=0).cpu().numpy()
        cls_result.x = torch.cat(buffer.x_data, dim=0).cpu().numpy()
        cls_result.attn_channel = [
            torch.cat(layer, dim=0).to(torch.float32).cpu().numpy()
            for layer in zip(*buffer_channel)
        ]
        cls_result.attn_branch = [
            torch.cat(layer, dim=0).to(torch.float32).cpu().numpy()
            for layer in zip(*buffer_branch)
        ]
        cls_result.attn_temporal = [
            torch.cat(layer, dim=0).to(torch.float32).cpu().numpy()
            for layer in zip(*buffer_temporal)
        ]


        
        if self.args.model != 'DNN':
            cls_result.p = torch.cat(buffer.p, dim=0).cpu()
            cls_result.d = torch.cat(buffer.d, dim=0).cpu()
            cls_result.shapelet_preds = torch.cat(buffer.shapelet_preds, dim=0).cpu()
            if self.args.model == 'InterpGN':
                cls_result.eta = torch.cat(buffer.eta, dim=0).cpu()
                cls_result.w = self.model.sbm.output_layer.weight.detach().cpu()
                cls_result.shapelets = self.model.sbm.get_shapelets()
            else:
                cls_result.w = self.model.output_layer.weight.detach().cpu()
                cls_result.shapelets = self.model.get_shapelets()

        if save_csv:
            summary_dict = dict()
            for arg in vars(self.args):
                if arg in [
                    'model', 'dataset', 'dnn_type', 
                    'train_epochs', 'num_shapelet', 'lambda_reg', 'lambda_div', 'epsilon', 'lr', 
                    'seed', 'pos_weight', 'beta_schedule', 'gating_value',
                    'distance_func', 'sbm_cls'
                ]:
                    summary_dict[arg] = [getattr(self.args, arg)]

            summary_dict['test_accuracy'] = accuracy
            summary_dict['epoch_stop'] = self.epoch_stop
            if self.args.model != 'DNN':
                summary_dict['eta_mean'] = cls_result.eta.mean().cpu().item() if self.args.model == 'InterpGN' else None
                summary_dict['eta_std'] = cls_result.eta.std().cpu().item() if self.args.model == 'InterpGN' else None
                summary_dict['shapelet_score'] = compute_shapelet_score(cls_result.d, cls_result.w, cls_result.preds, cls_result.trues)
                summary_dict['w_count_1'] = (cls_result.w.abs() > 1).float().sum().item()
                summary_dict['w_ratio_1'] = (cls_result.w.abs() > 1).float().mean().item()
                summary_dict['w_count_0.5'] = (cls_result.w.abs() > 0.5).float().sum().item()
                summary_dict['w_ratio_0.5'] = (cls_result.w.abs() > 0.5).float().mean().item()
                summary_dict['w_count_0.1'] = (cls_result.w.abs() > 0.1).float().sum().item()
                summary_dict['w_ratio_0.1'] = (cls_result.w.abs() > 0.1).float().mean().item()
                summary_dict['w_max'] = cls_result.w.abs().max().item()
                summary_dict['w_gini_clip'] = gini_coefficient(np.clip(cls_result.w, 0, None))
                summary_dict['w_gini_abs'] = gini_coefficient(np.abs(cls_result.w))
                for attr, value in vars(cls_result).items():
                    if isinstance(value, torch.Tensor):
                        if value.ndim == 0:
                            summary_dict[attr] = value.item()
                        else:
                            summary_dict[attr] = value.to(torch.float32).cpu().numpy()
                    else:
                        summary_dict[attr] = value
            # summary_dict_clean = {}
            # for k, v in summary_dict.items():
            #     if isinstance(v, np.ndarray):
            #         if v.ndim == 0:
            #             summary_dict_clean[k] = v.item()
            #         elif v.ndim == 1:
            #             summary_dict_clean[k] = v
            #         else:
            #             summary_dict_clean[k] = v.tolist()  
            #     else:
            #         summary_dict_clean[k] = v
            # summary_df = pd.DataFrame(summary_dict_clean)
            # current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            # summary_df.to_csv(f"{result_dir}/{self.args.dataset}-{self.args.seed}-{self.args.model}-{self.args.num_shapelet}-{self.args.lambda_div}-{self.args.lambda_reg}-{current_time}.csv", index=False)
            # print(f"Test summary saved at: {result_dir}/{self.args.dataset}-{self.args.seed}-{self.args.model}-{self.args.num_shapelet}-{self.args.lambda_div}-{self.args.lambda_reg}-{current_time}.csv")
        # loss, accuracy = cls_result.loss, cls_result.accuracy
        # self.metrics_log.append({
        #     "epoch": "test", 
        #     "train_loss": None,
        #     "val_loss": None,
        #     "val_accuracy": None,
        #     "test_loss": loss,
        #     "test_accuracy": accuracy,
        #     "time_elapsed_sec": None,
        #     "time_remaining_sec": None
        # })
        # metrics_df = pd.DataFrame(self.metrics_log)
        # metrics_df = metrics_df.sort_values(by="epoch", key=lambda col: pd.to_numeric(col, errors="coerce"))
        # current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # csv_path = f"{result_dir}/metrics_log_{self.args.dataset}_{current_time}.csv"
        # metrics_df.to_csv(csv_path, index=False)
        # print(f"Full training/testing log saved at: {csv_path}")
        return cls_result.loss, cls_result.accuracy, summary_dict
    
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args_from_dict(config_dict):
    args = SimpleNamespace(**config_dict)
    # 添加派生属性
    args.root_path = f"{args.data_root}/{args.dataset}"
    args.is_training = True
    return args
# def get_dnn_model(configs):
#     dnn_dict = {
#         'FCN': FullyConvNetwork,
#         'Transformer': Transformer,
#         'TimesNet': TimesNet,
#         'PatchTST': PatchTST,
#         'ResNet': ResNet
# MultiScaleAttentionNet
#     }
#     return dnn_dict[configs.dnn_type](configs)


# class Experiment(object):
#     model_dict = {
#         'InterpGN': InterpGN,
#         'SBM': ShapeBottleneckModel,
#         'LTS': DistThresholdSBM,
#         'DNN': get_dnn_model
#     }


default_args_dict = {
    "data": "UEA",
    "data_root": "./data/UEA_multivariate",
    "model": "SBM",
    "dnn_type": "FCN",
    "dataset": "ours",
    "lambda_reg": 0.1,
    "lambda_div": 0.1,
    "epsilon": 1.0,
    "num_shapelet": 10,
    "gating_value": None,
    "pos_weight": False,
    "sbm_cls": "linear",
    "distance_func": "euclidean",
    "beta_schedule": "constant",
    "memory_efficient": False,
    "lr": 5e-3,
    "lr_decay": False,
    "gradient_accumulation_steps": 1,
    "gradient_clip": 0,
    "batch_size": 64,
    "log_interval": 20,
    "min_epochs": 0,
    "train_epochs": 500,
    "num_workers": 0,
    "patience": 50,
    "multi_gpu": False,
    "test_only": True,
    "seed": 42,
    "amp": True,
    "task_name": "classification",
    "model_id": "test",
    "embed": "timeF",
    "freq": "h",
    "top_k": 5,
    "num_kernels": 6,
    "enc_in": 4,
    "dec_in": 7,
    "c_out": 7,
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 2048,
    "moving_avg": 25,
    "factor": 1,
    "distil": True,
    "dropout": 0.0,
    "activation": "gelu",
    "output_attention": False,
    "label_len": 48,
    "pred_len": 96,
    "seasonal_patterns": "Monthly",
    "inverse": False,
    "step": 1,  
    "seq_len": 128, 
    "folder_path": "ourtrain/testdata", 
    "data_columns": ['T0', '粉尘', '温度', '风速'],
    "label_column": "label"
}


if __name__ == "__main__":
    args = get_args_from_dict(default_args_dict)

    random_seeds = [0, 42, 1234, 8237, 2023] if args.seed == -1 else [copy.deepcopy(args.seed)]

    for i, seed in enumerate(random_seeds):
        set_seed(seed)
        args.seed = seed

        print(f"{'=' * 5} Experiment {i} {'=' * 5} ", flush=True)
        experiment = Experiment(args=args)
        experiment.print_args()
        print()

        if not args.test_only:
            experiment.train()
            print(f"{'=' * 5} Training Done {'=' * 5} ")
            torch.cuda.empty_cache()
            print()
        
        experiment.model.load_state_dict(torch.load(f"{experiment.checkpoint_dir}/checkpoint.pth"))
        print(f"{'=' * 5} Test {'=' * 5} ")
        # test_loss, _, test_df = experiment.test(
        #     save_csv=True,
        #     result_dir=f"./result/{args.model}"
        # )
        test_loss, _, test_df = experiment.test(
            save_csv=True,
            result_dir=f"./result/{args.model}/{args.dnn_type}"
        )
        import pickle
        with open(f"{experiment.checkpoint_dir}/test_results.pkl", 'wb') as f:
            pickle.dump(test_df, f)
        print(f"Test results saved at: {experiment.checkpoint_dir}/test_results.pkl")

        print(f"Test | Loss {test_loss:.4f}")
        print()
        torch.cuda.empty_cache()

