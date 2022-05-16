# パッケージのimport
import datetime
import random
import math
import time
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2

# 初期設定
# Setup seeds
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

from utils.dataloader import make_datapath_list, DataTransform, COCOkeypointsDataset

# ITOPのファイルパスリスト作成
train_img_list, train_mask_list, val_img_list, val_mask_list, train_meta_list, val_meta_list = make_datapath_list(
    rootpath="./data/")

# Dataset作成
# 訓練データ
train_dataset = COCOkeypointsDataset(
    train_img_list, train_mask_list, train_meta_list, phase="train", transform=DataTransform())

# 検証データ
val_dataset = COCOkeypointsDataset(
    val_img_list, val_mask_list, val_meta_list, phase="val", transform=DataTransform())



# DataLoader作成
# batch_size = 128
batch_size = 16

train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

val_dataloader = data.DataLoader(
   val_dataset, batch_size=batch_size, shuffle=False)

# 辞書型変数にまとめる
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

train_dataset.__len__()

val_dataset.__len__()

train_img_list[24]

print(train_img_list[24])

img = cv2.imread(train_img_list[24])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

img = cv2.imread(val_img_list[240])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

val_mask_list[24]

train_mask_list[24]

print(val_meta_list[24])

from utils.openpose_net import OpenPoseNet

# 学習済みモデルと本章のモデルでネットワークの層の名前が違うので、対応させてロードする
# モデルの定義
net = OpenPoseNet()
# マルチGPUを使う場合
# net = torch.nn.DataParallel(OpenPoseNet())

# 学習済みパラメータをロードする
net_weights = torch.load(
    './weights/pose_model_scratch.pth', map_location={'cuda:0': 'cpu'})
keys = list(net_weights.keys())

weights_load = {}

# ロードした内容を、本書で構築したモデルの
# パラメータ名net.state_dict().keys()にコピーする
for i in range(len(keys)):
    weights_load[list(net.state_dict().keys())[i]
                 ] = net_weights[list(keys)[i]]

# コピーした内容をモデルに与える
state = net.state_dict()
state.update(weights_load)
net.load_state_dict(state)

# ネットワークを訓練モードにする
net.train()

print('ネットワーク設定完了:学習済みの重みをロードし、訓練モードに設定しました')

# 損失関数の設定
class OpenPoseLoss(nn.Module):
    """OpenPoseの損失関数のクラスです。"""

    def __init__(self):
        super(OpenPoseLoss, self).__init__()

    def forward(self, saved_for_loss, heatmap_target, heat_mask, paf_target, paf_mask):
        """
        損失関数の計算。

        Parameters
        ----------
        saved_for_loss : OpenPoseNetの出力(リスト)

        heatmap_target : [num_batch, 19, 46, 46]
            正解の部位のアノテーション情報

        heatmap_mask : [num_batch, 19, 46, 46]
            heatmap画像のmask

        paf_target : [num_batch, 38, 46, 46]
            正解のPAFのアノテーション情報

        paf_mask : [num_batch, 38, 46, 46]
            PAF画像のmask

        Returns
        -------
        loss : テンソル
            損失の値
        """

        total_loss = 0
        # ステージごとに計算します
        for j in range(6):

            # PAFsとheatmapsにおいて、マスクされている部分（paf_mask=0など）は無視させる
            # PAFs
            pred1 = saved_for_loss[2 * j] * paf_mask
            gt1 = paf_target.float() * paf_mask

            # heatmaps
            pred2 = saved_for_loss[2 * j + 1] * heat_mask
            gt2 = heatmap_target.float()*heat_mask

            total_loss += F.mse_loss(pred1, gt1, reduction='mean') + \
                F.mse_loss(pred2, gt2, reduction='mean')

        return total_loss


criterion = OpenPoseLoss()

optimizer = optim.SGD(net.parameters(), lr=1e-2,
                      momentum=0.9,
                      weight_decay=0.0001)

print(net)

# ファインチューニングで学習させるパラメータを、変数params_to_updateの1～3に格納する

params_to_update_1 = []
params_to_update_2 = []
params_to_update_3 = []

# 学習させる層のパラメータ名を指定
update_param_names_1 = ["model0.model"]
update_param_names_2 = ["model1_1.0.weight",
                        "model2_1.0.weight", "model1_2.0.weight", "model2_2.0.weight"]
update_param_names_3 = ["model6_2.10.weight", "model6_2.12.weight"]

# パラメータごとに各リストに格納する
for name, param in net.named_parameters():
    if update_param_names_1[0] in name:
        param.requires_grad = True
        params_to_update_1.append(param)
        print("params_to_update_1に格納:", name)

    elif name in update_param_names_2:
        param.requires_grad = True
        params_to_update_2.append(param)
        print("params_to_update_2に格納:", name)

    elif name in update_param_names_3:
        param.requires_grad = True
        params_to_update_3.append(param)
        print("params_to_update_3に格納:", name)

    else:
        param.requires_grad = False
        print("勾配計算なし。学習しない:", name)

# 最適化手法の設定 今回は特に学習率を層で変更したりスケジューラを設定する必要はなかった
optimizer = optim.SGD([
    {'params': params_to_update_1, 'lr': 1},
    {'params': params_to_update_2, 'lr': 1},
    {'params': params_to_update_3, 'lr': 1}
], momentum=0.9)

# 結果を描画するためのリスト
train_losses = []
val_losses = []

# モデルを学習させる関数を作成
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    # GPUが使えるかを確認
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("using", torch.cuda.device_count(), "GPUs!")

    # ネットワークをGPUへ
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # 画像の枚数
    num_train_imgs = len(dataloaders_dict["train"].dataset)
    num_val_imgs = len(dataloaders_dict["val"].dataset)

    batch_size = dataloaders_dict["train"].batch_size

    # イテレーションカウンタをセット
    iteration = 1
    
    # epochのループ
    for epoch in range(num_epochs):

        # 開始時刻を保存
        t_epoch_start = time.time()
        t_iter_start = time.time()
        epoch_train_loss = 0.0  # epochの損失和
        epoch_val_loss = 0.0  # epochの損失和

        print('-------------')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
                optimizer.zero_grad()
                print('（train）')
            else:

                net.eval()   # モデルを検証モードに
                print('-------------')
                print('（val）')
            
            # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
            if (epoch == 0) and (phase == 'train'):
                continue

            # データローダーからminibatchずつ取り出すループ
            for imges, heatmap_target, heat_mask, paf_target, paf_mask in dataloaders_dict[phase]:
                # ミニバッチがサイズが1だと、バッチノーマライゼーションでエラーになるのでさける
                if imges.size()[0] == 1:
                    continue

                # GPUが使えるならGPUにデータを送る
                imges = imges.to(device)
                heatmap_target = heatmap_target.to(device)
                heat_mask = heat_mask.to(device)
                paf_target = paf_target.to(device)
                paf_mask = paf_mask.to(device)

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    # (out6_1, out6_2)は使わないので _ で代替
                    _, saved_for_loss = net(imges)

                    loss = criterion(saved_for_loss, heatmap_target,
                                     heat_mask, paf_target, paf_mask)
                    del saved_for_loss
                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if (iteration % 20 == 0):  # 10iterに1度、lossを表示
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print('イテレーション {} || Loss: {:.4f} || 20iter: {:.4f} sec.'.format(
                                iteration, loss.item()/batch_size, duration))
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item()
                        iteration += 1

                    # 検証時
                    else:
                        epoch_val_loss += loss.item()

        # epochのphaseごとのlossと正解率
        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(
            epoch+1, epoch_train_loss/num_train_imgs, epoch_val_loss/num_val_imgs))
        
        train_losses.append(epoch_train_loss/num_train_imgs)
        val_losses.append(epoch_val_loss/num_val_imgs)
        
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

    # 最後のネットワークを保存する
    torch.save(net.state_dict(), 'weights/Illust100' + datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S") 
               + '.pth')


# 学習・検証を実行する
num_epochs = 100
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

# 結果の描画
plt.plot(train_losses, label='training loss')
plt.plot(val_losses, label='validation loss')
plt.title('Loss at the end of each epoch')
plt.legend()
plt.show()