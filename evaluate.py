import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
from util import *
from dataset import *
from gfcorrlap_unetlight import *
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, recall_score, precision_score
import sklearn
import segmentation_models_pytorch as smp
import albumentations as A
import glob
import cv2
import segmentation_models_pytorch as smp
from performance_evaluation import dice_coeff,Percision,Recall,F1,Specificity

Precautions_msg = '(주의사항) ---- \n'

'''

 python evaluate.py --kernel-type unet --model-type unet --data-dir C:/Users/user/Desktop/ --data-folder newDIdataset/  --n-epochs 30 --batch-size 4

'''
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--kernel-type', type=str, required=True)
    # kernel_type : 실험 세팅에 대한 전반적인 정보가 담긴 고유 이름

    parser.add_argument('--model-type', type=str, required=False)
    # model_type : 모델 유형

    parser.add_argument('--data-dir', type=str, required=True)
    # base 데이터 폴더 ('./data/')

    parser.add_argument('--data-folder', type=str, required=True)
    # 데이터 세부 폴더 예: 'original_stone/'
    # os.path.join(data_dir, data_folder, 'train.csv')

    parser.add_argument('--image-size', type=int, default='512')
    # 입력으로 넣을 이미지 데이터 사이즈

    parser.add_argument('--use-amp', action='store_true')
    # 'A Pytorch EXtension'(APEX)
    # APEX의 Automatic Mixed Precision (AMP)사용
    # 기능을 사용하면 속도가 증가한다. 성능은 비슷
    # 옵션 00, 01, 02, 03이 있고, 01과 02를 사용하는게 적절
    # LR Scheduler와 동시 사용에 버그가 있음 (고쳐지기전까지 비활성화)
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/2309

    parser.add_argument('--out-dim', type=int, default=1)
    # 모델 출력 output dimension

    parser.add_argument('--DEBUG', action='store_true')
    # 디버깅용 파라미터 (실험 에포크를 5로 잡음)

    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    # 학습에 사용할 GPU 번호

    # parser.add_argument('--k-fold', type=int, default=4)
    # # data cross-validation
    # # k-fold의 k 값을 명시

    parser.add_argument('--log-dir', type=str, default='./logs')
    # Evaluation results will be printed out and saved to ./logs/
    # Out-of-folds prediction results will be saved to ./oofs/
    # 분할 했다가 다시 합친 결과

    parser.add_argument('--accumulation_step', type=int, default=2)
    # Gradient accumulation step
    # GPU 메모리가 부족할때, 배치를 잘개 쪼개서 처리한 뒤 합치는 기법
    # 배치가 30이면, 60으로 합쳐서 모델 업데이트함

    parser.add_argument('--model-dir', type=str, default='./weights')
    # weight 저장 폴더 지정
    # best :

    parser.add_argument('--batch-size', type=int, default=2)  # 배치 사이즈
    parser.add_argument('--num-workers', type=int, default=6)  # 데이터 읽어오는 스레드 개수
    parser.add_argument('--init-lr', type=float, default=1e-4)  # 초기 러닝 레이트. pretrained를 쓰면 매우 작은값
    parser.add_argument('--n-epochs', type=int, default=20)  # epoch 수

    args, _ = parser.parse_known_args()
    return args


def val_epoch(model, loader):

    model.eval()

    val_loss = []
    val_iou = []
    val_recall = []
    val_precision = []
    val_f1 = []
    val_auc=[]

    with torch.no_grad():
        for (data, target, image_name) in tqdm(loader):

            data, target = data.to(device), target.to(device)
            logits= model(data)

            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())

            logits = logits.detach().cpu()

            target = target.detach().cpu()

            logits = mask_binarization(logits, 0.5)

            iou_np = dice_coeff(logits, target).numpy()

            target = target.numpy()
            logits = logits.numpy()
            precision = Percision(logits, target)
            recall = Recall(logits, target)
            f1score = F1(logits, target)
            auc = roc_auc_score(target.flatten(), logits.flatten(), average='macro')

            val_iou.append(iou_np)
            val_recall.append(recall)
            val_precision.append(precision)
            val_f1.append(f1score)
            val_auc.append(auc)


        val_loss = np.mean(val_loss)
        val_iou = np.mean(val_iou)
        val_precision = np.mean(val_precision)
        val_f1 = np.mean(val_f1)
        val_recall = np.mean(val_recall)
        val_auc = np.mean(val_auc)

    return val_loss, val_iou, val_precision, val_f1, val_recall, val_auc

def main():
    data_dir = args.data_dir  
    data_folder = args.data_folder  

    df_val = pd.read_csv(os.path.join(data_dir, data_folder, 'test_default.csv'))

    aug = A.Compose([
        A.Resize(args.image_size, args.image_size)
    ])

    valid_dataset = EvalDataset(df_val, aug)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    model_file = os.path.join(args.model_dir, f'{args.kernel_type}_bests.pth')
    model = GfDecoder()



    model = model.to(device)
    try:  # single GPU model_file
        model.load_state_dict(torch.load(model_file), strict=True)
    except:  # multi GPU model_file
        state_dict = torch.load(model_file)
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=True)


    val_loss, val_iou, val_precision, val_f1, val_recall, val_auc = val_epoch(model, valid_loader)

    print('-------------------------------------최종 결과---------------------------------')
    print('test loss : ', f'{val_loss:.5f}')
    print('dice coefficient : ', f'{val_iou:.5f}')
    print('f1-score:', f'{val_f1:.5f}')
    print('recall-score:', f'{val_recall:.5f}')
    print('precision-score:', f'{val_precision:.5f}')
    print('auc-score:', f'{val_auc:.5f}')

if __name__ == '__main__':
    print('----------------------------')
    print(Precautions_msg)
    print('----------------------------')

    # argument값 만들기
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    # GPU가 여러개인 경우 멀티 GPU를 사용함
    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    # 실험 재현을 위한 random seed 부여하기
    set_seed(2359)
    device = torch.device('cuda')
    criterion = DiceBCELoss()

    # 메인 기능 수행
    main()
