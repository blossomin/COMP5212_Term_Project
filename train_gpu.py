import os
import gc
import sys
import json
import time
import random
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm.notebook import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from efficientnet_pytorch import model as enet
import albumentations as albu
from apex import amp
import warnings
# import logging, sys
# logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
warnings.filterwarnings('ignore', category=UserWarning) 

LOGPRINT = True

def log(s):
    if LOGPRINT:
        print(s)

log("hello world for plant training")


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda')
log(f"device:{DEVICE}")

# logging.debug(f'cuda:{DEVICE}')
# logging.info('We processed %d records', len(processed_records))

VER = 'v0.1'
DEBUG = False
PARAMS = {
    'version': VER,
    'folds': 3,
    'folds_train': None,
    'img_size': 300, #224=B0 240=B1 260=B2 300=B3 380=B4 456=B5 528=B6 600=B7
    'batch_size': 16,
    'workers': 8,
    'epochs': 2 if DEBUG else 20,
    'warmup': False,
    'dropout': .4,
    'backbone': 'efficientnet-b3', # 'efficientnet-bX' or 'resnext'
    'seed': 20221126,
    'aughard': True,
    'lr': .0005,
    'average': 'macro', # 'micro', 'macro' or 'samples'
    'apex': True,
    'comments': 'f1 score'
}
DATA_PATH = '../../dataset'
IMGS_PATH = f'{DATA_PATH}/train_images/'
MDLS_PATH = f'./models_{VER}'
if not os.path.exists(MDLS_PATH):
    os.mkdir(MDLS_PATH)

def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_all(PARAMS['seed'])

start_time = time.time()

if DEBUG:
    df_train = pd.read_csv(f'{DATA_PATH}/train.csv').sample(100).reset_index(drop=True)
else:
    df_train = pd.read_csv(f'{DATA_PATH}/train.csv')
df_sub = pd.read_csv(f'{DATA_PATH}/sample_submission.csv')
# display(df_train.head())
# display(df_train.labels.value_counts())
labels = []
for lbl in list(set(df_train.labels)):
    labels.extend(lbl.split())
labels = list(set(labels))

log(f"labels:{labels}")

LABELS = {i: x for i, x in enumerate(sorted(labels))}
LABELS_ = {x: i for i, x in enumerate(sorted(labels))}
PARAMS['labels'] = LABELS
PARAMS['labels_'] = LABELS_
with open(f'{MDLS_PATH}/params.json', 'w') as file:
    json.dump(PARAMS, file)
del file;
print('labels:', LABELS)
print('labels_:', LABELS_)

skf = StratifiedKFold(PARAMS['folds'], shuffle=True, random_state=PARAMS['seed'])
df_train['fold'] = -1
for i, (train_idx, valid_idx) in enumerate(skf.split(df_train, df_train['labels'])):
    df_train.loc[valid_idx, 'fold'] = i
# display(df_train.head())

df_occurence = pd.DataFrame({
    'origin': df_train.labels.value_counts(normalize=True),
    'fold_0': df_train[df_train.fold == 0].labels.value_counts(normalize=True),
    'fold_1': df_train[df_train.fold == 1].labels.value_counts(normalize=True),
    'fold_2': df_train[df_train.fold == 2].labels.value_counts(normalize=True),
    'fold_3': df_train[df_train.fold == 3].labels.value_counts(normalize=True),
    'fold_4': df_train[df_train.fold == 4].labels.value_counts(normalize=True)})
df_occurence.plot.barh(figsize=[12, 6], colormap='plasma')
plt.show()


#!g1.1
if PARAMS['aughard']:
    aug = albu.Compose([
        albu.OneOf([
            albu.RandomBrightness(limit=.2, p=1), 
            albu.RandomContrast(limit=.2, p=1), 
            albu.RandomGamma(p=1)
        ], p=.5),
        albu.OneOf([
            albu.Blur(blur_limit=3, p=1),
            albu.MedianBlur(blur_limit=3, p=1)
        ], p=.25),
        albu.OneOf([
            albu.GaussNoise(0.002, p=.5),
            albu.augmentations.geometric.transforms.Affine(p=.5),
        ], p=.25),
        albu.RandomRotate90(p=.5),
        albu.HorizontalFlip(p=.5),
        albu.VerticalFlip(p=.5),
        albu.Transpose(p=.5),
        albu.Cutout(
            num_holes=10, 
            max_h_size=int(.1 * PARAMS['img_size']), 
            max_w_size=int(.1 * PARAMS['img_size']), 
            p=.25),
        albu.ShiftScaleRotate(p=.5)
    ])
else:
    aug = albu.Compose([
        albu.OneOf([
            albu.RandomBrightness(limit=.2, p=1), 
            albu.RandomContrast(limit=.2, p=1), 
            albu.RandomGamma(p=1)
        ], p=.5),
        albu.RandomRotate90(p=.25),
        albu.HorizontalFlip(p=.25),
        albu.VerticalFlip(p=.25)
    ])


#!g1.1
def flip(img, axis=0):
    if axis == 1:
        return img[::-1, :, ]
    elif axis == 2:
        return img[:, ::-1, ]
    elif axis == 3:
        return img[::-1, ::-1, ]
    else:
        return img

class PlantDataset(data.Dataset):
    
    def __init__(self, df, size, labels, transform=None, tta=0):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.labels = labels
        self.transform = transform
        self.tta = tta
    
    def __len__(self):
        return self.df.shape[0]
    
      
    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.image
        img_path = f'{IMGS_PATH}/{img_name}'
        img = cv2.imread(img_path)
        if not np.any(img):
            print('no img file read:', img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255
        if self.transform is not None:
            img = self.transform(image=img)['image']
        if self.labels:
            img = img.transpose(2, 0, 1)
            label = np.zeros(len(self.labels)).astype(np.float32)
            for lbl in row.labels.split():
                label[self.labels[lbl]] = 1
            return torch.tensor(img), torch.tensor(label)
        else:
            img = flip(img, axis=self.tta)
            img = img.transpose(2, 0, 1)
            return torch.tensor(img.copy())


dataset_show = PlantDataset(
    df=df_train,
    size=PARAMS['img_size'],
    labels=LABELS_,
    transform=aug
)
img_test, lbl_test = dataset_show.__getitem__(7)
img_test = img_test.numpy().transpose([1, 2, 0])
img_test = np.clip(img_test, 0, 1)
plt.imshow(img_test)
plt.title(lbl_test)
plt.show()

#!g1.1
class EffNet(nn.Module):
    
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        self.enet = enet.EfficientNet.from_name(params['backbone'])
        nc = self.enet._fc.in_features
        self.enet._fc = nn.Identity()
        self.myfc = nn.Sequential(
            nn.Dropout(params['dropout']),
            nn.Linear(nc, int(nc / 4)),
            nn.Dropout(params['dropout']),
            nn.Linear(int(nc / 4), out_dim)
        )
        
    def extract(self, x):
        return self.enet(x)
    
    def forward(self, x):
        x = self.enet(x)
        x = self.myfc(x)
        return x

class ResNext(nn.Module):
    
    def __init__(self, params, out_dim):
        super(ResNext, self).__init__()
        self.rsnxt = torchvision.models.resnext50_32x4d(pretrained=True)
        nc = self.rsnxt.fc.in_features
        self.rsnxt.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nc, int(nc / 4)),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(int(nc / 4), out_dim)
        )
        self.rsnxt = nn.DataParallel(self.rsnxt)
        
    def forward(self, x):
        x = self.rsnxt(x)
        return x
    

#!g1.1
criterion = nn.BCEWithLogitsLoss()

def train_epoch(loader, optimizer):
    model.train()
    train_loss = []
    bar = tqdm(loader, desc='ep')
    for (data, target) in bar:
        data, target = data.to(DEVICE), target.to(DEVICE)
        loss_func = criterion
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_func(logits, target)
        if PARAMS['apex']:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: {:.4f}, smth: {:.4f}'.format(loss_np, smooth_loss))
    return train_loss


def val_epoch(loader, get_output=False, verbose=False):
    model.eval()
    val_loss = []
    val_logits = []
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for (data, target) in tqdm(loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            logits = model(data)
            loss = criterion(logits, target)
            pred = logits.sigmoid().detach().round()
            val_logits.append(logits)
            val_preds.append(pred)
            val_targets.append(target)
            val_loss.append(loss.detach().cpu().numpy())
    val_loss = np.mean(val_loss)
    val_logits = torch.cat(val_logits).cpu().numpy()
    val_preds = torch.cat(val_preds).cpu().numpy()
    val_targets = torch.cat(val_targets).cpu().numpy()
    val_acc = (val_preds == val_targets).mean() * 100
    val_f1 = f1_score(val_targets, val_preds, average=PARAMS['average'])
    if verbose:
        print('val acc: {:.2f} | val loss: {:.4f} | val f1: {:.4f}'.format(val_acc, val_loss, val_f1))
    if get_output:
        return val_logits
    else:
        return val_loss, val_acc, val_f1


pred, target = [], []
preds_val, target_val = [], []

if DEBUG:
    n_folds_train = 2
else:
    n_folds_train = PARAMS['folds'] if not PARAMS['folds_train'] else PARAMS['folds_train']
start_folds_train = 0

for fold_num in range(start_folds_train, n_folds_train):
    print('=' * 20, 'FOLD:', fold_num, '=' * 20)
    train_idxs = np.where((df_train['fold'] != fold_num))[0]
    val_idxs = np.where((df_train['fold'] == fold_num))[0]
    df_fold  = df_train.loc[train_idxs]
    df_val = df_train.loc[val_idxs]
    dataset_train = PlantDataset(
        df=df_fold,
        size=PARAMS['img_size'],
        labels=LABELS_,
        transform=aug
    )
    dataset_val = PlantDataset(
        df=df_val,
        size=PARAMS['img_size'],
        labels=LABELS_,
        transform=None
    )
    train_loader = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=PARAMS['batch_size'], 
        sampler=RandomSampler(dataset_train), 
        num_workers=PARAMS['workers']
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=PARAMS['batch_size'], 
        sampler=SequentialSampler(dataset_val), 
        num_workers=PARAMS['workers']
    )
    if PARAMS['backbone'] == 'resnext':
        model = ResNext(params=PARAMS, out_dim=len(LABELS_)) 
    else:
        model = EffNet(params=PARAMS, out_dim=len(LABELS_)) 
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=PARAMS['lr'])
    if PARAMS['apex']:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    if PARAMS['warmup']:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=PARAMS['lr'], 
            total_steps=PARAMS['epochs'],
            div_factor=(PARAMS['lr'] / 1e-5), 
            final_div_factor=1000,
            pct_start=(int(.1 * PARAMS['epochs']) / PARAMS['epochs']),
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, PARAMS['epochs'])
    print('train len:', len(dataset_train),'| val len:', len(dataset_val))
    best_file = '{}/model_best_{}.pth'.format(MDLS_PATH, fold_num)
    acc_max = 0
    f1_max = 0
    for epoch in tqdm(range(PARAMS['epochs']), desc='epochs'):
        print(time.ctime(), 'epoch:', epoch)
        train_loss = train_epoch(train_loader, optimizer)
        val_loss, acc, f1 = val_epoch(val_loader)
        scheduler.step(epoch)
        content = '{} epoch {}, lr: {:.8f}, train loss: {:.4f}, val loss: {:.4f}, acc: {:.2f}, val f1: {:.4f}'.format(
            time.ctime(),
            epoch, 
            optimizer.param_groups[0]['lr'], 
            np.mean(train_loss),
            np.mean(val_loss),
            acc,
            f1
        )
        print(content)
        with open('{}/log_{}.txt'.format(MDLS_PATH, fold_num), 'a') as appender:
            appender.write(content + '\n')
        if f1 > f1_max:
            torch.save(model.state_dict(), best_file)
            print('f1 improved {:.2f} --> {:.2f} model saved'.format(f1_max, f1))
            f1_max = f1
            preds_best, target_best = [], []
            with torch.no_grad():
                for img_data, img_lbls in tqdm(val_loader):
                    img_data = img_data.to(DEVICE)
                    preds = np.squeeze(model(img_data).sigmoid().cpu().numpy())
                    preds_best.extend(preds)
                    target_best.extend(img_lbls.cpu().numpy())
            print('val preds done:', len(preds_best), len(target_best))
    preds_val.extend(preds_best)
    target_val.extend(target_best)
    with open('log_total.txt', 'a') as appender:
        appender.write('{} | fold: {} | max f1: {:.2f}\n'.format(PARAMS, fold_num, f1_max))
    torch.save(
        model.state_dict(), 
        os.path.join('{}/model_final_{}.pth'.format(MDLS_PATH, fold_num))
    )
    del model, dataset_train, dataset_val, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

preds_val = np.array(preds_val)
target_val = np.array(target_val)

elapsed_time = time.time() - start_time
print(f'time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec')


th_dict = {}
for i, lbl in LABELS.items():
    f1_max = 0
    for th in np.linspace(.1, 1, 100):
        f1 = f1_score(preds_val[:, i] > th, target_val[:, i])
        if f1 > f1_max:
            f1_max = f1
            th_max = th
    th_dict[i] = th_max
    print(lbl, '| f1 max:', np.round(f1_max, 2), '| th max:', np.round(th_max, 2))
    
with open(f'{MDLS_PATH}/ths.json', 'w') as file:
    json.dump(th_dict, file)
del file;
