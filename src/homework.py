from dataclasses import dataclass
import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from collections import OrderedDict
from torch import nn
from torch.autograd import Variable
from torchvision import transforms, models


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


@dataclass(frozen=True)
class MODEL_TYPE():
    ALEX: str = 'alex'
    SIX_ONE_THREE: str = '613'
    HYDRA: str = 'hydra'


class Sketch(Dataset):

    def __init__(self, root_dir, set_type, labels=None):
        self.root_dir = Path(root_dir)
        self.data = []
        self.labels = labels or OrderedDict({
            "hair": [],
            "gender": [],
            "earring": [],
            "smile": [],
            "frontal_face": [],
            "hair_color": [],
            "style": [],
        })
        with open(self.root_dir / f'anno_{set_type}.json') as f:
            for sketch in json.load(f):
                img_name = sketch['image_name'].replace('photo', 'sketch').replace('/image', '_sketch')
                img_path = self.root_dir / set_type / 'sketch' / ''.join([img_name, '.png'])
                self.data.append(self.get_img(img_path))
                for k in self.labels:
                    if k =='hair_color':
                        lbl = np.zeros(5)
                        lbl[sketch[k]] = 1
                    elif k == 'style':
                        lbl = np.zeros(3)
                        lbl[sketch[k]] = 1
                    else:
                        lbl = sketch[k]
                    self.labels[k].append(lbl)

    def __getitem__(self, item):
        trans = transforms.ToTensor()
        # return trans(self.data[item]), np.array([self.labels[k][item] for k in self.labels], dtype=np.float32)
        return trans(self.data[item]).to(device), OrderedDict({k: self.labels[k][item] for k in self.labels})

    def __len__(self):
        return len(self.data)

    def get_img(self, img_path, fixedsize=256):
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        iarr = np.array(img)
        if h < w:
            iarr = iarr.transpose((1, 0, 2))
        h_w = abs(h - w)
        left = h_w // 2
        right = h_w - left
        iarr = np.pad(iarr, ((0, 0), (left, right), (0, 0)), 'constant', constant_values=(1, 1))
        if h < w:
            iarr = iarr.transpose((1, 0, 2))
        return Image.fromarray(iarr).resize((fixedsize, fixedsize), Image.ANTIALIAS)


class HydraNet(nn.Module):

    def __init__(self, model_type=MODEL_TYPE.HYDRA):
        super(HydraNet, self).__init__()
        self.model_type = model_type
        # self.conv_block1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 16, (5, 5), padding=2),                  # 256*256*3 -> 256*256*16
            nn.ReLU(),                                            # ReLU
            nn.MaxPool2d(2, stride=2)                             # 256*256*16 -> 128*128*16
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, (3, 3)),                            # 128*128*16 -> 126*126*32
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)                             # 126*126*32 -> 63*63*32
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 8, (3, 3), padding=1, stride=(2, 2)),   # 63*63*32 -> 32*32*8
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)                             # 32*32*8 -> 16*16*8
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 16 * 8, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 72),
            nn.ReLU()
        )
        self.hair = nn.Sequential(
            nn.Linear(72, 1),
            nn.Sigmoid())
        self.gender = nn.Sequential(
            nn.Linear(72, 1),
            nn.Sigmoid())
        self.earring = nn.Sequential(
            nn.Linear(72, 1),
            nn.Sigmoid())
        self.smile = nn.Sequential(
            nn.Linear(72, 1),
            nn.Sigmoid())
        self.front_face = nn.Sequential(
            nn.Linear(72, 1),
            nn.Sigmoid())
        self.hair_color = nn.Sequential(  # 多分类
            nn.Linear(72, 5),
            nn.Softmax())
        self.style = nn.Sequential(       # 多分类
            nn.Linear(72, 3),
            nn.Softmax())

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # x = torch.flatten(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return dict(
            hair = self.hair(x),
            gender = self.gender(x),
            earring = self.earring(x),
            smile = self.smile(x),
            frontal_face = self.front_face(x),
            hair_color = self.hair_color(x),
            style = self.style(x)
        )


class ModAlexNet(nn.Module):

    def __init__(self, model_type=MODEL_TYPE.ALEX) -> None:
        super(ModAlexNet, self).__init__()
        self.model_type = model_type
        self.alex= models.alexnet(pretrained=True)
        in_features = self.alex.classifier[6].out_features
        self.alex.classifier = nn.Sequential(
            *list(self.alex.classifier.children()),
            nn.ReLU()
        )

        self.hair = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid())
        self.gender = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid())
        self.earring = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid())
        self.smile = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid())
        self.front_face = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid())
        self.hair_color = nn.Sequential(  # 多分类
            nn.Linear(in_features, 5),
            nn.Softmax())
        self.style = nn.Sequential(       # 多分类
            nn.Linear(in_features, 3),
            nn.Softmax())

    def forward(self, x_in):
        x = self.alex(x_in)
        return dict(
            hair = self.hair(x),
            gender = self.gender(x),
            earring = self.earring(x),
            smile = self.smile(x),
            frontal_face = self.front_face(x),
            hair_color = self.hair_color(x),
            style = self.style(x)
        )


class SixOneThreeNet(nn.Module):

    def __init__(self):
        super().__init__()
        # self.conv_block1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 6, (5, 5), padding=2),                   # 256*256*3 -> 256*256*6
            nn.ReLU(),                                            # ReLU
            nn.MaxPool2d(2, stride=2)                             # 256*256*6 -> 128*128*6
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(6, 16, (3, 3)),                             # 128*128*6 -> 126*126*16
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)                             # 126*126*16 -> 63*63*16
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(16, 16, (3, 3), padding=1, stride=(2, 2)),  # 63*63*16 -> 32*32*16
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)                             # 32*32*16 -> 16*16*16
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 16 * 16, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 72),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(72, 5),
            nn.Sigmoid()
        )

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # x = torch.flatten(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class SigmoidBCELoss(nn.BCELoss):
    def __init__(self):
        super(SigmoidBCELoss, self).__init__()

    def forward(self, outputs, y):
        sigmoid = nn.Sigmoid()
        bce = nn.BCELoss()
        loss = bce(sigmoid(outputs), y)
        return loss


def dataloader(batch_size, label):
    train_set = Sketch('../FS2K', 'train', label)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_set = Sketch('../FS2K', 'test', label)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader, test_loader


def get_loss(outs, y):
    total_loss = 0
    losses = {}
    f_loss_multi = nn.CrossEntropyLoss()
    f_loss_bin = nn.BCELoss()
    for k in y:
        if k in ['hair_color', 'style']:
            losses[k] = f_loss_multi(outs[k], Variable(y[k].to(device)))
        else:
            losses[k] = f_loss_bin(outs[k], Variable(y[k].to(device).reshape((-1, 1)).float()))
        total_loss += losses[k]
    return total_loss


def train(model, learn_rate, loader, epoch):
    optimizer = torch.optim.Adam(model.parameters(), learn_rate)
    for i in range(epoch):
        batch_loss = 0.0
        acc_sum = 0.0
        n = 0
        for j, (x, y) in enumerate(loader):
            x = Variable(x, requires_grad=True)
            # y = Variable(y)
            optimizer.zero_grad()
            model_out = model(x)
            loss = get_loss(model_out, y)
            loss.backward()
            optimizer.step()

            batch_loss += loss
            if (j + 1) % 8 == 0:  # print every 8 mini-batches
                print(f'[{i + 1}, {j + 1:5}] loss: {batch_loss / 10:.3f}')
                batch_loss = 0.0
    torch.save(model.state_dict(), f'../model/model-{model.model_type}_lr-{learn_rate}_ep-{epoch}.pth')


def test(model, loader):
    total = {}
    correct = {}
    diffuse = dict(
        hair_color=torch.zeros((5, 5)),
        style=torch.zeros((3, 3))
    )
    cases = {"hair_color": torch.zeros(5), "style": torch.zeros(3)}
    for img, label in loader:
        img = Variable(img)
        pred = model(img)
        for k in label:
            total.setdefault(k, 0)
            total[k] += pred[k].shape[0]
            diffuse.setdefault(k, torch.zeros((2,2)))
            cases.setdefault(k, torch.zeros(2))
            if k in ['hair_color', 'style']:
                lbl_pos = label[k].argmax(dim=1)
                pred_pos = pred[k].argmax(dim=1)
            else:
                pred_pos = pred[k].ge(0.5).reshape(-1).int()
                lbl_pos = label[k]
            for i, j in zip(lbl_pos, pred_pos):
                cases[k][i] += 1
                diffuse[k][i, j] += 1
            res = (pred_pos.cpu() == lbl_pos.cpu()).int().sum()
            correct.setdefault(k, 0)
            correct[k] += res
    for k, v in total.items():
        print(f'{k} test acc: {correct[k] / v}')
    show_indicate_table(diffuse)
    # print(f'test acc: {correct / total}')


def create_model(model_type):
    if model_type == MODEL_TYPE.ALEX:
        model = ModAlexNet()
    elif model_type == MODEL_TYPE.SIX_ONE_THREE:
        model = SixOneThreeNet()
    else:
        model = HydraNet()
    model.to(device)
    return model
    

def show_indicate_table(diffuse):
    show_confusion(diffuse.pop('hair_color'), '头发颜色')
    show_confusion(diffuse.pop('style'), '风格')
    indi = []
    kmap = dict(
        gender='性别',
        hair='有无头发',
        earring='有无耳朵',
        smile='是否微笑',
        frontal_face='是否正脸',
        # hair_color='头发颜色',
        # style='素描风格'
    )
    for k in kmap:
        m = diffuse[k]
        acc = m.diagonal().sum()/m.sum()
        pcs = (m.diagonal()/m.sum(dim=0))[1]
        rcl = (m.diagonal()/m.sum(dim=1))[1]
        tex = f'{kmap[k]}&{acc:.2%}&{pcs:.2%}&{rcl:.2%} \\\\'
        indi.append(tex)
    print('\n\\hline\n'.join(indi).replace(r'%', r'\%'))
    return


def show_confusion(m, title='Matrix', labels=('预测', '实际')):
    ax = plt.matshow(m, cmap=plt.cm.Reds)
    # plt.colorbar(ax.colorbar, fraction=0.025)
    plt.xlabel(labels[0], fontproperties='SimSun')
    plt.ylabel(labels[1], fontproperties='SimSun')
    plt.title(title, fontproperties='SimSun')
    for i, r in enumerate(m):
        for j, e in enumerate(r):
            plt.annotate(e.item(), xy=(j, i), horizontalalignment='center', verticalalignment='center')
    plt.show()
