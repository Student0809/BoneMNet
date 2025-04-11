import os
import sys
import json
import pickle
import random
import math
import torch
from open_clip import ClipLoss
import torch.nn.functional as F
from sklearn.metrics import classification_report
from collections import OrderedDict
import numpy as np
import time
label_queue = {"neutrophilic stab granulocyte": ['neutrophilic stab granulocyte belongs to granulocyte',  #
                                                     'A cell image of granulocyte',  # 细胞核形状
                                                     'neutrophilic stab granulocyte chromatin is clumpy',  # 染色质形状
                                                     'neutrophilic stab granulocyte cytoplasm is abandant and filled with granules and appears light blue'],
                   # 胞质颜色与特征
                   "polychromatic normoblast": ['polychromatic normoblast belongs to normoblast',
                                                'A cell image of normoblast',
                                                'polychromatic normoblast chromatin is clumpy',
                                                'polychromatic normoblast cytoplasm is abandant and appears gray and without granules'],
                   "neutrophilic myelocyte": ['neutrophilic myelocyte belongs to granulocyte',
                                              'A cell image of granulocyte',
                                              'neutrophilic myelocyte chromatin is clumpy',
                                              'neutrophilic myelocyte cytoplasm is abandant and appears blue and with granules'],
                   "neutrophilic segmented granulocyte": ['neutrophilic segmented granulocyte belongs to granulocyte',
                                                          'A cell image of granulocyte',
                                                          'neutrophilic segmented granulocyte chromatin is clumpy',
                                                          'neutrophilic segmented granulocyte cytoplasm is abandant and filled with granules and appears light blue'],
                   "Iymphoblast": ['Iymphoblast belongs to Iymphoblast',
                                   'A cell image of Iymphoblast',
                                   'Iymphoblast chromatin is granular',
                                   'Iymphoblast cytoplasm is sparse and without granules and appears blue'],
                   "neutrophilic metamyelocyte": ['neutrophilic metamyelocyte belongs to granulocyte',
                                                  'A cell image of granulocyte',
                                                  'neutrophilic metamyelocyte chromatin is clumpy',
                                                  'neutrophilic metamyelocyte cytoplasm is abandant and filled with granules'],
                   "myeloblast": ['myeloblast belongs to granulocyte',  # 细胞体
                                  'A cell image of granulocyte',  # 细胞核形状
                                  'myeloblast chromatin is granular',  # 染色质形状
                                  'myeloblast cytoplasm is sparse and without granules and appears blue'],  # 胞质颜色与特征
                   "orthochromatic normoblast": ['orthochromatic normoblast belongs to normoblast',  # 细胞体
                                                 'A cell image of normoblast',  # 细胞核形状
                                                 'orthochromatic normoblast chromatin is clumpy with dark cluster',
                                                 # 染色质形状
                                                 'orthochromatic normoblast cytoplasm is abundant and without granules and appears light red'],
                   # 胞质颜色与特征
                   "prelymphocyte": ['prelymphocyte belongs to lymphocyte',  # 细胞体
                                     'A cell image of lymphocyte',  # 细胞核形状
                                     'prelymphocyte chromatin is clumpy',  # 染色质形状
                                     'prelymphocyte cytoplasm is sparse and with granules and appears blue'],  # 胞质颜色与特征
                   "abnormal promyelocyte": ['abnormal promyelocyte belongs to granulocyte',  # 细胞体
                                             'A cell image of granulocyte',  # 细胞核形状
                                             'abnormal promyelocyte chromatin is fine',  # 染色质形状
                                             'abnormal promyelocyte cytoplasm is sparse and with granules and appears purple'],
                   # 胞质颜色与特征
                   "monocyte": ['monocyte belongs to monocyte',  # 细胞体
                                'A cell image of monocyte',  # 细胞核形状
                                'monocyte chromatin is stripe',  # 染色质形状
                                'monocyte cytoplasm is sparse and with granules and appears light grey or light blue'],
                   # 胞质颜色与特征
                   "early normoblast": ['early normoblast belongs to normoblast',  # 细胞体
                                        'A cell image of normoblast',  # 细胞核形状
                                        'early normoblast chromatin is coarse granular',  # 染色质形状
                                        'early normoblast cytoplasm is sparse and without granules and appears dark blue'],
                   # 胞质颜色与特征
                   "monoblast": ['monoblast belongs to monocyte',  # 细胞体
                                 'A cell image of monocyte',  # 细胞核形状
                                 'monoblast chromatin is coarse granular',  # 染色质形状
                                 'monoblast cytoplasm is abundant and without granules and appears blue'],  # 胞质颜色与特征
                   "promyelocyte": ['promyelocyte belongs to granulocyte',  # 细胞体
                                    'A cell image of granulocyte',  # 细胞核形状
                                    'promyelocyte chromatin is fine',  # 染色质形状
                                    'promyelocyte cytoplasm is abundant and with granules and appears blue'],  # 胞质颜色与特征
                   "eosinophilic segmented granulocyte": ['eosinophilic segmented granulocyte belongs to granulocyte',
                                                          'A cell image of granulocyte',
                                                          'eosinophilic segmented granulocyte chromatin is clumpy',
                                                          'eosinophilic segmented granulocyte cytoplasm is abandant and filled with eosinophilic granules and appears light blue'],
                   "eosinophilic myelocyte": ['eosinophilic myelocyte belonsg to granulocyte',
                                              'A cell image of granulocyte',
                                              'eosinophilic myelocyte chromatin is clumpy',
                                              'eosinophilic myelocyte cytoplasm is abandant with eosinophilic granules and appears blue and with granules'],
                   "Multiple Myeloma Cells": ['Multiple Myeloma Cells belongs to plasmacyte',  # 细胞体
                                              'A cell image of plasmacyte',  # 细胞核形状
                                              'Multiple Myeloma Cells chromatin is irregular',  # 染色质形状
                                              'Multiple Myeloma Cells cytoplasm is abundant and with colorful granules'],
                   # 胞质颜色与特征
                   "smudge cells": ['smudge cells belongs to other',  # 细胞体
                                    'A cell image of other',  # 细胞核形状
                                    'smudge cells chromatin is unclear',  # 染色质形状
                                    'smudge cells cytoplasm is unclear'],  # 胞质颜色与特征
                   "plasmacyte": ['plasmacyte belongs to plasmacyte',  # 细胞体
                                  'A cell image of plasmacyte',  # 细胞核形状
                                  'plasmacyte chromatin is clumpy',  # 染色质形状
                                  'plasmacyte cytoplasm is abundant and with granules and appears dark blue'],
                   # 胞质颜色与特征
                   "other": ['other cell belongs to other',  # 细胞体
                             'A cell image of other',  # 细胞核形状
                             'other chromatin is unclear',  # 染色质形状
                             'other cytoplasm is unclear'],  # 胞质颜色与特征
                   }
def clip_loss(logit_per_image,logit_per_text,device):
    loss_per_image = F.cross_entropy(logit_per_image,torch.arange(logit_per_image.shape[0],device=device,dtype=torch.long))
    loss_per_text = F.cross_entropy(logit_per_text,torch.arange(logit_per_text.shape[0],device=device,dtype=torch.long))
    return (loss_per_image+loss_per_text)/2
def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引,这里应该为粗粒度的
        image_class = class_indices[cla]
        #之后要自己写一个细粒度文本标签,cla是文件夹名称
        #两个描述，一个是位置信息，一个是特征标识


        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    # plot_image = False
    # if plot_image:
    #     # 绘制每种类别个数柱状图
    #     plt.bar(range(len(flower_class)), every_class_num, align='center')
    #     # 将横坐标0,1,2,3,4替换为相应的类别名称
    #     plt.xticks(range(len(flower_class)), flower_class)
    #     # 在柱状图上添加数值标签
    #     for i, v in enumerate(every_class_num):
    #         plt.text(x=i, y=v + 5, s=str(v), ha='center')
    #     # 设置x坐标
    #     plt.xlabel('image class')
    #     # 设置y坐标
    #     plt.ylabel('number of images')
    #     # 设置柱状图的标题
    #     plt.title('flower class distribution')
    #     plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    # for data in data_loader:
    #     images, labels = data
    #     for i in range(plot_num):
    #         # [C, H, W] -> [H, W, C]
    #         img = images[i].numpy().transpose(1, 2, 0)
    #         # 反Normalize操作
    #         img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
    #         label = labels[i].item()
    #         plt.subplot(1, plot_num, i+1)
    #         plt.xlabel(class_indices[str(label)])
    #         plt.xticks([])  # 去掉x轴的刻度
    #         plt.yticks([])  # 去掉y轴的刻度
    #         plt.imshow(img.astype('uint8'))
    #     plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, label_name,device, epoch, lr_scheduler,tokenizer=None,preprocess=None):
    model.train()
    optimizer.zero_grad()
    CNN_correct = 0
    lambda_1 = 1
    # if epoch >= 100:
    #     lambda_1 = (300-epoch)/200 +0.0001
    alpha=0.3
    loss1 = ClipLoss()
    len_src_loader=len(data_loader)
    start_time = time.time()
    len_src_dataset = len(data_loader.dataset)
    num_classes = 20


    for step,(imgs,labels) in enumerate(data_loader):

        model.train()
        #试着调大batch
        # 用于MIXUP
        if labels.ndim == 2:
            labels = torch.max(labels, dim=1)[1]
        imgs,labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        child_classnames = [ 'neutrophilic stab granulocyte',
                             'polychromatic normoblast',
                             'neutrophilic myelocyte',
                             'neutrophilic segmented granulocyte',
                             'Iymphoblast',
                             'neutrophilic metamyelocyte',
                             'myeloblast',
                            'orthochromatic normoblast',
                             'prelymphocyte',
                             'abnormal promyelocyte',
                             'monocyte',
                            'early normoblast',
                            'monoblast',
                             'promyelocyte',
                            'eosinophilic segmented granulocyte',
                             'eosinophilic myelocyte',
                             'Multiple Myeloma Cells',
                             'smudge cells',
                             'plasmacyte',
                             'other',
                           ]
        #text = tokenizer([f'A single cell photo of a {label_name[k]}' for k in labels], context_length=77).to(device)
        text = tokenizer([f'A single cell photo of a {random.choice(label_name[k])}' for k in labels], context_length=77).to(device)
        parent_text_1 = "All cell photos are "
        for k in child_classnames:
            parent_text_1 = "All cell photos are " + k
        parent_text = tokenizer([parent_text_1 for _ in labels], context_length=256).to(device)
        #parent_text = tokenizer(parent_prefix, context_length=256).to(device)
        attention_masks = torch.where(text != 0, 1, 0).to(device)
        attention_masks_parent = torch.where(parent_text != 0, 1, 0).to(device)
        
        #loss_clip,label_src_pred = model(imgs, text,label_name,attention_masks)
        loss_clip,label_src_pred = model(imgs, text,labels,attention_masks,attention_masks_parent,parent_text)
        # print(imgage_prob.shape)
        # label_src_pred = torch.argsort(logits_per_image, dim=-1, descending=True)
        # print(label_src_pred)
        loss_cls = F.nll_loss(F.log_softmax(label_src_pred, dim=1), labels.long())  # 分类损失
        # if loss_cls < 0.5:
        #     lambda_1 = 0.1
        loss =loss_cls + lambda_1 * loss_clip # 总体损失
        # loss = loss_clip
        loss.backward()
        optimizer.step()
        pred = label_src_pred.data.max(1)[1]
        CNN_correct += pred.eq(labels.data.view_as(pred)).to(device).sum()
        # update lr
        lr_scheduler.step()
        optimizer.zero_grad()
        if step % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(epoch, step * len(imgs), len_src_dataset,
                                                             100. * step / len_src_loader))


            end_time = time.time()
            print('loss: {:.6f},  loss_cls: {:.6f},  loss_coarse: {:.6f}.lr:{:.5f}'.format(
                loss.item(), loss_cls.item(), loss_clip.item(),optimizer.param_groups[0]["lr"]))
            print(end_time-start_time)
            start_time = time.time()

    CCN_acc = CNN_correct / len_src_dataset

    print('[epoch: {:4}]  Train Accuracy: {:.4f} | train sample number: {:6}'.format(epoch, CCN_acc, len_src_dataset))
    return model,CCN_acc


@torch.no_grad()
def evaluate(model, data_loader, device, epoch,label_name,tokenizer):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()


    correct = 0
    loss_coarse = 0
    pred_list, label_list = [], []
    sample_num = 0
    len_tar_loader = len(data_loader)
    len_tar_dataset=len(data_loader.dataset)
    loss = 0
    cumulative_loss=0
    num_samples=0
    num_classes = 20

    with torch.no_grad():
        for data, label in data_loader:

            images, label = data.to(device), label.to(device)
            texts = tokenizer([f'A single cell photo of a {label_name[k]}.' for k in label], context_length=77).to(device)
            attention_masks = torch.where(texts != 0,1, 0).to(device)


            loss_coarse_, label_src_pred = model(images, texts, label_name,attention_masks)

            pred = label_src_pred.data.max(1)[1]
            pred_list.append(pred.cpu().numpy())
            label_list.append(label.cpu().numpy())
            loss += F.nll_loss(F.log_softmax(label_src_pred, dim=1), label.long()).item()
            loss_coarse += loss_coarse_.item()
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        loss /= len_tar_loader
        loss_coarse /= len_tar_loader
        print(
            'Average test loss: {:.4f}, loss clip: {:.4f}, test Accuracy: {}/{} ({:.2f}%), | test sample number: {:6}\n'.format(
                loss, loss_coarse, correct, len_tar_dataset, 100. * correct / len_tar_dataset, len_tar_dataset))

    return correct, correct / len_tar_dataset, pred_list, label_list



def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias") or "biomedclip.visual.trunk" in name:
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())

@torch.no_grad()
def test(model, data_loader, device,label_name):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    loss = 0
    correct = 0
    loss_coarse = 0
    pred_list, label_list = [], []
    sample_num = 0

    len_tar_loader = len(data_loader)
    len_tar_dataset=len(data_loader.dataset)
    with torch.no_grad():
        for data, label in data_loader:

            images, label = data.to(device), label.to(device)


            loss_coarse_, label_src_pred = model(images, 1, label_name,1)

            pred = label_src_pred.data.max(1)[1]
            pred_list.append(pred.cpu().numpy())
            label_list.append(label.cpu().numpy())
            loss += F.nll_loss(F.log_softmax(label_src_pred, dim=1), label.long()).item()
            loss_coarse += loss_coarse_.item()
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        loss /= len_tar_loader
        loss_coarse /= len_tar_loader
        print(
            'Average test loss: {:.4f}, loss clip: {:.4f}, test Accuracy: {}/{} ({:.2f}%), | test sample number: {:6}\n'.format(
                loss, loss_coarse, correct, len_tar_dataset, 100. * correct / len_tar_dataset, len_tar_dataset))
        true_label_list = []
        pred_label_list = []

        for i in range(len(pred_list)):
            pred = pred_list[i]
            label = label_list[i]
            for j in range(len(pred)):
                true_label_list.append(label[j])
                pred_label_list.append(pred[j])


        report=classification_report(true_label_list, pred_label_list, digits=4, labels=list(range(20)))
        print(report)
        # for k in class_indict:
        #     class_list.append(class_indict[k])
        # draw_confusion_matrix(true_label_list, predict_label_List,class_list,pdf_save_path='./martix.pdf')

    return correct, correct / len_tar_dataset, pred_list, label_list