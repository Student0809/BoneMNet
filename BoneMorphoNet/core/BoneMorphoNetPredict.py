import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8

# 导入包内模块
from .BoneMorphoNetModel import CellLDGnet as BoneMorphoNetModel
from ..utils import read_split_data, evaluate

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

def loadData(filename):
    dataMat = []
    labelMat = []
    fullMat=[]
    fr = open(filename)
    for line in fr.readlines():  # 逐行读取
        lineArr = line.strip().split(' ')  # 滤除行首行尾空格，以\t作为分隔符，对这行进行分解
        lineArr1 = lineArr[0].strip().split('/')
        lineArr1.append(lineArr[1])
        num = np.shape(lineArr)[0]
        dataMat.append(''.join(lineArr[0:num - 1])) # 这一行的除最后一个被添加为数据
        labelMat.append(lineArr[num - 1])  # 这一行的最后一个数据被添加为标签
        fullMat.append(lineArr1)
    return dataMat, labelMat,fullMat

def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=500):
    """

    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')
    plt.figure(figsize=(15,15))
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    # plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)


# 将MyDataSet类移到外部，使其可以被正确序列化
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, images_path, images_class, transform):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
    
    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != "RGB":
            img = img.convert("RGB")
        label = self.images_class[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    def collate_fn(self, batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


class BoneMorphoNetPredictor:
    """
    骨骼形态学网络预测器
    用于加载模型并进行预测
    """
    def __init__(self, args=None):
        """
        初始化预测器
        
        Args:
            args: 包含预测参数的对象，如果为None则使用默认参数
        """
        # 设置默认参数
        if args is None:
            class Args:
                def __init__(self):
                    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    self.num_classes = 20
                    self.img_size = 224
                    self.model_weight_path = "./resources/weights/Biomed_CLIP_vistextmix_1_3texts.pth"
                    self.pretrained_path = "./resources/ViT-B-32.pt"
                    self.class_indices_path = "./resources/class_indices.json"
                    self.test_data_path = None
                    self.test_label_path = None
                    self.batch_size = 4
                    self.save_confusion_matrix = False
                    self.confusion_matrix_path = "./resources/confusion_matrix.pdf"
            args = Args()
        
        self.args = args
        self.device = torch.device(args.device)
        print(f"使用 {self.device} 设备进行预测。")
        
        # 初始化标签
        self.labels = self._init_labels()
        
        # 初始化模型
        self._init_model()
        
        # 初始化数据转换
        self.data_transform = transforms.Compose([
            transforms.Resize(int(236)),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _init_labels(self):
        """初始化标签列表"""
        return [
            'neutrophilic stab granulocyte',
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
            'smudge cells',
            'Multiple Myeloma Cells',
            'plasmacyte',
            'other',
        ]
    
    def _init_model(self):
        """初始化模型"""
        # 加载预训练权重
        pretrained_dict = torch.load(self.args.pretrained_path, map_location="cpu").state_dict()
        embed_dim = pretrained_dict["text_projection"].shape[1]  # 文本投影维度
        context_length = 77
        vocab_size = pretrained_dict["token_embedding.weight"].shape[0]  # 词汇表大小
        transformer_width = pretrained_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = 3
        
        # 获取tokenizer
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        
        # 创建模型
        self.model = BoneMorphoNetModel(embed_dim, self.args.num_classes, tokenizer=self.tokenizer, device=self.device).to(self.device)
        
        # 加载模型权重
        if os.path.exists(self.args.model_weight_path):
            self.model.load_state_dict(torch.load(self.args.model_weight_path, map_location=self.device))
            print(f"成功加载模型权重: {self.args.model_weight_path}")
        else:
            print(f"警告: 模型权重文件 {self.args.model_weight_path} 不存在，将使用随机初始化的权重")
        
        self.model.eval()
    
    def predict_single_image(self, image_path):
        """
        预测单张图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            predicted_class: 预测的类别索引
            confidence: 预测的置信度
        """
        # 加载图像
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # 应用数据转换
        img = self.data_transform(img)
        img = img.unsqueeze(0).to(self.device)
        
        # 进行预测
        with torch.no_grad():
            output = self.model(img)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        return predicted_class, confidence
    
    def predict_batch(self, image_paths):
        """
        批量预测图像
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            predicted_classes: 预测的类别索引列表
            confidences: 预测的置信度列表
        """
        # 创建数据集
        class BatchDataSet(torch.utils.data.Dataset):
            def __init__(self, images_path, transform):
                self.images_path = images_path
                self.transform = transform
            
            def __len__(self):
                return len(self.images_path)
            
            def __getitem__(self, item):
                img = Image.open(self.images_path[item])
                if img.mode != "RGB":
                    img = img.convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                return img
        
        dataset = BatchDataSet(image_paths, self.data_transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=min([os.cpu_count(), self.args.batch_size if self.args.batch_size > 1 else 0, 8])
        )
        
        # 进行预测
        predicted_classes = []
        confidences = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                output = self.model(batch)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                batch_predicted_classes = torch.argmax(probabilities, dim=1)
                batch_confidences = torch.max(probabilities, dim=1)[0]
                
                predicted_classes.extend(batch_predicted_classes.cpu().numpy())
                confidences.extend(batch_confidences.cpu().numpy())
        
        return predicted_classes, confidences
    
    def evaluate_on_test_set(self, test_data_path=None, test_label_path=None):
        """
        在测试集上评估模型
        
        Args:
            test_data_path: 测试数据路径，如果为None则使用args中的路径
            test_label_path: 测试标签路径，如果为None则使用args中的路径
            
        Returns:
            accuracy: 准确率
            predictions: 预测结果
            labels: 真实标签
        """
        if test_data_path is None:
            test_data_path = self.args.test_data_path
        if test_label_path is None:
            test_label_path = self.args.test_label_path
        
        if test_data_path is None or test_label_path is None:
            raise ValueError("测试数据路径和测试标签路径不能为空")
        
        # 准备测试数据
        test_images_path = []
        test_images_label = []
        
        if test_label_path.endswith('.txt'):
            # 从txt文件加载数据
            dataMat, labelMat, fullMat = loadData(test_label_path)
            for i in range(len(dataMat)):
                if dataMat[i][1] == '_':
                    dataMat[i] = '0' + dataMat[i]
                file_name = os.path.join(test_data_path, dataMat[i])
                test_images_path.append(file_name)
                test_images_label.append(int(labelMat[i]))
        else:
            # 从目录加载数据
            for class_idx, class_name in enumerate(self.labels):
                class_dir = os.path.join(test_data_path, class_name)
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(class_dir, img_name)
                            test_images_path.append(img_path)
                            test_images_label.append(class_idx)
        
        # 创建测试数据集
        test_dataset = MyDataSet(
            images_path=test_images_path,
            images_class=test_images_label,
            transform=self.data_transform
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=min([os.cpu_count(), self.args.batch_size if self.args.batch_size > 1 else 0, 8]),
            collate_fn=test_dataset.collate_fn
        )
        
        # 使用evaluate函数进行预测
        t_correct, val_acc, pred, label = evaluate(
            model=self.model,
            data_loader=test_loader,
            device=self.device,
            epoch=0,
            label_name=self.labels,
            tokenizer=self.tokenizer
        )
        
        # 如果需要绘制混淆矩阵
        if self.args.save_confusion_matrix:
            draw_confusion_matrix(
                label_true=label,
                label_pred=pred,
                label_name=self.labels,
                title="BoneMorphoNet Confusion Matrix",
                pdf_save_path=self.args.confusion_matrix_path
            )
        
        return val_acc, pred, label


def main():
    """主函数，用于测试预测器"""
    import argparse
    
    parser = argparse.ArgumentParser(description='骨骼形态学网络预测')
    parser.add_argument('--device', type=str, default='cuda:0', help='预测设备')
    parser.add_argument('--num-classes', type=int, default=20, help='类别数量')
    parser.add_argument('--img-size', type=int, default=224, help='图像大小')
    parser.add_argument('--model-weight-path', type=str, default='./resources/weights/Biomed_CLIP_vistextmix_1_3texts.pth', help='模型权重路径')
    parser.add_argument('--pretrained-path', type=str, default='./resources/ViT-B-32.pt', help='预训练模型路径')
    parser.add_argument('--class-indices-path', type=str, default='./resources/class_indices.json', help='类别索引文件路径')
    parser.add_argument('--test-data-path', type=str, default=None, help='测试数据路径')
    parser.add_argument('--test-label-path', type=str, default=None, help='测试标签路径')
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--save-confusion-matrix', action='store_true', help='是否保存混淆矩阵')
    parser.add_argument('--confusion-matrix-path', type=str, default='./resources/confusion_matrix.pdf', help='混淆矩阵保存路径')
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = BoneMorphoNetPredictor(args)
    
    # 如果有测试数据，则在测试集上评估
    if args.test_data_path is not None:
        accuracy, predictions, labels = predictor.evaluate_on_test_set()
        print(f"测试集准确率: {accuracy:.4f}")
        
        # 打印分类报告
        from sklearn.metrics import classification_report
        print("\n分类报告:")
        print(classification_report(labels, predictions, target_names=predictor.labels))
    else:
        print("未提供测试数据路径，请使用predict_single_image或predict_batch方法进行预测")


if __name__ == '__main__':
    main()
