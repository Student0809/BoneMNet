import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
import os
import argparse
import torch
from urllib.request import urlopen
from PIL import Image
from torchvision import transforms
import time
import torch.optim as optim

# 导入包内模块
from ..utils import read_split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate, MyDataSet
from ..data_augmentation import get_advanced_transforms, get_cutmix_transform
from .BoneMorphoNetModel import CellLDGnet

class BoneMorphoNetTrainer:
    def __init__(self, args):
        """
        初始化训练器
        
        Args:
            args: 包含训练参数的对象，包括num_classes, epochs, batch_size, lr, wd, data_path, weights, freeze_layers, device等
        """
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"使用 {self.device} 设备进行训练。")
        
        # 创建必要的目录
        self.checkpoint_dir = args.checkpoint_dir if hasattr(args, 'checkpoint_dir') else './resources/checkpoint/'
        self.weights_dir = args.weights_dir if hasattr(args, 'weights_dir') else './resources/weights'
        self.pretrained_path = args.pretrained_path if hasattr(args, 'pretrained_path') else './resources/ViT-B-32.pt'
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
            
        # 初始化标签
        self.labels = self._init_labels()
        
        # 初始化数据加载器
        self._init_data_loaders()
        
        # 初始化模型
        self._init_model()
        
        # 初始化优化器和学习率调度器
        self._init_optimizer_and_scheduler()
        
        # 初始化训练状态
        self.best_acc = 0.0
        self.start_epoch = 0
        
    def _init_labels(self):
        """初始化标签列表"""
        return [
            ["Neutrophilic stab granulocyte's cell shape is round-like.",
             "Neutrophilic stab granulocyte's nuclear shape is rod-shaped, S-shaped, or U-shaped.",
             "Neutrophilic stab granulocyte's cytoplasm is light blue.",
             ],

            ["Polychromatic normoblast's cell shape is round.",
             "Polychromatic normoblast's nuclear shape is round.",
             "Polychromatic normoblast's cytoplasm is blue-gray, gray, or gray-red.",
             ],

            ["Neutrophilic myelocyte's cell shape is round-like.",
             "Neutrophilic myelocyte's nuclear shape is oval, semicircular, or slightly indented.",
             "Neutrophilic myelocyte's cytoplasm is blue or light blue.",
             ],

            ["Neutrophilic segmented granulocyte's cell shape is round.",
             "Neutrophilic segmented granulocyte's nuclear shape is segmented, with 2 to 5 lobes.",
             "Neutrophilic segmented granulocyte's cytoplasm is light blue.",
             ],

            ["Lymphoblast's cell shape is regular or irregular.",
             "Lymphoblast's nuclear shape is round or irregular.",
             "Lymphoblast's cytoplasm is blue or dark blue.",
             ],

            ["Neutrophilic metamyelocyte's cell shape is round-like.",
             "Neutrophilic metamyelocyte's nuclear shape is kidney-shaped or crescent-shaped.",
             "Neutrophilic metamyelocyte's cytoplasm is light blue.",
             ],

            ["Myeloblast's cell shape is round-like.",
             "Myeloblast's nuclear shape is round.",
             "Myeloblast's cytoplasm is blue or dark blue.",
             ],

            ["Orthochromatic normoblast's cell shape is round or oval.",
             "Orthochromatic normoblast's nuclear shape is round.",
             "Orthochromatic normoblast's cytoplasm is light red or gray-red.",
            ],

            ["Prelymphocyte's cell shape is regular or irregular.",
             "Prelymphocyte's nuclear shape is roughly round or irregular.",
             "Prelymphocyte's cytoplasm is blue or dark blue.",
             ],

            ["Abnormal promyelocyte's cell shape is variable.",
             "Abnormal promyelocyte's nuclear shape is irregular, folded, twisted, or segmented.",
             "Abnormal promyelocyte's cytoplasm contains abundant purple-red granules.",
             ],

            ["Monocyte's cell shape is round-like or irregular.",
             "Monocyte's nuclear shape is irregular, folded, twisted, horseshoe-shaped, or S-shaped.",
             "Monocyte's cytoplasm is light gray-blue or light blue.",
            ],

            ["Early normoblast's cell shape is round or oval.",
             "Early normoblast's nuclear shape is round.",
             "Early normoblast's cytoplasm is dark blue.",
             ],

            ["Monoblast's cell shape is round-like or irregular.",
             "Monoblast's nuclear shape is round or irregular.",
             "Monoblast's cytoplasm is gray-blue or blue.",
             ],

            ["Promyelocyte's cell shape is round or oval.",
             "Promyelocyte's nuclear shape is round or oval.",
             "Promyelocyte's cytoplasm is blue or dark blue, containing purple-red granules.",
             ],

            ["Eosinophilic segmented granulocyte's cell shape is round.",
             "Eosinophilic segmented granulocyte's nuclear shape is segmented.",
             "Eosinophilic segmented granulocyte's cytoplasm is orange-red, dark yellow, or brown.",
             ],

            ["Eosinophilic myelocyte's cell shape is round-like.",
             "Eosinophilic myelocyte's nuclear shape is round or oval.",
             "Eosinophilic myelocyte's cytoplasm is orange-red, dark yellow, or brown.",
             ],

            ["Multiple myeloma cells' cell shape is irregular.",
             "Multiple myeloma cells' nuclear shape is irregular, sometimes with multiple nuclei.",
             "Multiple myeloma cells' cytoplasm contains multicolored granules.",
             "Multiple myeloma cells have multicolored granules in their cytoplasm."],

            ["Smudge cells' cell shape is irregular.",
             "Smudge cells' nuclear shape is irregular, often unclear due to fragmentation.",
             "Smudge cells' cytoplasm is enlarged and incomplete.",
             "Smudge cell has no granules in its cytoplasm (but often appears as a naked nucleus, with incomplete cytoplasm)."],

            ["Plasmacyte's cell shape is oval.",
             "Plasmacyte's nuclear shape is round or eccentrically placed, sometimes with two or more nuclei.",
             "Plasmacyte's cytoplasm is dark blue, occasionally red.",
             "Plasmacyte has few purple-red granules in its cytoplasm."],

            ["Other's cell shape is unclear.",
             "Other's nuclear shape is unclear.",
             "Other's cytoplasm is unclear.",
             "Other's granules are unclear."]
        ]
    
    def _init_data_loaders(self):
        """初始化数据加载器"""
        # 读取数据集
        train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(self.args.data_path)
        
        # 设置图像大小和数据增强
        img_size = 224
        
        # 使用data_augmentation模块提供的便捷函数获取数据变换
        data_transform = get_advanced_transforms(img_size=img_size, auto_augment_config='rand-m9-n2-mstd0.5')
        
        # 实例化训练数据集
        train_dataset = MyDataSet(images_path=train_images_path,
                                  images_class=train_images_label,
                                  transform=data_transform["train"])

        # 实例化验证数据集
        val_dataset = MyDataSet(images_path=val_images_path,
                                images_class=val_images_label,
                                transform=data_transform["val"])

        # 设置数据加载器
        batch_size = self.args.batch_size
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        print('使用 {} 个数据加载器工作进程'.format(nw))

        # 获取CutMix变换
        cutmix_transform = get_cutmix_transform(
            mixup_alpha=1.0,
            cutmix_alpha=0.0,
            prob=1.0,
            switch_prob=0.5,
            mode='batch',
            correct_lam=True,
            label_smoothing=0.1,
            num_classes=self.args.num_classes
        )

        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=nw,
                                                   collate_fn=train_dataset.collate_fn_cutmix)

        self.val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=nw,
                                                 collate_fn=val_dataset.collate_fn)
    
    def _init_model(self):
        """初始化模型"""
        # 加载预训练权重
        pretrained_dict = torch.load(self.pretrained_path, map_location="cpu").state_dict()
        embed_dim = pretrained_dict["text_projection"].shape[1]  # 文本投影维度
        context_length = pretrained_dict["positional_embedding"].shape[0]
        vocab_size = pretrained_dict["token_embedding.weight"].shape[0]  # 词汇表大小
        transformer_width = pretrained_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = 3
        print("embed_dim: {}, context_length: {}, vocab_size: {},transformer_width: {}, transformer_heads: {}, transformer_layers:{}"
              .format(embed_dim, context_length, vocab_size,transformer_width, transformer_heads, transformer_layers))
        
        # 获取tokenizer
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        
        # 创建模型
        self.model = CellLDGnet(embed_dim, 20, tokenizer=self.tokenizer, device=self.device).to(self.device)
        
        # 加载预训练权重（如果有）
        if self.args.weights != "":
            assert os.path.exists(self.args.weights), "权重文件: '{}' 不存在。".format(self.args.weights)
            weights_dict = torch.load(self.args.weights, map_location=self.device)
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
                elif "visual" not in k:
                    del weights_dict[k]

            print(self.model.load_state_dict(weights_dict, strict=False))
        
        # 冻结层（如果需要）
        if self.args.freeze_layers:
            for name, para in self.model.named_parameters():
                # 除head外，其他权重全部冻结
                if "head" not in name:
                    para.requires_grad_(False)
                else:
                    print("训练 {}".format(name))
    
    def _init_optimizer_and_scheduler(self):
        """初始化优化器和学习率调度器"""
        # 获取参数组
        pg = get_params_groups(self.model, weight_decay=self.args.wd)
        
        # 创建优化器
        self.optimizer = optim.AdamW(pg, lr=self.args.lr, weight_decay=self.args.wd)
        
        # 创建学习率调度器
        self.lr_scheduler = create_lr_scheduler(self.optimizer, len(self.train_loader), self.args.epochs,
                                               warmup=True, warmup_epochs=3)
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点文件路径
        """
        print('-----------------------------')
        checkpoint = torch.load(checkpoint_path)  # 加载断点
        self.model.load_state_dict(checkpoint['model'])  # 加载模型可学习参数
        self.optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.start_epoch = checkpoint['epoch'] + 1  # 设置开始的epoch
        print('加载 epoch {} 成功！'.format(self.start_epoch))
        print('-----------------------------')
    
    def train(self):
        """训练模型"""
        print('无保存模型，将从头开始训练！')
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # 训练一个epoch
            start_time = time.time()
            self.model, train_acc = train_one_epoch(model=self.model,
                                                 optimizer=self.optimizer,
                                                 data_loader=self.train_loader,
                                                label_name=self.labels,
                                                device=self.device,
                                                epoch=epoch,
                                                lr_scheduler=self.lr_scheduler,
                                                tokenizer=self.tokenizer
                                            )
            end_time = time.time()
            exc_time = end_time - start_time
            print(f"训练耗时 = {exc_time}")
            
            # 验证
            t_correct, val_acc, pred, label = evaluate(model=self.model,
                                            data_loader=self.val_loader,
                                            device=self.device,
                                            label_name=self.labels,
                                            epoch=epoch,
                                        tokenizer=self.tokenizer)
            
            # 保存模型参数
            if (epoch + 1) % 60 == 0:
                print('epoch:', epoch + 1)
                checkpoint = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'lr_scheduler': self.lr_scheduler.state_dict()
                }
                torch.save(checkpoint, self.checkpoint_dir + 'ckpt_best_RANDOM_300e_%s.pth' % (str(epoch + 1)))
                print('已保存所有参数！\n')
            
            # 保存最佳模型
            if self.best_acc < val_acc:
                torch.save(self.model.state_dict(), self.weights_dir + '/Biomed_CLIP_vistextmix_1_3texts.pth')
                self.best_acc = val_acc


def main(args):
    """主函数"""
    trainer = BoneMorphoNetTrainer(args)
    
    # 判断是否需要恢复训练
    RESUME = False  # 控制是否是恢复训练。False：初次训练;True：继续训练
    if RESUME:
        trainer.load_checkpoint(args.checkpoint_dir + 'ckpt_best_135.pth')
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--wd', type=float, default=5e-2)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str,
                        default="/home/liuzhengqing/unzip_202408081355_CellLDGnet/CellLDGnet/data/photo")
    
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./resources/weights/Biomed_CLIP_all_alpha_1=0.7_4w.pth',
                        help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    
    # 添加目录相关参数
    parser.add_argument('--checkpoint-dir', type=str, default='./resources/checkpoint/',
                        help='checkpoint directory path')
    parser.add_argument('--weights-dir', type=str, default='./resources/weights',
                        help='weights directory path')
    parser.add_argument('--pretrained-path', type=str, default='./resources/ViT-B-32.pt',
                        help='pretrained model path')

    opt = parser.parse_args()

    main(opt)