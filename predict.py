import os
import argparse
import torch
import time
import numpy as np

# 导入Mypackage包
from Mypackage import BoneMorphoNetPredictor

def main():
    """
    主函数，用于预测骨骼形态学网络模型
    """
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='骨骼形态学网络预测')
    parser.add_argument('--device', type=str, default='cuda:0', help='预测设备')
    parser.add_argument('--num-classes', type=int, default=20, help='类别数量')
    parser.add_argument('--img-size', type=int, default=224, help='图像大小')
    parser.add_argument('--model-weight-path', type=str, default='./Mypackage/resources/weights/Biomed_CLIP_vistextmix.pth', help='模型权重路径')
    parser.add_argument('--pretrained-path', type=str, default='./Mypackage/resources/ViT-B-32.pt', help='预训练模型路径')
    parser.add_argument('--class-indices-path', type=str, default='./Mypackage/resources/class_indices.json', help='类别索引文件路径')
    parser.add_argument('--test-data-path', type=str, default="F:/WorkSpaces/ConvNext_RE/CellLDGnet/data/test/", help='测试数据路径')
    parser.add_argument('--test-label-path', type=str, default="F:/WorkSpaces/ConvNext_RE/CellLDGnet/data/meta/test.txt", help='测试标签路径')
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--save-confusion-matrix', action='store_true', help='是否保存混淆矩阵')
    parser.add_argument('--confusion-matrix-path', type=str, default='./Mypackage/resources/confusion_matrix.pdf', help='混淆矩阵保存路径')
    parser.add_argument('--single-image', type=str, default=None, help='单张图像路径')
    parser.add_argument('--image-dir', type=str, default=None, help='图像目录路径')
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = BoneMorphoNetPredictor(args)
    
    # 根据参数选择预测模式
    if args.single_image is not None:
        # 预测单张图像
        predicted_class, confidence = predictor.predict_single_image(args.single_image)
        print(f"图像: {args.single_image}")
        print(f"预测类别: {predictor.labels[predicted_class]}")
        print(f"置信度: {confidence:.4f}")
    elif args.image_dir is not None:
        # 预测目录中的所有图像
        image_paths = []
        for img_name in os.listdir(args.image_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(args.image_dir, img_name)
                image_paths.append(img_path)
        
        if not image_paths:
            print(f"目录 {args.image_dir} 中没有找到图像文件")
            return
        
        predicted_classes, confidences = predictor.predict_batch(image_paths)
        
        print(f"共预测 {len(image_paths)} 张图像:")
        for i, (img_path, pred_class, conf) in enumerate(zip(image_paths, predicted_classes, confidences)):
            print(f"{i+1}. 图像: {os.path.basename(img_path)}")
            print(f"   预测类别: {predictor.labels[pred_class]}")
            print(f"   置信度: {conf:.4f}")
    elif args.test_data_path is not None:
        # 在测试集上评估
        accuracy, predictions, labels = predictor.evaluate_on_test_set()
        print(f"测试集准确率: {accuracy:.4f}")
        print(f"标签数量: {len(labels)}")
        print(f"预测数量: {len(predictions)}")
        true_label_list = []
        pred_label_list = []

        for i in range(len(predictions)):
            pred = predictions[i]
            label = labels[i]
            for j in range(len(pred)):
                true_label_list.append(label[j])
                pred_label_list.append(pred[j])
        # 打印分类报告
        from sklearn.metrics import classification_report
        print("\n分类报告:")
        print(classification_report(true_label_list, pred_label_list, target_names=predictor.labels,digits=4))
    else:
        print("请提供单张图像路径、图像目录路径或测试数据路径")


if __name__ == '__main__':
    main()