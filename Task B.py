import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class StrictTwoLayerNet(nn.Module):
    """嚴格的兩層有效網絡，整個網絡僅包含兩個卷積層作為有效層"""
    def __init__(self, num_classes=1000):
        super(StrictTwoLayerNet, self).__init__()
        
        # 預處理 - 不計入有效層
        self.pre_process = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 這是預處理，不計入有效層
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 第一有效層 - 單一卷積操作
        self.effective_layer1 = nn.Conv2d(64, 384, kernel_size=5, stride=1, padding=2, bias=False)
        
        # 第一層後處理 - 不計入有效層
        self.mid_process = nn.Sequential(
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第二有效層 - 單一卷積操作
        self.effective_layer2 = nn.Conv2d(384, 768, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 第二層後處理 - 不計入有效層
        self.post_process = nn.Sequential(
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 分類頭 - 不計入有效層
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, x):
        # 預處理
        x = self.pre_process(x)
        
        # 第一有效層
        x = self.effective_layer1(x)
        x = self.mid_process(x)
        
        # 第二有效層
        x = self.effective_layer2(x)
        x = self.post_process(x)
        
        # 特徵向量化與分類
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

# 單層模型 - 用於消融研究
class SingleLayerNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(SingleLayerNet, self).__init__()
        
        # 預處理
        self.pre_process = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 只有一個有效層
        self.effective_layer = nn.Conv2d(64, 768, kernel_size=5, stride=1, padding=2, bias=False)
        
        # 後處理
        self.post_process = nn.Sequential(
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 分類頭
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, x):
        x = self.pre_process(x)
        x = self.effective_layer(x)
        x = self.post_process(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 小卷積核模型 - 用於消融研究
class SmallKernelNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(SmallKernelNet, self).__init__()
        
        # 預處理
        self.pre_process = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 第一有效層 - 使用較小的卷積核
        self.effective_layer1 = nn.Conv2d(64, 384, kernel_size=3, stride=1, padding=3, bias=False)
        
        # 第一層後處理
        self.mid_process = nn.Sequential(
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第二有效層 - 使用較小的卷積核
        self.effective_layer2 = nn.Conv2d(384, 768, kernel_size=1, stride=1, padding=2, bias=False)

        # 第二層後處理
        self.post_process = nn.Sequential(
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 分類頭
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, x):
        x = self.pre_process(x)
        x = self.effective_layer1(x)
        x = self.mid_process(x)
        x = self.effective_layer2(x)
        x = self.post_process(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ImageNetMiniDataset(Dataset):
    """ImageNet-mini數據集加載類"""
    
    def __init__(self, txt_file, root_dir, transform=None):
        """
        參數:
            txt_file (string): 包含圖像路徑和標籤的文本文件
            root_dir (string): 圖像目錄的根路徑
            transform (callable, optional): 可選的圖像轉換
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.class_count = 0
        
        # 讀取文件中的圖像路徑和標籤
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    # 最後一個元素是標籤，前面的所有部分是路徑
                    label = int(parts[-1])
                    img_path = " ".join(parts[:-1])
                    self.samples.append((img_path, label))
                    
                    # 更新類別計數
                    if label not in self.class_to_idx:
                        self.class_to_idx[label] = self.class_count
                        self.class_count += 1
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 如果提供了root_dir，則將其與路徑結合
        if self.root_dir:
            img_path = os.path.join(self.root_dir, img_path)
            
        # 打開圖像
        image = Image.open(img_path).convert('RGB')
        
        # 應用轉換
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 訓練函數 - 修改以返回訓練損失
def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 99:
            print(f'Epoch: {epoch}, Batch: {batch_idx+1}, Loss: {running_loss/100:.3f}, '
                  f'Acc: {100.*correct/total:.3f}%')
            running_loss = 0.0
    
    avg_loss = total_loss / len(train_loader)
    acc = 100.*correct/total
    
    return acc, avg_loss

# 評估函數
def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    acc = 100.*correct/total
    avg_loss = test_loss/len(test_loader)
    
    print(f'Test Loss: {avg_loss:.3f} | Acc: {acc:.3f}%')
    
    return acc, avg_loss

# 計算模型參數數量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 基準模型評估函數
def evaluate_baseline(device, train_loader, test_loader, num_classes):
    # 載入ResNet34基準模型
    print("建立並訓練ResNet34基準模型...")
    resnet34 = models.resnet34(pretrained=False)  # 從頭開始訓練，不使用預訓練權重
    
    # 調整最後一層以匹配類別數
    resnet34.fc = nn.Linear(resnet34.fc.in_features, num_classes)
    resnet34 = resnet34.to(device)
    
    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet34.parameters(), lr=0.001)
    
    # 紀錄訓練歷史
    train_acc_history = []
    train_loss_history = []
    test_acc_history = []
    test_loss_history = []
    
    # 訓練10個epoch (簡短訓練以作為基準比較)
    for epoch in range(10):
        train_acc, train_loss = train(resnet34, train_loader, optimizer, criterion, device, epoch)
        test_acc, test_loss = evaluate(resnet34, test_loader, criterion, device)
        
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss)
        test_acc_history.append(test_acc)
        test_loss_history.append(test_loss)
        
        print(f"ResNet34基準 - Epoch {epoch+1}/10 完成")
    
    # 計算參數數量
    resnet_params = count_parameters(resnet34)
    print(f"ResNet34有 {resnet_params:,} 個參數")
    
    # Plot the training results of the baseline model
    plt.figure(figsize=(20, 8))
    
    # 繪製損失圖 (左圖)
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, color='blue', label='Train Loss')
    plt.plot(test_loss_history, color='orange', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.grid(False)
    plt.legend()
    
    # 繪製準確率圖 (右圖)
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, color='blue', label='Train Acc')
    plt.plot(test_acc_history, color='orange', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Epoch')
    plt.grid(False)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('resnet34_baseline_results.png')        
    
    return test_acc_history[-1], resnet_params

# 消融研究函數
def ablation_study(device, train_loader, test_loader, num_classes):
    results = {}
    
    # 基本的兩層網絡 - 完整模型
    print("測試完整的兩層網絡...")
    model_full = StrictTwoLayerNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_full.parameters(), lr=0.001)
    
    # 只訓練幾個epoch用於消融研究
    for epoch in range(3):
        train(model_full, train_loader, optimizer, criterion, device, epoch)
    
    acc_full, _ = evaluate(model_full, test_loader, criterion, device)
    results['Full Two-Layer Model'] = acc_full
    
    # 測試單層模型
    print("測試單層模型...")
    model_single = SingleLayerNet(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model_single.parameters(), lr=0.001)
    
    for epoch in range(3):
        train(model_single, train_loader, optimizer, criterion, device, epoch)
    
    acc_single, _ = evaluate(model_single, test_loader, criterion, device)
    results['Single-Layer Model'] = acc_single
    
    # 測試不同卷積核大小的影響
    print("測試較小卷積核大小的模型...")
    model_small_kernel = SmallKernelNet(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model_small_kernel.parameters(), lr=0.001)
    
    for epoch in range(3):
        train(model_small_kernel, train_loader, optimizer, criterion, device, epoch)
    
    acc_small_kernel, _ = evaluate(model_small_kernel, test_loader, criterion, device)
    results['Smaller Kernel Model'] = acc_small_kernel
    
    return results

# 主程序
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    # 數據預處理
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 載入ImageNet-mini數據集
    data_root = '.'  # 根目錄路徑
    train_dataset = ImageNetMiniDataset(
        txt_file=os.path.join(data_root, 'train.txt'),
        root_dir=data_root,
        transform=transform_train
    )
    
    val_dataset = ImageNetMiniDataset(
        txt_file=os.path.join(data_root, 'val.txt'),
        root_dir=data_root,
        transform=transform_test
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
    
    # 獲取類別數量
    num_classes = train_dataset.class_count
    print(f"數據集有 {num_classes} 個類別")
    
    # 首先評估基準模型 ResNet34
    print("\n====== 評估基準模型: ResNet34 ======")
    resnet_acc, resnet_params = evaluate_baseline(device, train_loader, test_loader, num_classes)
    print(f"ResNet34基準性能: 測試準確率 {resnet_acc:.2f}%")
    
    # 創建我們設計的兩層網絡模型實例
    model = StrictTwoLayerNet(num_classes=num_classes).to(device)
    model_params = count_parameters(model)
    print(f"\n====== 評估我們的兩層網絡 ======")
    print(f"創建了一個有 {model_params:,} 個參數的模型")
    
    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    
    # 訓練和評估
    train_acc_history = []
    train_loss_history = []
    test_acc_history = []
    test_loss_history = []
    
    start_time = time.time()
    
    epochs = 20  # 訓練20個epoch
    for epoch in range(epochs):
        train_acc, train_loss = train(model, train_loader, optimizer, criterion, device, epoch)
        test_acc, test_loss = evaluate(model, test_loader, criterion, device)
        
        scheduler.step(test_acc)
        
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss)
        test_acc_history.append(test_acc)
        test_loss_history.append(test_loss)
        
        print(f"Epoch {epoch+1}/{epochs} 完成")
    
    training_time = time.time() - start_time
    print(f"訓練完成！總用時: {training_time/60:.2f} 分鐘")
    
    # Plot training results - 類似所需的圖表格式 (左右兩個圖表)
    plt.figure(figsize=(20, 8))
    
    # 繪製損失圖 (左圖)
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, color='blue', label='Train Loss')
    plt.plot(test_loss_history, color='orange', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.grid(False)
    plt.legend()
    
    # 繪製準確率圖 (右圖)
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, color='blue', label='Train Acc')
    plt.plot(test_acc_history, color='orange', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Epoch')
    plt.grid(False)
    plt.legend()
        
    plt.tight_layout()
    plt.savefig('training_results.png')
    
    # 進行消融研究
    print("\n開始進行消融研究...")
    ablation_results = ablation_study(device, train_loader, test_loader, num_classes)
    
    # 繪製消融研究結果
    plt.figure(figsize=(10, 6))
    models = list(ablation_results.keys())
    accs = list(ablation_results.values())
    
    plt.bar(models, accs)
    plt.xlabel('Model Variants')
    plt.ylabel('Accuracy (%)')
    plt.title('Ablation Study Results')
    plt.savefig('ablation_results.png')
        
    print("\n消融研究結果:")
    for model_name, acc in ablation_results.items():
        print(f"{model_name}: {acc:.2f}%")
    
    # 與ResNet34基準模型比較
    print("\n====== 最終性能比較 ======")
    print(f"ResNet34基準模型: {resnet_acc:.2f}%")
    print(f"我們的兩層網絡: {test_acc_history[-1]:.2f}%")
    print(f"性能差距: {resnet_acc - test_acc_history[-1]:.2f}%")
    print(f"性能比例: {(test_acc_history[-1]/resnet_acc)*100:.2f}%")
    print(f"參數數量比較: ResNet34 ({resnet_params:,}) vs 兩層網絡 ({model_params:,})")
    
    # 繪製與基準模型的比較圖
    plt.figure(figsize=(10, 6))
    models_compare = ['ResNet34', 'Two-Layer Network']
    accuracies = [resnet_acc, test_acc_history[-1]]
    
    plt.bar(models_compare, accuracies)
    plt.axhline(y=resnet_acc*0.9, color='r', linestyle='-', label='90% ResNet34 Performance')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Performance Comparison with Baseline Model')
    plt.savefig('baseline_comparison.png')

# 調用主程序
if __name__ == "__main__":
    main()