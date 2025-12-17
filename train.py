# 必要なライブラリのインポート
import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim

# ==============================================================================
# 🌟 実験モードの定義 🌟
# MODE = 'LAYER4' : layer4 と fc を学習 (元のコードの動作)
# MODE = 'LAYER3' : layer3, layer4, fc を学習
# ==============================================================================
MODE = 'LAYER4' 
# MODE = 'LAYER3' 

print(f"==========================================")
print(f"  実験モード: {MODE} - 学習対象層の解凍")
print(f"==========================================")

# ==============================================================================
# ├─ transform 定義
# ==============================================================================

# 学習データ用の前処理を定義
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 評価データ（テストデータ）用の前処理を定義
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================================================================
# ├─ データ読み込み
# ==============================================================================

# CIFAR-10データセットの読み込み
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)

# DataLoaderの設定
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 使用デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# ├─ モデル定義（ResNet18）
# ==============================================================================

# 既存の学習済みResNet-18モデルをロード
model = models.resnet18(
    weights=models.ResNet18_Weights.IMAGENET1K_V1
)

# ==============================================================================
# ├─ freeze / unfreeze (層の凍結/凍結解除)
# ==============================================================================

# --- 1. 初期設定: 全てのパラメータを一旦凍結 (requires_grad = False) ---
for param in model.parameters():
    param.requires_grad = False

# --- 2. 全結合層 (fc) の再定義と凍結解除 ---
# 最終層をCIFAR-10のクラス数（10）へ変更。この層は常に学習対象とする。
model.fc = nn.Linear(model.fc.in_features, 10)

# --- 3. 実験モードに基づく中間層の凍結解除 (ファインチューニングの対象) ---
if MODE == 'LAYER4':
    # MODE: LAYER4 の場合 (layer4 のみ解凍)
    # layer4 (最終ブロック) のパラメータを凍結解除
    for param in model.layer4.parameters():
        param.requires_grad = True
    print("  -> layer4 と fc を学習対象として解凍します。")

elif MODE == 'LAYER3':
    # MODE: LAYER3 の場合 (layer3, layer4 を解凍)
    # layer4 (最終ブロック) のパラメータを凍結解除
    for param in model.layer4.parameters():
        param.requires_grad = True
    # layer3 (その前のブロック) のパラメータを凍結解除
    for param in model.layer3.parameters():
        param.requires_grad = True
    print("  -> layer3, layer4, および fc を学習対象として解凍します。")

# モデルをデバイスへ移動
model = model.to(device)

# ==============================================================================
# ├─ optimizer 設定
# ==============================================================================

# 損失関数としてCrossEntropyLossを使用
criterion = nn.CrossEntropyLoss()

# OptimizerとしてAdamを使用し、層ごとに異なる学習率を設定
# MODEによってOptimizerに渡すパラメータリストを変更
if MODE == 'LAYER4':
    # layer4 と fc のみを学習対象とする
    optimizer = optim.Adam([
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(),     "lr": 1e-3},
    ])
elif MODE == 'LAYER3':
    # layer3, layer4, fc を学習対象とする
    # layer3とlayer4には同じ学習率 (1e-4) を設定
    optimizer = optim.Adam([
        {"params": model.layer3.parameters(), "lr": 1e-4}, # layer3を追加
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(),     "lr": 1e-3},
    ])

# ==============================================================================
# └─ 評価 (evaluate関数定義)
# ==============================================================================

def evaluate(model, loader):
    """
    モデルの評価（精度計算）を行う関数
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

# ==============================================================================
# ├─ 学習ループ
# ==============================================================================

# エポック数の設定
epochs = 50

print(f"Start training on {device} for {epochs} epochs...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    # 訓練データローダからバッチを取得
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 順伝播、損失計算
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 勾配のリセット、逆伝播、パラメータの更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 1エポック終了後の評価
    test_acc = evaluate(model, test_loader)

    # 結果の表示
    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {running_loss/len(train_loader):.4f} "
        f"Test Acc: {test_acc*100:.2f}%"
    )

print("Training finished.")
final_acc = evaluate(model, test_loader)
print(f"Final Test Accuracy ({MODE}): {final_acc*100:.2f}%")