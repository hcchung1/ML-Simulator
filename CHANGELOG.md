# Changelog

## v1.11 — Canvas Zoom & Pan

### New Features

- **縮放功能** — Ctrl+滾輪縮放畫布 (10%–500%)，工具欄顯示目前縮放百分比，可直接輸入數字調整
- **平移功能** — 滾輪上下滝動，Alt+滾輪左右滝動
- **工具欄縮放控制** — 「−」/「+」按鈕 + 百分比輸入框

---

## v1.1 — CNN / ResNet / Transformer + Project Files

### New Features

- **CNN 支援** — 新增 `Conv2D`、`MaxPool2D`、`BatchNorm2D`、`Flatten`、`GlobalAvgPool2D` 五種 Op，可在 canvas 拖曳組合任意 CNN
- **ResNet 支援** — 新增 `ResNetBuilder`，自動建構含 residual skip connection 的 ResNet 圖（1×1 Conv 下採樣 + Add 殘差連接）
- **Transformer 支援** — 新增 `MultiHeadAttention`、`LayerNorm`、`PositionalEncoding`、`FlatLinear3D`、`MeanPool1D` Op + `TransformerBuilder`（encoder stack: PosEnc → [MHA → Add → LN → FFN → Add → LN] × N → Linear）
- **專案檔案系統** — 支援 `.nsim` 專案檔（JSON 格式）。使用者可以儲存模型配置與 canvas 佈局，下次直接「開啟」即可還原；也方便分享給其他人
- **Inspector 擴充** — Inspector 面板支援 Conv2D（InCh/OutCh/K/S/P）、MaxPool2D（K/S）、BatchNorm2D（Channels）、MultiHeadAttention（d_model/Heads）、LayerNorm（Dim）、PositionalEncoding（d_model/MaxSeq）配置編輯

### New Ops (Palette)

| Category | Ops |
|---|---|
| CNN | Conv2D, MaxPool2D, BatchNorm2D, Flatten, GlobalAvgPool2D |
| Transformer | MultiHeadAttention, LayerNorm, PositionalEncoding, MeanPool1D |

### Tensor Engine Enhancements

- 3D/4D indexer (`Get3D`/`Set3D`/`Get4D`/`Set4D`)
- `Conv2D`, `MaxPool2D`, `GlobalAvgPool2D`, `BatchNorm2D` (inference) static methods
- `BatchedMatMul` (3D batched matrix multiply)
- `LayerNorm` (last-axis normalization)
- `KaimingUniform` init (for ReLU/Conv)
- `Softmax` 支援 3D tensor
- `Add` 支援 3D + 1D broadcasting

### Builders

- `CnnBuilder` — Conv → BN → ReLU → Pool → … → Flatten → Linear → Softmax
- `ResNetBuilder` — Stem → [ResBlock(Conv→BN→ReLU→Conv→BN + Skip→Add→ReLU)] × N → GAP → Linear → Softmax
- `TransformerBuilder` — PosEnc → [MHA→Add→LN→FFN→Add→LN] × N → MeanPool → Linear → Softmax

### Tests

- 新增 29 個測試 (CNN 8 + ResNet 4 + Transformer 13 + ProjectFile 2 + existing 12 = 共 41)

---

## v1.0 — Initial Release

### Features

- **Drag & Drop Graph Editor** — 從左側 Op Palette 拖曳積木到 canvas 建構神經網路
- **支援的 Operations**：Linear, ReLU, Sigmoid, Tanh, Softmax, Add
- **視覺化連線** — 點擊 output port（紅）→ input port（藍）建立資料流連線
- **Forward Pass 逐步執行** — Step Forward / Step Backward / Run to End，觀察每一層的運算過程
- **即時 Tensor 顯示** — 每個積木上顯示 output tensor 摘要
- **Input 積木直接輸入** — 在 Input 積木中直接編輯 input vector，無需切換面板
- **Output 積木顯示結果** — 執行結束自動停留在 Output 積木，顯示最終輸出
- **Inspector 面板** — 選取任意積木查看完整 inputs/outputs/parameters 資訊
- **Linear 層設定** — 可在 Inspector 中調整 in_features / out_features
- **Random Seed 控制** — 固定 seed 確保權重可重現
- **Self-contained 發布** — 單一 exe，無需安裝 .NET runtime

### Tech Stack

- WPF (.NET 9) + CommunityToolkit.Mvvm
- 自製 Tensor / Graph / Executor / Trace 計算引擎
