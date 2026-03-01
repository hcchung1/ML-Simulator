# Changelog

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
