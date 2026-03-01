<div align="center">

# NeuralSim

**A visual, drag-and-drop neural network forward-pass simulator**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![.NET 9](https://img.shields.io/badge/.NET-9.0-purple.svg)](https://dotnet.microsoft.com/)
[![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey.svg)]()

</div>

---

## Overview

NeuralSim is a desktop application that lets you **visually build neural network graphs** by dragging operation blocks onto a canvas, wiring them together, and stepping through the forward pass one operation at a time. It is designed as an educational tool for understanding how data flows through layers like Linear, ReLU, Sigmoid, and more.

No Python. No GPU. Just a single `.exe` — open it and start experimenting.

## Features

- **Drag & Drop Graph Editor** — Build networks by dragging ops from the palette onto the canvas
- **Visual Wiring** — Connect output ports to input ports to define data flow
- **Step-by-Step Execution** — Step forward/backward through the forward pass and watch tensors propagate
- **Live Tensor Inspection** — Click any block to see full input/output tensors and parameter matrices
- **Inline Input Editing** — Type your input vector directly into the Input block
- **Output Display** — Final result is shown on the Output block when execution completes
- **Supported Operations** — Linear, ReLU, Sigmoid, Tanh, Softmax, Add
- **Reproducible** — Random seed control for deterministic weight initialization
- **Self-Contained** — Ships as a single executable, no runtime installation required

## Screenshot

<!-- TODO: Add a screenshot of the app -->
<!-- ![NeuralSim Screenshot](docs/screenshot.png) -->

## Getting Started

### Download

Grab the latest release from the [Releases](https://github.com/hcchung1/ML-Simulator/releases) page — just download, unzip, and run.

### Build from Source

**Prerequisites:** [.NET 9 SDK](https://dotnet.microsoft.com/download/dotnet/9.0)

```bash
# Clone
git clone https://github.com/hcchung1/ML-Simulator.git
cd ML-Simulator

# Build & run
dotnet run --project src/NeuralSim.App

# Run tests
dotnet test
```

### Publish a Self-Contained Executable

```bash
# Windows
publish.bat

# Output → src/NeuralSim.App/bin/Release/net9.0-windows/win-x64/publish/
```

## Usage

1. **Add operations** — Drag blocks from the left palette onto the canvas
2. **Wire them** — Click an output port (red) then an input port (blue) to connect
3. **Set input** — Edit the comma-separated values directly in the Input block
4. **Configure layers** — Select a Linear block and adjust `In` / `Out` features in the Inspector
5. **Run** — Click **▶ Run**, then use **Step ▶** to walk through each operation
6. **Inspect** — Click any block to view its tensor data in the Inspector panel

| Shortcut | Action |
|----------|--------|
| `DEL` | Delete selected node |
| `ESC` | Cancel wiring |
| Drag | Move nodes on canvas |

## Architecture

```
NeuralSim.sln
├── src/
│   ├── NeuralSim.Core/        # Tensor, Graph, Executor, Trace (pure C#, no UI)
│   │   ├── Tensor.cs           # N-dimensional tensor with flat float[] storage
│   │   ├── Graph.cs            # DAG of Ops with topological sort
│   │   ├── Executor.cs         # Runs forward pass, produces Trace
│   │   ├── Trace.cs            # Step-by-step execution record
│   │   └── Ops/                # Linear, ReLU, Sigmoid, Tanh, Softmax, Add
│   └── NeuralSim.App/         # WPF front-end
│       ├── ViewModels/         # MainViewModel (MVVM with CommunityToolkit)
│       └── Views/              # MainWindow, GraphCanvas (drag/drop/wiring)
└── tests/
    └── NeuralSim.Tests/        # Unit tests for Tensor ops and MLP builder
```

The core engine (`NeuralSim.Core`) is completely decoupled from the UI and can be used independently as a library.

## Tech Stack

| Component | Technology |
|-----------|------------|
| UI Framework | WPF (.NET 9) |
| MVVM | CommunityToolkit.Mvvm |
| Compute Engine | Custom Tensor / Graph / Executor |
| Language | C# 13 |
| Packaging | Single-file self-contained publish |

## License

[MIT](LICENSE) © 鍾承翰