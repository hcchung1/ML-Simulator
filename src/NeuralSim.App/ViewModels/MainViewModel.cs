using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using NeuralSim.Core;
using NeuralSim.Core.Ops;
using NeuralSim.Core.Builders;

namespace NeuralSim.App.ViewModels;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Canvas Node (draggable op block on canvas)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

public partial class CanvasNodeViewModel : ObservableObject
{
    private static int _counter;

    public string Id { get; init; } = $"node_{Interlocked.Increment(ref _counter)}";

    [ObservableProperty] private string _opType = "Linear";
    [ObservableProperty] private string _label = "Linear";
    [ObservableProperty] private double _x;
    [ObservableProperty] private double _y;
    [ObservableProperty] private string _status = "idle"; // idle, active, done
    [ObservableProperty] private string _outputSummary = "";
    [ObservableProperty] private bool _isSelected;

    // Linear-specific config
    [ObservableProperty] private int _inFeatures = 4;
    [ObservableProperty] private int _outFeatures = 8;

    // CNN-specific config
    [ObservableProperty] private int _inChannels = 1;
    [ObservableProperty] private int _outChannels = 16;
    [ObservableProperty] private int _kernelSize = 3;
    [ObservableProperty] private int _stride = 1;
    [ObservableProperty] private int _padding = 1;

    // Transformer-specific config
    [ObservableProperty] private int _numHeads = 4;
    [ObservableProperty] private int _dModel = 64;
    [ObservableProperty] private int _maxSeqLen = 100;

    /// <summary>Whether this node is one of the fixed Input/Output endpoints.</summary>
    public bool IsFixed { get; init; }

    /// <summary>Port names this node type exposes.</summary>
    public IReadOnlyList<string> InputPortNames => OpType switch
    {
        "GraphInput" => [],
        "GraphOutput" => ["input"],
        "Add" => ["a", "b"],
        _ => ["input"]
    };

    public IReadOnlyList<string> OutputPortNames => OpType switch
    {
        "GraphOutput" => [],         // no outputs
        _ => ["output"]              // single output
    };

    public bool IsLinear => OpType == "Linear";
    public bool IsConv2D => OpType == "Conv2D";
    public bool IsMaxPool2D => OpType == "MaxPool2D";
    public bool IsBatchNorm2D => OpType == "BatchNorm2D";
    public bool IsMultiHeadAttention => OpType == "MultiHeadAttention";
    public bool IsLayerNorm => OpType == "LayerNorm";
    public bool IsPositionalEncoding => OpType == "PositionalEncoding";
    public bool IsGraphInput => OpType == "GraphInput";
    public bool IsGraphOutput => OpType == "GraphOutput";
    public bool HasChannelConfig => OpType is "Conv2D" or "BatchNorm2D";
    public bool HasKernelConfig => OpType is "Conv2D" or "MaxPool2D";
    public bool HasTransformerConfig => OpType is "MultiHeadAttention" or "LayerNorm" or "PositionalEncoding";

    public string BlockColor => OpType switch
    {
        "Linear"     => "#4A90D9",
        "ReLU"       => "#E57373",
        "Sigmoid"    => "#66BB6A",
        "Tanh"       => "#FFA726",
        "Softmax"    => "#AB47BC",
        "Add"        => "#7986CB",
        "Conv2D"     => "#26A69A",
        "MaxPool2D"  => "#4DD0E1",
        "BatchNorm2D"=> "#5C6BC0",
        "Flatten"    => "#78909C",
        "GlobalAvgPool2D" => "#4FC3F7",
        "MultiHeadAttention" => "#FFB300",
        "LayerNorm"  => "#7E57C2",
        "PositionalEncoding" => "#C0CA33",
        "MeanPool1D" => "#4FC3F7",
        "GraphInput" => "#78909C",
        "GraphOutput"=> "#78909C",
        _            => "#607D8B"
    };

    /// <summary>Node width constant for edge calculation.</summary>
    public const double NodeWidth = 180;
    public const double NodeHeight = 60;

    /// <summary>Center X for edge drawing.</summary>
    public double CenterX => X + NodeWidth / 2;
    /// <summary>Top Y for input port.</summary>
    public double TopY => Y;
    /// <summary>Bottom Y for output port.</summary>
    public double BottomY => Y + NodeHeight;

    partial void OnXChanged(double value) { OnPropertyChanged(nameof(CenterX)); OnPropertyChanged(nameof(TopY)); OnPropertyChanged(nameof(BottomY)); }
    partial void OnYChanged(double value) { OnPropertyChanged(nameof(CenterX)); OnPropertyChanged(nameof(TopY)); OnPropertyChanged(nameof(BottomY)); }

    partial void OnOpTypeChanged(string value)
    {
        OnPropertyChanged(nameof(IsLinear));
        OnPropertyChanged(nameof(IsConv2D));
        OnPropertyChanged(nameof(IsMaxPool2D));
        OnPropertyChanged(nameof(IsBatchNorm2D));
        OnPropertyChanged(nameof(IsMultiHeadAttention));
        OnPropertyChanged(nameof(IsLayerNorm));
        OnPropertyChanged(nameof(IsPositionalEncoding));
        OnPropertyChanged(nameof(IsGraphInput));
        OnPropertyChanged(nameof(IsGraphOutput));
        OnPropertyChanged(nameof(HasChannelConfig));
        OnPropertyChanged(nameof(HasKernelConfig));
        OnPropertyChanged(nameof(HasTransformerConfig));
        OnPropertyChanged(nameof(BlockColor));
        OnPropertyChanged(nameof(InputPortNames));
        OnPropertyChanged(nameof(OutputPortNames));
    }

    partial void OnInFeaturesChanged(int value) => UpdateLabel();
    partial void OnOutFeaturesChanged(int value) => UpdateLabel();
    partial void OnInChannelsChanged(int value) => UpdateLabel();
    partial void OnOutChannelsChanged(int value) => UpdateLabel();
    partial void OnKernelSizeChanged(int value) => UpdateLabel();
    partial void OnDModelChanged(int value) => UpdateLabel();
    partial void OnNumHeadsChanged(int value) => UpdateLabel();

    public void UpdateLabel()
    {
        Label = OpType switch
        {
            "Linear" => $"Linear ({InFeatures}→{OutFeatures})",
            "Conv2D" => $"Conv2D ({InChannels}→{OutChannels}, k{KernelSize})",
            "MaxPool2D" => $"MaxPool (k{KernelSize})",
            "BatchNorm2D" => $"BN ({InChannels})",
            "MultiHeadAttention" => $"MHA (d{DModel}, h{NumHeads})",
            "LayerNorm" => $"LN ({DModel})",
            "PositionalEncoding" => $"PosEnc (d{DModel})",
            "GraphInput" => "Input",
            "GraphOutput" => "Output",
            _ => OpType
        };
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Connection (edge between two nodes)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

public partial class ConnectionViewModel : ObservableObject
{
    public string Id { get; init; } = Guid.NewGuid().ToString("N")[..8];
    public CanvasNodeViewModel From { get; init; } = null!;
    public string FromPort { get; init; } = "output";
    public CanvasNodeViewModel To { get; init; } = null!;
    public string ToPort { get; init; } = "input";

    /// <summary>Line coordinates, updated when nodes move.</summary>
    [ObservableProperty] private double _x1;
    [ObservableProperty] private double _y1;
    [ObservableProperty] private double _x2;
    [ObservableProperty] private double _y2;

    public void UpdatePositions()
    {
        X1 = From.CenterX;
        Y1 = From.BottomY;
        X2 = To.CenterX;
        Y2 = To.TopY;
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Palette item (left panel)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

public class PaletteItem
{
    public string OpType { get; init; } = "";
    public string Label { get; init; } = "";
    public string Color { get; init; } = "#607D8B";
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Main ViewModel
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

public partial class MainViewModel : ObservableObject
{
    // ───── Palette ─────
    public ObservableCollection<PaletteItem> Palette { get; } =
    [
        // Core
        new() { OpType = "Linear",  Label = "Linear",  Color = "#4A90D9" },
        new() { OpType = "Add",     Label = "Add",     Color = "#7986CB" },
        // Activations
        new() { OpType = "ReLU",    Label = "ReLU",    Color = "#E57373" },
        new() { OpType = "Sigmoid", Label = "Sigmoid", Color = "#66BB6A" },
        new() { OpType = "Tanh",    Label = "Tanh",    Color = "#FFA726" },
        new() { OpType = "Softmax", Label = "Softmax", Color = "#AB47BC" },
        // CNN
        new() { OpType = "Conv2D",      Label = "Conv2D",      Color = "#26A69A" },
        new() { OpType = "MaxPool2D",   Label = "MaxPool2D",   Color = "#4DD0E1" },
        new() { OpType = "BatchNorm2D", Label = "BatchNorm2D", Color = "#5C6BC0" },
        new() { OpType = "Flatten",     Label = "Flatten",     Color = "#78909C" },
        new() { OpType = "GlobalAvgPool2D", Label = "AvgPool", Color = "#4FC3F7" },
        // Transformer
        new() { OpType = "MultiHeadAttention", Label = "MH-Attention", Color = "#FFB300" },
        new() { OpType = "LayerNorm",  Label = "LayerNorm",  Color = "#7E57C2" },
        new() { OpType = "PositionalEncoding", Label = "PosEncoding", Color = "#C0CA33" },
        new() { OpType = "MeanPool1D", Label = "MeanPool1D", Color = "#4FC3F7" },
    ];

    // ───── Canvas ─────
    public ObservableCollection<CanvasNodeViewModel> CanvasNodes { get; } = [];
    public ObservableCollection<ConnectionViewModel> Connections { get; } = [];

    /// <summary>The fixed "Input" node.</summary>
    public CanvasNodeViewModel InputNode { get; }
    /// <summary>The fixed "Output" node.</summary>
    public CanvasNodeViewModel OutputNode { get; }

    // ───── Wiring state ─────
    [ObservableProperty] private CanvasNodeViewModel? _wiringSource;
    [ObservableProperty] private string _wiringSourcePort = "";
    [ObservableProperty] private bool _isWiring;
    [ObservableProperty] private bool _wiringSourceIsOutput;

    // ───── Input ─────
    [ObservableProperty] private string _inputText = "0.5, -0.3, 0.8, 0.1";
    [ObservableProperty] private int _inputSize = 4;
    [ObservableProperty] private int _seed = 42;

    // ───── Trace state ─────
    [ObservableProperty] private int _currentStep = -1;
    [ObservableProperty] private int _totalSteps;
    [ObservableProperty] private string _stepInfo = "Drop ops onto canvas, wire them, then Run.";
    [ObservableProperty] private bool _hasTrace;

    // ───── Inspector ─────
    [ObservableProperty] private CanvasNodeViewModel? _selectedNode;
    [ObservableProperty] private string _inspectorTitle = "Inspector";
    [ObservableProperty] private string _inspectorContent = "";
    [ObservableProperty] private string _parametersContent = "";

    // ───── Internal ─────
    private Trace? _trace;

    public MainViewModel()
    {
        // Fixed Input/Output nodes
        InputNode = new CanvasNodeViewModel
        {
            OpType = "GraphInput", Label = "Input", X = 310, Y = 20, IsFixed = true
        };
        OutputNode = new CanvasNodeViewModel
        {
            OpType = "GraphOutput", Label = "Output", X = 310, Y = 600, IsFixed = true
        };
        CanvasNodes.Add(InputNode);
        CanvasNodes.Add(OutputNode);
    }

    // ═══════ Canvas Commands ═══════

    /// <summary>Add a new op node onto the canvas at specified position.</summary>
    [RelayCommand]
    private void AddNodeToCanvas(PaletteDropInfo info)
    {
        var node = new CanvasNodeViewModel
        {
            OpType = info.OpType,
            X = info.X,
            Y = info.Y,
        };
        ApplyDefaults(node);
        node.UpdateLabel();
        CanvasNodes.Add(node);
    }

    private void ApplyDefaults(CanvasNodeViewModel node)
    {
        switch (node.OpType)
        {
            case "Linear":
                node.InFeatures = InputSize;
                node.OutFeatures = 8;
                break;
            case "Conv2D":
                node.InChannels = 1; node.OutChannels = 16;
                node.KernelSize = 3; node.Stride = 1; node.Padding = 1;
                break;
            case "MaxPool2D":
                node.KernelSize = 2; node.Stride = 2;
                break;
            case "BatchNorm2D":
                node.InChannels = 16;
                break;
            case "MultiHeadAttention":
                node.DModel = 64; node.NumHeads = 4;
                break;
            case "LayerNorm":
                node.DModel = 64;
                break;
            case "PositionalEncoding":
                node.DModel = 64; node.MaxSeqLen = 100;
                break;
        }
    }

    /// <summary>Remove a non-fixed node from the canvas.</summary>
    [RelayCommand]
    private void RemoveNode(CanvasNodeViewModel node)
    {
        if (node.IsFixed) return;
        var toRemove = Connections.Where(c => c.From == node || c.To == node).ToList();
        foreach (var c in toRemove) Connections.Remove(c);
        CanvasNodes.Remove(node);
    }

    /// <summary>Begin or complete a wiring operation when a port is clicked.</summary>
    [RelayCommand]
    private void PortClicked(PortClickInfo info)
    {
        if (!IsWiring)
        {
            // Start wiring
            WiringSource = info.Node;
            WiringSourcePort = info.PortName;
            WiringSourceIsOutput = info.IsOutput;
            IsWiring = true;
            string dir = info.IsOutput ? "output" : "input";
            StepInfo = $"Wiring from {info.Node.Label}.{info.PortName} ({dir}) — click a port to connect.";
            return;
        }

        // Complete wiring
        if (WiringSource == null) { CancelWiring(); return; }

        CanvasNodeViewModel from, to;
        string fromPort, toPort;

        if (WiringSourceIsOutput && !info.IsOutput)
        {
            from = WiringSource; fromPort = WiringSourcePort;
            to = info.Node; toPort = info.PortName;
        }
        else if (!WiringSourceIsOutput && info.IsOutput)
        {
            from = info.Node; fromPort = info.PortName;
            to = WiringSource; toPort = WiringSourcePort;
        }
        else
        {
            StepInfo = "Must connect output → input.";
            CancelWiring();
            return;
        }

        if (from == to) { StepInfo = "Cannot connect to self."; CancelWiring(); return; }

        // Remove existing connection to same input port
        var existing = Connections.FirstOrDefault(c => c.To == to && c.ToPort == toPort);
        if (existing != null) Connections.Remove(existing);

        var conn = new ConnectionViewModel { From = from, FromPort = fromPort, To = to, ToPort = toPort };
        conn.UpdatePositions();
        Connections.Add(conn);

        StepInfo = $"Connected {from.Label}.{fromPort} → {to.Label}.{toPort}";
        CancelWiring();
    }

    [RelayCommand]
    private void CancelWiring()
    {
        WiringSource = null;
        WiringSourcePort = "";
        IsWiring = false;
    }

    [RelayCommand]
    private void RemoveConnection(ConnectionViewModel conn)
    {
        Connections.Remove(conn);
    }

    /// <summary>Called when a node is dragged — update connected edge positions.</summary>
    public void OnNodeMoved(CanvasNodeViewModel node)
    {
        foreach (var c in Connections)
        {
            if (c.From == node || c.To == node)
                c.UpdatePositions();
        }
    }

    // ═══════ Build & Run ═══════

    [RelayCommand]
    private void BuildAndRun()
    {
        try
        {
            var inputValues = InputText
                .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(float.Parse)
                .ToArray();

            InputSize = inputValues.Length;
            var rng = new Random(Seed);

            var graph = new Graph();
            var opMap = new Dictionary<string, Op>();

            foreach (var node in CanvasNodes)
            {
                if (node.OpType is "GraphInput" or "GraphOutput") continue;

                Op op = node.OpType switch
                {
                    "Linear"  => new LinearOp(node.Id, node.InFeatures, node.OutFeatures, rng, node.Label),
                    "ReLU"    => new ReLUOp(node.Id, "ReLU"),
                    "Sigmoid" => new SigmoidOp(node.Id, "Sigmoid"),
                    "Tanh"    => new TanhOp(node.Id, "Tanh"),
                    "Softmax" => new SoftmaxOp(node.Id, "Softmax"),
                    "Add"     => new AddOp(node.Id, "Add"),
                    "Conv2D"  => new Conv2DOp(node.Id, node.InChannels, node.OutChannels,
                                    node.KernelSize, node.Stride, node.Padding, rng, node.Label),
                    "MaxPool2D" => new MaxPool2DOp(node.Id, node.KernelSize,
                                    node.Stride > 0 ? node.Stride : node.KernelSize, node.Label),
                    "BatchNorm2D" => new BatchNorm2DOp(node.Id, node.InChannels, rng, node.Label),
                    "Flatten"  => new FlattenOp(node.Id, "Flatten"),
                    "GlobalAvgPool2D" => new GlobalAvgPool2DOp(node.Id, "GlobalAvgPool2D"),
                    "LayerNorm" => new LayerNormOp(node.Id, node.DModel, node.Label),
                    "MultiHeadAttention" => new MultiHeadAttentionOp(node.Id, node.DModel, node.NumHeads, rng, node.Label),
                    "PositionalEncoding" => new PositionalEncodingOp(node.Id, node.DModel, node.MaxSeqLen, node.Label),
                    "MeanPool1D" => new MeanPool1DOp(node.Id, "MeanPool1D"),
                    _ => throw new InvalidOperationException($"Unknown op: {node.OpType}")
                };
                graph.AddOp(op);
                opMap[node.Id] = op;
            }

            // Wire connections
            foreach (var conn in Connections)
            {
                if (conn.To == OutputNode) continue; // output node is visual only

                string fromId = conn.From == InputNode ? "graph_input" : conn.From.Id;
                string fromPort = conn.FromPort;

                var targetOp = opMap[conn.To.Id];
                targetOp.InputPorts[conn.ToPort] = new OpPort(fromId, fromPort);
            }

            var input = new Tensor([1, inputValues.Length], inputValues);
            _trace = Executor.Run(graph, input);

            foreach (var n in CanvasNodes) { n.Status = "idle"; n.OutputSummary = ""; }

            TotalSteps = _trace.StepCount + 1; // +1 for the virtual output display step
            CurrentStep = -1;
            HasTrace = true;
            StepInfo = $"Traced {_trace.StepCount} ops. Step through to visualize.";
        }
        catch (Exception ex)
        {
            StepInfo = $"Error: {ex.Message}";
        }
    }

    [RelayCommand]
    private void StepForward()
    {
        if (_trace == null || CurrentStep >= TotalSteps - 1) return;
        CurrentStep++;
        ApplyStep(CurrentStep);
    }

    [RelayCommand]
    private void StepBackward()
    {
        if (_trace == null || CurrentStep <= 0) return;
        CurrentStep--;
        foreach (var n in CanvasNodes) { n.Status = "idle"; n.OutputSummary = ""; }
        for (int i = 0; i <= CurrentStep; i++) ApplyStep(i);
    }

    [RelayCommand]
    private void Reset()
    {
        CurrentStep = -1;
        foreach (var n in CanvasNodes) { n.Status = "idle"; n.OutputSummary = ""; }
        StepInfo = "Reset.";
        SelectedNode = null;
        InspectorTitle = "Inspector"; InspectorContent = ""; ParametersContent = "";
    }

    [RelayCommand]
    private void RunToEnd()
    {
        if (_trace == null) return;
        for (int i = CurrentStep + 1; i < TotalSteps; i++) { CurrentStep = i; ApplyStep(i); }
    }

    [RelayCommand]
    private void SelectNode(CanvasNodeViewModel? node)
    {
        if (SelectedNode != null) SelectedNode.IsSelected = false;
        SelectedNode = node;
        if (node != null) node.IsSelected = true;

        if (node == null || _trace == null)
        {
            InspectorTitle = "Inspector"; InspectorContent = ""; ParametersContent = "";
            return;
        }

        // Special handling for fixed Input/Output nodes
        if (node == InputNode)
        {
            InspectorTitle = "Input";
            InspectorContent = $"── Input Vector ──\n  {InputText}";
            ParametersContent = "";
            return;
        }

        if (node == OutputNode)
        {
            InspectorTitle = "Output";
            if (string.IsNullOrEmpty(node.OutputSummary))
            {
                InspectorContent = "(not yet executed)"; ParametersContent = ""; return;
            }

            // Find the source tensor feeding into the Output node
            var outputConn = Connections.FirstOrDefault(c => c.To == OutputNode);
            if (outputConn != null)
            {
                var sourceStep = _trace.Steps.FirstOrDefault(s => s.OpId == outputConn.From.Id);
                if (sourceStep != null)
                {
                    var outTensor = sourceStep.Outputs.Values.FirstOrDefault();
                    if (outTensor != null)
                    {
                        var sb = new System.Text.StringBuilder();
                        sb.AppendLine("── Final Output ──");
                        sb.AppendLine($"  shape=[{string.Join(",", outTensor.Shape)}]");
                        sb.AppendLine($"  {FormatTensorValues(outTensor)}");
                        InspectorContent = sb.ToString();
                        ParametersContent = "";
                        return;
                    }
                }
            }
            InspectorContent = node.OutputSummary;
            ParametersContent = "";
            return;
        }

        InspectorTitle = $"{node.Label} ({node.OpType})";
        var step = _trace.Steps.FirstOrDefault(s => s.OpId == node.Id);
        if (step == null || CurrentStep < step.Index)
        {
            InspectorContent = "(not yet executed)"; ParametersContent = ""; return;
        }

        var sb2 = new System.Text.StringBuilder();
        sb2.AppendLine("── Inputs ──");
        foreach (var (port, tensor) in step.Inputs)
        {
            sb2.AppendLine($"  {port}: shape=[{string.Join(",", tensor.Shape)}]");
            sb2.AppendLine($"    {FormatTensorValues(tensor)}");
        }
        sb2.AppendLine("\n── Outputs ──");
        foreach (var (port, tensor) in step.Outputs)
        {
            sb2.AppendLine($"  {port}: shape=[{string.Join(",", tensor.Shape)}]");
            sb2.AppendLine($"    {FormatTensorValues(tensor)}");
        }
        InspectorContent = sb2.ToString();

        if (step.Parameters.Count > 0)
        {
            var psb = new System.Text.StringBuilder();
            psb.AppendLine("── Parameters ──");
            foreach (var (name, tensor) in step.Parameters)
            {
                psb.AppendLine($"  {name}: shape=[{string.Join(",", tensor.Shape)}]");
                psb.AppendLine($"    {FormatTensorValues(tensor)}");
            }
            ParametersContent = psb.ToString();
        }
        else ParametersContent = "(no parameters)";
    }

    // ═══════ Internal ═══════

    private void ApplyStep(int stepIndex)
    {
        if (_trace == null) return;

        // Virtual final step: display result on Output block
        if (stepIndex >= _trace.StepCount)
        {
            // Mark all executed ops as "done"
            foreach (var node in CanvasNodes)
            {
                if (_trace.Steps.Any(s => s.OpId == node.Id))
                    node.Status = "done";
            }
            ShowOutputResult();
            return;
        }

        var step = _trace.Steps[stepIndex];

        foreach (var node in CanvasNodes)
        {
            if (node.Id == step.OpId)
            {
                node.Status = "active";
                var outTensor = step.Outputs.Values.FirstOrDefault();
                node.OutputSummary = outTensor != null ? FormatTensorShort(outTensor) : "";
            }
            else if (_trace.Steps.Any(s => s.Index < stepIndex && s.OpId == node.Id))
            {
                node.Status = "done";
            }
        }

        StepInfo = $"Step {stepIndex + 1}/{TotalSteps}: {step.Description}";
        var activeNode = CanvasNodes.FirstOrDefault(n => n.Id == step.OpId);
        if (activeNode != null) SelectNode(activeNode);
    }

    /// <summary>Show the final output tensor on the Output canvas block.</summary>
    private void ShowOutputResult()
    {
        var outputConn = Connections.FirstOrDefault(c => c.To == OutputNode);
        if (outputConn != null && _trace != null)
        {
            var sourceStep = _trace.Steps.FirstOrDefault(s => s.OpId == outputConn.From.Id);
            if (sourceStep != null)
            {
                var outTensor = sourceStep.Outputs.Values.FirstOrDefault();
                OutputNode.OutputSummary = outTensor != null ? FormatTensorShort(outTensor) : "";
            }
        }
        OutputNode.Status = "active";
        StepInfo = $"Step {TotalSteps}/{TotalSteps}: Final output → {OutputNode.OutputSummary}";
        SelectNode(OutputNode);
    }

    private static string FormatTensorValues(Tensor t)
    {
        if (t.Length <= 20) return $"[{string.Join(", ", t.Data.Select(v => v.ToString("F4")))}]";
        return $"[{string.Join(", ", t.Data.Take(8).Select(v => v.ToString("F4")))} ... ({t.Length} total)]";
    }

    private static string FormatTensorShort(Tensor t)
    {
        if (t.Length <= 6) return string.Join(", ", t.Data.Select(v => v.ToString("F3")));
        return $"{string.Join(", ", t.Data.Take(4).Select(v => v.ToString("F3")))} ...";
    }

    // ═══════ Project Save/Load ═══════

    public ProjectFile ExportProject()
    {
        var project = new ProjectFile
        {
            Name = "Untitled",
            InputText = InputText,
            Seed = Seed,
        };

        foreach (var node in CanvasNodes)
        {
            project.Nodes.Add(new ProjectNode
            {
                Id = node.Id,
                OpType = node.OpType,
                Label = node.Label,
                X = node.X,
                Y = node.Y,
                IsFixed = node.IsFixed,
                InFeatures = node.InFeatures,
                OutFeatures = node.OutFeatures,
                InChannels = node.InChannels,
                OutChannels = node.OutChannels,
                KernelSize = node.KernelSize,
                Stride = node.Stride,
                Padding = node.Padding,
                NumHeads = node.NumHeads,
                DModel = node.DModel,
                MaxSeqLen = node.MaxSeqLen,
            });
        }

        foreach (var conn in Connections)
        {
            project.Connections.Add(new ProjectConnection
            {
                FromId = conn.From.Id,
                FromPort = conn.FromPort,
                ToId = conn.To.Id,
                ToPort = conn.ToPort,
            });
        }

        return project;
    }

    public void ImportProject(ProjectFile project)
    {
        // Clear non-fixed nodes and all connections
        var toRemove = CanvasNodes.Where(n => !n.IsFixed).ToList();
        foreach (var n in toRemove) CanvasNodes.Remove(n);
        Connections.Clear();

        InputText = project.InputText;
        Seed = project.Seed;

        var nodeMap = new Dictionary<string, CanvasNodeViewModel>();

        foreach (var pNode in project.Nodes)
        {
            if (pNode.OpType == "GraphInput")
            {
                InputNode.X = pNode.X;
                InputNode.Y = pNode.Y;
                nodeMap[pNode.Id] = InputNode;
                continue;
            }
            if (pNode.OpType == "GraphOutput")
            {
                OutputNode.X = pNode.X;
                OutputNode.Y = pNode.Y;
                nodeMap[pNode.Id] = OutputNode;
                continue;
            }

            var node = new CanvasNodeViewModel
            {
                OpType = pNode.OpType,
                Label = pNode.Label,
                X = pNode.X,
                Y = pNode.Y,
                InFeatures = pNode.InFeatures,
                OutFeatures = pNode.OutFeatures,
                InChannels = pNode.InChannels,
                OutChannels = pNode.OutChannels,
                KernelSize = pNode.KernelSize,
                Stride = pNode.Stride,
                Padding = pNode.Padding,
                NumHeads = pNode.NumHeads,
                DModel = pNode.DModel,
                MaxSeqLen = pNode.MaxSeqLen,
            };
            CanvasNodes.Add(node);
            nodeMap[pNode.Id] = node;
        }

        foreach (var pConn in project.Connections)
        {
            if (!nodeMap.TryGetValue(pConn.FromId, out var from)) continue;
            if (!nodeMap.TryGetValue(pConn.ToId, out var to)) continue;

            var conn = new ConnectionViewModel
            {
                From = from,
                FromPort = pConn.FromPort,
                To = to,
                ToPort = pConn.ToPort,
            };
            conn.UpdatePositions();
            Connections.Add(conn);
        }

        ResetCommand.Execute(null);
        StepInfo = $"Loaded project: {project.Name}";
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DTOs for command parameters
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

public record PaletteDropInfo(string OpType, double X, double Y);
public record PortClickInfo(CanvasNodeViewModel Node, string PortName, bool IsOutput);
