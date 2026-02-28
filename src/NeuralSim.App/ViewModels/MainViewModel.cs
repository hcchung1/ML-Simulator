using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using NeuralSim.Core;
using NeuralSim.Core.Builders;

namespace NeuralSim.App.ViewModels;

/// <summary>
/// Node visual model for the graph canvas.
/// </summary>
public partial class OpNodeViewModel : ObservableObject
{
    public string OpId { get; init; } = "";
    public string Label { get; init; } = "";
    public string OpType { get; init; } = "";
    public int LayerIndex { get; init; }

    [ObservableProperty] private string _status = "idle"; // "idle", "active", "done"
    [ObservableProperty] private string _outputSummary = "";
    [ObservableProperty] private double _x;
    [ObservableProperty] private double _y;
}

/// <summary>
/// Edge visual model.
/// </summary>
public class EdgeViewModel
{
    public string FromOpId { get; init; } = "";
    public string ToOpId { get; init; } = "";
}

public partial class MainViewModel : ObservableObject
{
    // ───── Model config (user-editable) ─────
    [ObservableProperty] private string _layerSizesText = "4, 8, 6, 3";
    [ObservableProperty] private string _activationChoice = "ReLU";
    [ObservableProperty] private string _outputActivationChoice = "Softmax";
    [ObservableProperty] private int _seed = 42;

    // ───── Input ─────
    [ObservableProperty] private string _inputText = "0.5, -0.3, 0.8, 0.1";

    // ───── Trace state ─────
    [ObservableProperty] private int _currentStep = -1;
    [ObservableProperty] private int _totalSteps;
    [ObservableProperty] private string _stepInfo = "Ready — build a model and run.";
    [ObservableProperty] private bool _hasTrace;

    // ───── Inspector ─────
    [ObservableProperty] private OpNodeViewModel? _selectedNode;
    [ObservableProperty] private string _inspectorTitle = "Inspector";
    [ObservableProperty] private string _inspectorContent = "";
    [ObservableProperty] private string _parametersContent = "";

    // ───── Graph visual elements ─────
    public ObservableCollection<OpNodeViewModel> Nodes { get; } = [];
    public ObservableCollection<EdgeViewModel> Edges { get; } = [];

    // ───── Available choices ─────
    public string[] ActivationChoices { get; } = ["ReLU", "Sigmoid", "Tanh", "None"];
    public string[] OutputActivationChoices { get; } = ["Softmax", "Sigmoid", "None"];

    // ───── internal state ─────
    private Graph? _graph;
    private Trace? _trace;

    public MainViewModel()
    {
        BuildAndRunCommand.Execute(null);
    }

    // ═══════ Commands ═══════

    [RelayCommand]
    private void BuildAndRun()
    {
        try
        {
            // Parse layer sizes
            var sizes = LayerSizesText
                .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(int.Parse)
                .ToArray();

            // Parse input
            var inputValues = InputText
                .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(float.Parse)
                .ToArray();

            if (inputValues.Length != sizes[0])
            {
                StepInfo = $"Error: input has {inputValues.Length} values but first layer expects {sizes[0]}.";
                return;
            }

            var hiddenAct = ParseActivation(ActivationChoice);
            var outAct = ParseActivation(OutputActivationChoice);

            // Build graph
            _graph = MlpBuilder.Build(sizes, hiddenAct, outAct, Seed);

            // Create input tensor (1 sample, batch=1)
            var input = new Tensor([1, sizes[0]], inputValues);

            // Execute and trace
            _trace = Executor.Run(_graph, input);

            // Build visual nodes
            RebuildVisualGraph();

            TotalSteps = _trace.StepCount;
            CurrentStep = -1;
            HasTrace = true;
            StepInfo = $"Model built: {sizes.Length} layers, {_trace.StepCount} ops. Use Step/Play to visualize.";
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
        // Reset all to idle then replay up to current step
        foreach (var n in Nodes) { n.Status = "idle"; n.OutputSummary = ""; }
        for (int i = 0; i <= CurrentStep; i++)
            ApplyStep(i);
    }

    [RelayCommand]
    private void Reset()
    {
        CurrentStep = -1;
        foreach (var n in Nodes) { n.Status = "idle"; n.OutputSummary = ""; }
        StepInfo = "Reset. Press Step or Run to begin.";
        SelectedNode = null;
        InspectorTitle = "Inspector";
        InspectorContent = "";
        ParametersContent = "";
    }

    [RelayCommand]
    private void RunToEnd()
    {
        if (_trace == null) return;
        for (int i = CurrentStep + 1; i < TotalSteps; i++)
        {
            CurrentStep = i;
            ApplyStep(i);
        }
    }

    [RelayCommand]
    private void SelectNode(OpNodeViewModel? node)
    {
        SelectedNode = node;
        if (node == null || _trace == null)
        {
            InspectorTitle = "Inspector";
            InspectorContent = "";
            ParametersContent = "";
            return;
        }

        InspectorTitle = $"{node.Label} ({node.OpType})";

        // Find the corresponding trace step
        var step = _trace.Steps.FirstOrDefault(s => s.OpId == node.OpId);
        if (step == null || CurrentStep < step.Index)
        {
            InspectorContent = "(not yet executed)";
            ParametersContent = "";
            return;
        }

        // Show inputs & outputs
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("── Inputs ──");
        foreach (var (port, tensor) in step.Inputs)
        {
            sb.AppendLine($"  {port}: shape=[{string.Join(",", tensor.Shape)}]");
            sb.AppendLine($"    {FormatTensorValues(tensor)}");
        }
        sb.AppendLine();
        sb.AppendLine("── Outputs ──");
        foreach (var (port, tensor) in step.Outputs)
        {
            sb.AppendLine($"  {port}: shape=[{string.Join(",", tensor.Shape)}]");
            sb.AppendLine($"    {FormatTensorValues(tensor)}");
        }
        InspectorContent = sb.ToString();

        // Show parameters
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
        else
        {
            ParametersContent = "(no parameters)";
        }
    }

    // ═══════ Internal ═══════

    private void RebuildVisualGraph()
    {
        Nodes.Clear();
        Edges.Clear();

        if (_graph == null) return;

        var order = _graph.TopologicalOrder();
        double canvasWidth = 800;
        double ySpacing = 90;
        double yStart = 30;

        for (int i = 0; i < order.Count; i++)
        {
            var op = order[i];
            Nodes.Add(new OpNodeViewModel
            {
                OpId = op.Id,
                Label = op.Name,
                OpType = op.OpType,
                LayerIndex = i,
                X = canvasWidth / 2 - 80,
                Y = yStart + i * ySpacing,
                Status = "idle"
            });
        }

        // Edges
        for (int i = 0; i < order.Count; i++)
        {
            var op = order[i];
            foreach (var port in op.InputPorts.Values)
            {
                if (port.OpId == "graph_input") continue;
                Edges.Add(new EdgeViewModel { FromOpId = port.OpId, ToOpId = op.Id });
            }
        }
    }

    private void ApplyStep(int stepIndex)
    {
        if (_trace == null) return;
        var step = _trace.Steps[stepIndex];

        // Update node status
        foreach (var node in Nodes)
        {
            if (node.OpId == step.OpId)
            {
                node.Status = "active";
                // Show output summary on node
                var outTensor = step.Outputs.Values.FirstOrDefault();
                node.OutputSummary = outTensor != null ? FormatTensorShort(outTensor) : "";
            }
            else if (_trace.Steps.Any(s => s.Index < stepIndex && s.OpId == node.OpId))
            {
                node.Status = "done";
            }
        }

        StepInfo = $"Step {stepIndex + 1}/{TotalSteps}: {step.Description}";

        // Auto-select the active node for inspector
        var activeNode = Nodes.FirstOrDefault(n => n.OpId == step.OpId);
        if (activeNode != null) SelectNode(activeNode);
    }

    private static string FormatTensorValues(Tensor t)
    {
        if (t.Length <= 20)
            return $"[{string.Join(", ", t.Data.Select(v => v.ToString("F4")))}]";
        return $"[{string.Join(", ", t.Data.Take(8).Select(v => v.ToString("F4")))} ... ({t.Length} elements)]";
    }

    private static string FormatTensorShort(Tensor t)
    {
        if (t.Length <= 6)
            return string.Join(", ", t.Data.Select(v => v.ToString("F3")));
        return $"{string.Join(", ", t.Data.Take(4).Select(v => v.ToString("F3")))} ...";
    }

    private static MlpBuilder.Activation ParseActivation(string name) => name switch
    {
        "ReLU" => MlpBuilder.Activation.ReLU,
        "Sigmoid" => MlpBuilder.Activation.Sigmoid,
        "Tanh" => MlpBuilder.Activation.Tanh,
        "Softmax" => MlpBuilder.Activation.Softmax,
        _ => MlpBuilder.Activation.None
    };
}
