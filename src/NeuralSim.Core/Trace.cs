namespace NeuralSim.Core;

/// <summary>
/// A single step in an execution trace.
/// Captures everything needed for the UI to replay and inspect.
/// </summary>
public sealed class TraceStep
{
    /// <summary>Step index (0-based).</summary>
    public int Index { get; init; }

    /// <summary>Op that was executed.</summary>
    public string OpId { get; init; } = "";

    /// <summary>Op display name.</summary>
    public string OpName { get; init; } = "";

    /// <summary>Op type tag (e.g. "Linear", "ReLU").</summary>
    public string OpType { get; init; } = "";

    /// <summary>Human-readable description.</summary>
    public string Description { get; init; } = "";

    /// <summary>Input tensors consumed by this op (port name → cloned tensor).</summary>
    public Dictionary<string, Tensor> Inputs { get; init; } = new();

    /// <summary>Output tensors produced by this op (port name → cloned tensor).</summary>
    public Dictionary<string, Tensor> Outputs { get; init; } = new();

    /// <summary>Snapshot of parameters at this step (weight, bias, etc.).</summary>
    public Dictionary<string, Tensor> Parameters { get; init; } = new();
}

/// <summary>
/// Complete execution trace for a single forward pass.
/// UI replays this step-by-step.
/// </summary>
public sealed class Trace
{
    public List<TraceStep> Steps { get; } = [];

    /// <summary>The original graph input tensor.</summary>
    public Tensor? Input { get; set; }

    /// <summary>The final output tensor.</summary>
    public Tensor? Output { get; set; }

    public int StepCount => Steps.Count;
}
