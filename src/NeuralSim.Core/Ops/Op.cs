namespace NeuralSim.Core.Ops;

/// <summary>
/// Base class for all compute graph operations.
/// Every op knows its inputs/outputs (by port name) and can compute forward.
/// Designed as a node in a DAG — not restricted to sequential chains.
/// </summary>
public abstract class Op
{
    /// <summary>Unique node id within a graph.</summary>
    public string Id { get; }

    /// <summary>Human-readable display name (e.g. "Linear_0", "ReLU_1").</summary>
    public string Name { get; set; }

    /// <summary>Short type tag for UI rendering (e.g. "Linear", "ReLU", "Conv2D", "Attention").</summary>
    public abstract string OpType { get; }

    /// <summary>
    /// Named input ports. Each port references an (OpId, OutputPortName) tuple,
    /// resolved at execution time. For simple sequential graphs the wiring is
    /// done automatically by <see cref="Graph"/>.
    /// </summary>
    public Dictionary<string, OpPort> InputPorts { get; } = new();

    /// <summary>Names of output ports this op produces.</summary>
    public abstract IReadOnlyList<string> OutputNames { get; }

    /// <summary>Learnable parameters (weights, biases, etc.).</summary>
    public Dictionary<string, Tensor> Parameters { get; } = new();

    protected Op(string id, string? name = null)
    {
        Id = id;
        Name = name ?? id;
    }

    /// <summary>
    /// Execute the op given resolved input tensors.
    /// Returns a dict of output-port-name → tensor.
    /// </summary>
    public abstract Dictionary<string, Tensor> Compute(Dictionary<string, Tensor> inputs);

    /// <summary>
    /// Generate a short summary string for the trace (e.g. shape info).
    /// </summary>
    public virtual string Describe() => $"{OpType} ({Name})";
}

/// <summary>
/// Reference to another op's output port.
/// </summary>
public readonly record struct OpPort(string OpId, string PortName);
