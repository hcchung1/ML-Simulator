namespace NeuralSim.Core.Ops;

/// <summary>
/// Element-wise addition of two tensors (for residual / skip connections).
/// Input ports: "a", "b"
/// Output port: "output"
/// </summary>
public sealed class AddOp : Op
{
    public override string OpType => "Add";
    public override IReadOnlyList<string> OutputNames { get; } = ["output"];

    public AddOp(string id, string? name = null) : base(id, name) { }

    public override Dictionary<string, Tensor> Compute(Dictionary<string, Tensor> inputs)
    {
        var a = inputs["a"];
        var b = inputs["b"];
        return new() { ["output"] = Tensor.Add(a, b) };
    }
}
