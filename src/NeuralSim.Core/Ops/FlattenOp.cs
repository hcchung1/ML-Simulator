namespace NeuralSim.Core.Ops;

/// <summary>
/// Flatten all dimensions except batch: (N, ...) → (N, D).
/// Input port:  "input"
/// Output port: "output"
/// </summary>
public sealed class FlattenOp : Op
{
    public override string OpType => "Flatten";
    public override IReadOnlyList<string> OutputNames { get; } = ["output"];

    public FlattenOp(string id, string? name = null) : base(id, name) { }

    public override Dictionary<string, Tensor> Compute(Dictionary<string, Tensor> inputs)
    {
        var x = inputs["input"];
        if (x.Rank <= 2) return new() { ["output"] = x };
        int batch = x.Shape[0];
        int flatSize = x.Length / batch;
        return new() { ["output"] = x.Reshape(batch, flatSize) };
    }

    public override string Describe() => "Flatten";
}
