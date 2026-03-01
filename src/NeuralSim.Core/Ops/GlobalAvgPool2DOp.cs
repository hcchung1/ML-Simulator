namespace NeuralSim.Core.Ops;

/// <summary>
/// Global average pooling: (N, C, H, W) → (N, C).
/// Input port:  "input"
/// Output port: "output"
/// </summary>
public sealed class GlobalAvgPool2DOp : Op
{
    public override string OpType => "GlobalAvgPool2D";
    public override IReadOnlyList<string> OutputNames { get; } = ["output"];

    public GlobalAvgPool2DOp(string id, string? name = null) : base(id, name) { }

    public override Dictionary<string, Tensor> Compute(Dictionary<string, Tensor> inputs)
    {
        var x = inputs["input"];
        return new() { ["output"] = Tensor.GlobalAvgPool2D(x) };
    }

    public override string Describe() => "GlobalAvgPool2D";
}
