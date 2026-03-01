namespace NeuralSim.Core.Ops;

/// <summary>
/// Max pooling 2D.
/// Input port:  "input"  — shape (N, C, H, W)
/// Output port: "output" — shape (N, C, Hout, Wout)
/// </summary>
public sealed class MaxPool2DOp : Op
{
    public override string OpType => "MaxPool2D";
    public override IReadOnlyList<string> OutputNames { get; } = ["output"];

    public int KernelSize { get; }
    public int Stride { get; }

    public MaxPool2DOp(string id, int kernelSize = 2, int stride = -1, string? name = null)
        : base(id, name)
    {
        KernelSize = kernelSize;
        Stride = stride > 0 ? stride : kernelSize;
    }

    public override Dictionary<string, Tensor> Compute(Dictionary<string, Tensor> inputs)
    {
        var x = inputs["input"];
        return new() { ["output"] = Tensor.MaxPool2D(x, KernelSize, Stride) };
    }

    public override string Describe() =>
        $"MaxPool2D(k={KernelSize}, s={Stride})";
}
