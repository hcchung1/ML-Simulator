using NeuralSim.Core;
using NeuralSim.Core.Ops;
using NeuralSim.Core.Builders;
using Xunit;

namespace NeuralSim.Tests;

public class TransformerTests
{
    private const float Eps = 1e-4f;

    // ───── Tensor Transformer ops ─────

    [Fact]
    public void BatchedMatMul_CorrectResult()
    {
        // (2,2,3) x (2,3,2) → (2,2,2)
        var a = new Tensor([2, 2, 3], [
            1,2,3, 4,5,6,   // batch 0
            7,8,9, 10,11,12  // batch 1
        ]);
        var b = new Tensor([2, 3, 2], [
            1,2, 3,4, 5,6,   // batch 0
            7,8, 9,10, 11,12  // batch 1
        ]);
        var c = Tensor.BatchedMatMul(a, b);
        Assert.Equal([2, 2, 2], c.Shape);
        // batch 0: [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
        Assert.Equal(22f, c.Get3D(0, 0, 0), Eps);
        Assert.Equal(28f, c.Get3D(0, 0, 1), Eps);
        Assert.Equal(49f, c.Get3D(0, 1, 0), Eps);
        Assert.Equal(64f, c.Get3D(0, 1, 1), Eps);
    }

    [Fact]
    public void LayerNorm_NormalizesLastDim()
    {
        // (1, 3) input, gamma=1, beta=0 → should normalize to mean≈0, var≈1
        var input = new Tensor([1, 3], [1f, 2f, 3f]);
        var gamma = new Tensor([3], [1f, 1f, 1f]);
        var beta = Tensor.Zeros(3);

        var output = Tensor.LayerNorm(input, gamma, beta);
        // Mean of output should be ~0
        float mean = output.Data.Average();
        Assert.InRange(mean, -0.01f, 0.01f);
    }

    [Fact]
    public void LayerNorm_3D_Works()
    {
        var input = Tensor.Random(new Random(0), 1, 4, 8); // (B, S, D)
        var gamma = new Tensor([8]);
        Array.Fill(gamma.Data, 1f);
        var beta = Tensor.Zeros(8);

        var output = Tensor.LayerNorm(input, gamma, beta);
        Assert.Equal([1, 4, 8], output.Shape);
    }

    [Fact]
    public void Softmax_3D_EachSliceSumsToOne()
    {
        var a = Tensor.Random(new Random(0), 2, 3, 4);
        var c = Tensor.Softmax(a);
        Assert.Equal([2, 3, 4], c.Shape);
        // Each (b,s) slice of length D=4 should sum to 1
        for (int b = 0; b < 2; b++)
            for (int s = 0; s < 3; s++)
            {
                float sum = 0;
                for (int d = 0; d < 4; d++)
                    sum += c.Get3D(b, s, d);
                Assert.Equal(1f, sum, Eps);
            }
    }

    [Fact]
    public void Add_3D_Broadcast_1D()
    {
        var a = new Tensor([1, 2, 3], [1, 2, 3, 4, 5, 6]);
        var b = new Tensor([3], [10, 20, 30]);
        var c = Tensor.Add(a, b);
        Assert.Equal([1, 2, 3], c.Shape);
        Assert.Equal(11f, c.Get3D(0, 0, 0), Eps);
        Assert.Equal(22f, c.Get3D(0, 0, 1), Eps);
        Assert.Equal(33f, c.Get3D(0, 0, 2), Eps);
        Assert.Equal(14f, c.Get3D(0, 1, 0), Eps);
    }

    // ───── Ops ─────

    [Fact]
    public void MultiHeadAttentionOp_CorrectOutputShape()
    {
        var mha = new MultiHeadAttentionOp("mha", dModel: 8, numHeads: 2, rng: new Random(42));
        var input = Tensor.Random(new Random(0), 1, 4, 8); // (B=1, S=4, D=8)
        var result = mha.Compute(new() { ["input"] = input });
        Assert.Equal([1, 4, 8], result["output"].Shape);
    }

    [Fact]
    public void PositionalEncodingOp_AddsToInput()
    {
        var pe = new PositionalEncodingOp("pe", dModel: 8, maxSeqLen: 16);
        var input = Tensor.Zeros(1, 4, 8); // all zeros
        var result = pe.Compute(new() { ["input"] = input });
        // Output should not be all zeros (since PE values are added)
        Assert.True(result["output"].Data.Any(v => v != 0f));
    }

    [Fact]
    public void LayerNormOp_Works()
    {
        var op = new LayerNormOp("ln", 8);
        var input = Tensor.Random(new Random(0), 1, 4, 8);
        var result = op.Compute(new() { ["input"] = input });
        Assert.Equal([1, 4, 8], result["output"].Shape);
    }

    [Fact]
    public void MeanPool1DOp_ReducesSeqDim()
    {
        var op = new MeanPool1DOp("pool");
        var input = Tensor.Random(new Random(0), 1, 4, 8);
        var result = op.Compute(new() { ["input"] = input });
        Assert.Equal([1, 8], result["output"].Shape);
    }

    [Fact]
    public void FlatLinear3DOp_Handles3DInput()
    {
        var op = new FlatLinear3DOp("ff", 8, 16, rng: new Random(42));
        var input = Tensor.Random(new Random(0), 1, 4, 8);
        var result = op.Compute(new() { ["input"] = input });
        Assert.Equal([1, 4, 16], result["output"].Shape);
    }

    // ───── Transformer Builder ─────

    [Fact]
    public void TransformerBuilder_Build_ExecutesCorrectly()
    {
        var graph = TransformerBuilder.Build(
            dModel: 8, numHeads: 2, ffDim: 16,
            numLayers: 1, numClasses: 3, maxSeqLen: 16, seed: 42);

        var input = Tensor.Random(new Random(0), 1, 4, 8); // (B=1, S=4, D=8)
        var trace = Executor.Run(graph, input);

        Assert.NotNull(trace.Output);
        Assert.Equal([1, 3], trace.Output!.Shape);

        // Softmax output sums to ~1
        float sum = trace.Output.Data.Sum();
        Assert.Equal(1f, sum, Eps);
    }

    [Fact]
    public void TransformerBuilder_HasResidualConnections()
    {
        var graph = TransformerBuilder.Build(
            dModel: 8, numHeads: 2, ffDim: 16,
            numLayers: 1, numClasses: 3, seed: 42);

        bool hasAdd = graph.Nodes.Any(n => n.OpType == "Add");
        Assert.True(hasAdd, "Transformer should have Add ops for residual connections");
    }
}
