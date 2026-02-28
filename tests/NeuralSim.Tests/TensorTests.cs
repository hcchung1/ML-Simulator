using NeuralSim.Core;
using Xunit;

namespace NeuralSim.Tests;

public class TensorTests
{
    private const float Eps = 1e-5f;

    [Fact]
    public void MatMul_2x3_times_3x2_CorrectResult()
    {
        // A = [[1,2,3],[4,5,6]]  B = [[7,8],[9,10],[11,12]]
        // Expected C = [[58,64],[139,154]]
        var a = new Tensor([2, 3], [1, 2, 3, 4, 5, 6]);
        var b = new Tensor([3, 2], [7, 8, 9, 10, 11, 12]);
        var c = Tensor.MatMul(a, b);

        Assert.Equal([2, 2], c.Shape);
        Assert.Equal(58f, c.Get2D(0, 0), Eps);
        Assert.Equal(64f, c.Get2D(0, 1), Eps);
        Assert.Equal(139f, c.Get2D(1, 0), Eps);
        Assert.Equal(154f, c.Get2D(1, 1), Eps);
    }

    [Fact]
    public void Add_BiassBroadcast_Works()
    {
        var a = new Tensor([2, 3], [1, 2, 3, 4, 5, 6]);
        var bias = new Tensor([3], [10, 20, 30]);
        var c = Tensor.Add(a, bias);

        Assert.Equal([2, 3], c.Shape);
        Assert.Equal(11f, c.Get2D(0, 0), Eps);
        Assert.Equal(22f, c.Get2D(0, 1), Eps);
        Assert.Equal(33f, c.Get2D(0, 2), Eps);
        Assert.Equal(14f, c.Get2D(1, 0), Eps);
        Assert.Equal(25f, c.Get2D(1, 1), Eps);
        Assert.Equal(36f, c.Get2D(1, 2), Eps);
    }

    [Fact]
    public void ReLU_NegativesBecomZero()
    {
        var a = new Tensor([4], [-2, -1, 0, 3]);
        var c = Tensor.ReLU(a);
        Assert.Equal([0f, 0f, 0f, 3f], c.Data);
    }

    [Fact]
    public void Sigmoid_KnownValues()
    {
        var a = new Tensor([3], [0, 2, -2]);
        var c = Tensor.Sigmoid(a);
        Assert.Equal(0.5f, c[0], Eps);
        Assert.InRange(c[1], 0.88f, 0.89f); // sigmoid(2) ≈ 0.8808
        Assert.InRange(c[2], 0.11f, 0.13f); // sigmoid(-2) ≈ 0.1192
    }

    [Fact]
    public void Softmax_SumsToOne()
    {
        var a = new Tensor([4], [1, 2, 3, 4]);
        var c = Tensor.Softmax(a);
        float sum = c.Data.Sum();
        Assert.Equal(1f, sum, 1e-4f);
    }

    [Fact]
    public void Softmax_2D_EachRowSumsToOne()
    {
        var a = new Tensor([2, 3], [1, 2, 3, 4, 5, 6]);
        var c = Tensor.Softmax(a);
        float row0 = c.Get2D(0, 0) + c.Get2D(0, 1) + c.Get2D(0, 2);
        float row1 = c.Get2D(1, 0) + c.Get2D(1, 1) + c.Get2D(1, 2);
        Assert.Equal(1f, row0, 1e-4f);
        Assert.Equal(1f, row1, 1e-4f);
    }

    [Fact]
    public void Transpose2D_CorrectShape()
    {
        var a = new Tensor([2, 3], [1, 2, 3, 4, 5, 6]);
        var t = Tensor.Transpose2D(a);
        Assert.Equal([3, 2], t.Shape);
        Assert.Equal(1f, t.Get2D(0, 0));
        Assert.Equal(4f, t.Get2D(0, 1));
        Assert.Equal(2f, t.Get2D(1, 0));
    }
}
