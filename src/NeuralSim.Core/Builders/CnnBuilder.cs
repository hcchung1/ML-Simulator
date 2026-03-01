using NeuralSim.Core.Ops;

namespace NeuralSim.Core.Builders;

/// <summary>
/// Builds a simple CNN graph: Conv → BN → ReLU → Pool → ... → Flatten → Linear → Softmax.
/// </summary>
public static class CnnBuilder
{
    /// <summary>
    /// Build a standard CNN classifier.
    /// </summary>
    /// <param name="inputChannels">Number of input channels (e.g. 1 for grayscale, 3 for RGB).</param>
    /// <param name="imageSize">Spatial size (H=W assumed square).</param>
    /// <param name="convChannels">Output channels for each conv layer, e.g. [16, 32].</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <param name="kernelSize">Conv kernel size (default 3).</param>
    /// <param name="seed">Random seed.</param>
    public static Graph Build(
        int inputChannels = 1,
        int imageSize = 8,
        int[]? convChannels = null,
        int numClasses = 10,
        int kernelSize = 3,
        int seed = 42)
    {
        convChannels ??= [16, 32];
        var rng = new Random(seed);
        var ops = new List<Op>();
        int idx = 0;
        int currentChannels = inputChannels;
        int currentSize = imageSize;
        int padding = kernelSize / 2; // same padding

        foreach (int outCh in convChannels)
        {
            // Conv
            ops.Add(new Conv2DOp($"conv_{idx}", currentChannels, outCh, kernelSize,
                stride: 1, padding: padding, rng: rng, name: $"Conv2D {idx} ({currentChannels}→{outCh})"));

            // BatchNorm
            ops.Add(new BatchNorm2DOp($"bn_{idx}", outCh, rng, $"BN {idx} ({outCh})"));

            // ReLU
            ops.Add(new ReLUOp($"relu_{idx}", $"ReLU {idx}"));

            // MaxPool
            ops.Add(new MaxPool2DOp($"pool_{idx}", kernelSize: 2, stride: 2, name: $"MaxPool {idx}"));
            currentSize /= 2;

            currentChannels = outCh;
            idx++;
        }

        // Flatten
        ops.Add(new FlattenOp("flatten", "Flatten"));

        int flatSize = currentChannels * currentSize * currentSize;

        // Classifier head
        ops.Add(new LinearOp("fc_out", flatSize, numClasses, rng, $"Linear ({flatSize}→{numClasses})"));
        ops.Add(new SoftmaxOp("softmax", "Softmax"));

        return Graph.Sequential([.. ops]);
    }
}
