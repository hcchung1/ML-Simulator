using NeuralSim.Core.Ops;

namespace NeuralSim.Core.Builders;

/// <summary>
/// Convenient builder to create MLP graphs from layer sizes.
/// Produces a sequential graph of Linear + Activation ops.
/// </summary>
public static class MlpBuilder
{
    public enum Activation { None, ReLU, Sigmoid, Tanh, Softmax }

    /// <summary>
    /// Build a standard MLP.
    /// </summary>
    /// <param name="layerSizes">Number of neurons per layer, including input. E.g. [784, 128, 64, 10].</param>
    /// <param name="hiddenActivation">Activation after each hidden layer.</param>
    /// <param name="outputActivation">Activation after the output layer (often Softmax or None).</param>
    /// <param name="seed">Random seed for reproducible weights.</param>
    public static Graph Build(
        int[] layerSizes,
        Activation hiddenActivation = Activation.ReLU,
        Activation outputActivation = Activation.Softmax,
        int seed = 42)
    {
        if (layerSizes.Length < 2)
            throw new ArgumentException("Need at least input + output layer sizes");

        var rng = new Random(seed);
        var ops = new List<Op>();
        int opIdx = 0;

        for (int i = 0; i < layerSizes.Length - 1; i++)
        {
            int fanIn = layerSizes[i];
            int fanOut = layerSizes[i + 1];
            bool isLast = i == layerSizes.Length - 2;

            // Linear
            var linear = new LinearOp(
                $"linear_{i}",
                fanIn, fanOut,
                rng,
                $"Linear {i} ({fanIn}→{fanOut})");
            ops.Add(linear);
            opIdx++;

            // Activation
            var act = isLast ? outputActivation : hiddenActivation;
            var actOp = CreateActivation(act, $"act_{i}");
            if (actOp != null)
            {
                ops.Add(actOp);
                opIdx++;
            }
        }

        return Graph.Sequential([.. ops]);
    }

    private static Op? CreateActivation(Activation act, string id) => act switch
    {
        Activation.ReLU => new ReLUOp(id, id),
        Activation.Sigmoid => new SigmoidOp(id, id),
        Activation.Tanh => new TanhOp(id, id),
        Activation.Softmax => new SoftmaxOp(id, id),
        Activation.None => null,
        _ => null
    };
}
