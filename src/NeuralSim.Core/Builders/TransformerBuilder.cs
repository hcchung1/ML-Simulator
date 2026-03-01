using NeuralSim.Core.Ops;

namespace NeuralSim.Core.Builders;

/// <summary>
/// Builds a Transformer encoder stack.
/// Structure: PosEnc → [MHA → Add → LN → FFN(Linear→ReLU→Linear) → Add → LN] × N → Linear.
/// </summary>
public static class TransformerBuilder
{
    /// <summary>
    /// Build a Transformer encoder.
    /// </summary>
    /// <param name="dModel">Model dimension.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="ffDim">Feed-forward hidden dimension.</param>
    /// <param name="numLayers">Number of encoder layers.</param>
    /// <param name="numClasses">Output classes (0 = no final linear).</param>
    /// <param name="maxSeqLen">Max sequence length for positional encoding.</param>
    /// <param name="seed">Random seed.</param>
    public static Graph Build(
        int dModel = 64,
        int numHeads = 4,
        int ffDim = 128,
        int numLayers = 2,
        int numClasses = 10,
        int maxSeqLen = 128,
        int seed = 42)
    {
        var rng = new Random(seed);
        var graph = new Graph();
        string lastOpId = "graph_input";
        string lastPort = "output";

        void AddSequential(Op op, string inputPort = "input")
        {
            graph.AddOp(op);
            op.InputPorts[inputPort] = new OpPort(lastOpId, lastPort);
            lastOpId = op.Id;
            lastPort = "output";
        }

        // Positional encoding
        var posEnc = new PositionalEncodingOp("pos_enc", dModel, maxSeqLen, "PositionalEncoding");
        AddSequential(posEnc);

        // Encoder layers
        for (int layer = 0; layer < numLayers; layer++)
        {
            string prefix = $"L{layer}";

            // ── Multi-Head Attention sub-block ──
            string beforeMhaId = lastOpId;
            string beforeMhaPort = lastPort;

            var mha = new MultiHeadAttentionOp($"{prefix}_mha", dModel, numHeads, rng,
                $"MHA {layer} (d={dModel}, h={numHeads})");
            AddSequential(mha);

            // Add (residual)
            var addMha = new AddOp($"{prefix}_add1", $"Add {layer}.1 (residual)");
            graph.AddOp(addMha);
            addMha.InputPorts["a"] = new OpPort(lastOpId, lastPort);
            addMha.InputPorts["b"] = new OpPort(beforeMhaId, beforeMhaPort);
            lastOpId = addMha.Id;
            lastPort = "output";

            // LayerNorm
            var ln1 = new LayerNormOp($"{prefix}_ln1", dModel, $"LayerNorm {layer}.1");
            AddSequential(ln1);

            // ── Feed-Forward sub-block ──
            string beforeFfId = lastOpId;
            string beforeFfPort = lastPort;

            // FFN: Linear → ReLU → Linear
            // Need to handle 3D→2D→3D for linear layers
            var ff1 = new FlatLinear3DOp($"{prefix}_ff1", dModel, ffDim, rng,
                $"FFN {layer} Linear1 ({dModel}→{ffDim})");
            AddSequential(ff1);

            var ffRelu = new ReLUOp($"{prefix}_ff_relu", $"FFN {layer} ReLU");
            AddSequential(ffRelu);

            var ff2 = new FlatLinear3DOp($"{prefix}_ff2", ffDim, dModel, rng,
                $"FFN {layer} Linear2 ({ffDim}→{dModel})");
            AddSequential(ff2);

            // Add (residual)
            var addFf = new AddOp($"{prefix}_add2", $"Add {layer}.2 (residual)");
            graph.AddOp(addFf);
            addFf.InputPorts["a"] = new OpPort(lastOpId, lastPort);
            addFf.InputPorts["b"] = new OpPort(beforeFfId, beforeFfPort);
            lastOpId = addFf.Id;
            lastPort = "output";

            // LayerNorm
            var ln2 = new LayerNormOp($"{prefix}_ln2", dModel, $"LayerNorm {layer}.2");
            AddSequential(ln2);
        }

        // Output projection (optional)
        if (numClasses > 0)
        {
            // Mean pooling across sequence → (B, D)
            var pool = new MeanPool1DOp("seq_pool", "SeqMeanPool");
            AddSequential(pool);

            var fcOut = new LinearOp("fc_out", dModel, numClasses, rng,
                $"Linear ({dModel}→{numClasses})");
            AddSequential(fcOut);

            var softmax = new SoftmaxOp("softmax", "Softmax");
            AddSequential(softmax);
        }

        return graph;
    }
}
