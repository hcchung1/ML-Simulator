using NeuralSim.Core.Ops;

namespace NeuralSim.Core.Builders;

/// <summary>
/// Builds a ResNet-style graph with residual (skip) connections.
/// Structure: Conv → BN → ReLU → [ResBlock × N] → GlobalAvgPool → Linear → Softmax.
/// Each ResBlock: Conv → BN → ReLU → Conv → BN → Add(skip) → ReLU.
/// </summary>
public static class ResNetBuilder
{
    /// <summary>
    /// Build a mini ResNet.
    /// </summary>
    /// <param name="inputChannels">Input channels.</param>
    /// <param name="imageSize">Spatial size (H=W).</param>
    /// <param name="channels">Channel count for each stage, e.g. [16, 32].</param>
    /// <param name="blocksPerStage">How many residual blocks per stage.</param>
    /// <param name="numClasses">Output classes.</param>
    /// <param name="seed">Random seed.</param>
    public static Graph Build(
        int inputChannels = 1,
        int imageSize = 8,
        int[]? channels = null,
        int blocksPerStage = 1,
        int numClasses = 10,
        int seed = 42)
    {
        channels ??= [16, 32];
        var rng = new Random(seed);
        var graph = new Graph();
        int opIdx = 0;
        string lastOpId = "graph_input";
        string lastPort = "output";

        // Helper to add op and wire it
        void AddSequential(Op op, string inputPortName = "input")
        {
            graph.AddOp(op);
            op.InputPorts[inputPortName] = new OpPort(lastOpId, lastPort);
            lastOpId = op.Id;
            lastPort = "output";
        }

        // Initial conv: change input channels to first stage channels
        int currentChannels = inputChannels;
        int padding = 1;

        var stemConv = new Conv2DOp($"stem_conv", currentChannels, channels[0], 3,
            stride: 1, padding: padding, rng: rng, name: $"Stem Conv ({currentChannels}→{channels[0]})");
        AddSequential(stemConv);

        var stemBn = new BatchNorm2DOp($"stem_bn", channels[0], rng, $"Stem BN");
        AddSequential(stemBn);

        var stemRelu = new ReLUOp($"stem_relu", "Stem ReLU");
        AddSequential(stemRelu);

        currentChannels = channels[0];

        // Stages
        for (int stage = 0; stage < channels.Length; stage++)
        {
            int outCh = channels[stage];
            bool needDownsample = stage > 0; // downsample at stage boundaries

            for (int block = 0; block < blocksPerStage; block++)
            {
                bool isFirstBlock = block == 0 && needDownsample;
                int stride = isFirstBlock ? 2 : 1;
                string blockPrefix = $"s{stage}_b{block}";

                // Save skip connection source
                string skipOpId = lastOpId;
                string skipPort = lastPort;

                // Conv1
                var conv1 = new Conv2DOp($"{blockPrefix}_conv1", currentChannels, outCh, 3,
                    stride: stride, padding: 1, rng: rng,
                    name: $"ResBlock {stage}.{block} Conv1 ({currentChannels}→{outCh})");
                AddSequential(conv1);

                var bn1 = new BatchNorm2DOp($"{blockPrefix}_bn1", outCh, rng, $"BN {stage}.{block}.1");
                AddSequential(bn1);

                var relu1 = new ReLUOp($"{blockPrefix}_relu1", $"ReLU {stage}.{block}.1");
                AddSequential(relu1);

                // Conv2
                var conv2 = new Conv2DOp($"{blockPrefix}_conv2", outCh, outCh, 3,
                    stride: 1, padding: 1, rng: rng,
                    name: $"ResBlock {stage}.{block} Conv2 ({outCh}→{outCh})");
                AddSequential(conv2);

                var bn2 = new BatchNorm2DOp($"{blockPrefix}_bn2", outCh, rng, $"BN {stage}.{block}.2");
                AddSequential(bn2);

                // Skip connection: if channels or spatial size changed, use 1x1 conv
                string skipSourceId;
                string skipSourcePort;

                if (isFirstBlock || currentChannels != outCh)
                {
                    var skipConv = new Conv2DOp($"{blockPrefix}_skip_conv", currentChannels, outCh, 1,
                        stride: stride, padding: 0, rng: rng,
                        name: $"Skip Conv {stage}.{block} ({currentChannels}→{outCh})");
                    graph.AddOp(skipConv);
                    skipConv.InputPorts["input"] = new OpPort(skipOpId, skipPort);

                    var skipBn = new BatchNorm2DOp($"{blockPrefix}_skip_bn", outCh, rng, $"Skip BN {stage}.{block}");
                    graph.AddOp(skipBn);
                    skipBn.InputPorts["input"] = new OpPort(skipConv.Id, "output");

                    skipSourceId = skipBn.Id;
                    skipSourcePort = "output";
                }
                else
                {
                    skipSourceId = skipOpId;
                    skipSourcePort = skipPort;
                }

                // Add (residual connection)
                var add = new AddOp($"{blockPrefix}_add", $"Add {stage}.{block}");
                graph.AddOp(add);
                add.InputPorts["a"] = new OpPort(lastOpId, lastPort); // from conv2->bn2
                add.InputPorts["b"] = new OpPort(skipSourceId, skipSourcePort); // skip
                lastOpId = add.Id;
                lastPort = "output";

                // ReLU after add
                var relu2 = new ReLUOp($"{blockPrefix}_relu2", $"ReLU {stage}.{block}.2");
                AddSequential(relu2);

                currentChannels = outCh;
                opIdx++;
            }
        }

        // Global average pooling
        var gap = new GlobalAvgPool2DOp("global_avg_pool", "GlobalAvgPool");
        AddSequential(gap);

        // Classifier
        var fc = new LinearOp("fc_out", currentChannels, numClasses, rng,
            $"Linear ({currentChannels}→{numClasses})");
        AddSequential(fc);

        var softmax = new SoftmaxOp("softmax", "Softmax");
        AddSequential(softmax);

        return graph;
    }
}
