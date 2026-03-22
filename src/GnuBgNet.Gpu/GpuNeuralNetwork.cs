// SPDX-License-Identifier: GPL-3.0-or-later
// GPU-accelerated neural network using ONNX Runtime with DirectML.

using GnuBgNet.Gpu.Onnx;
using GnuBgNet.NeuralNet;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace GnuBgNet.Gpu;

/// <summary>
/// GPU-accelerated neural network using ONNX Runtime + DirectML.
/// For single evaluations, this is SLOWER than CPU due to kernel launch overhead.
/// The win comes from <see cref="EvaluateBatch"/> which amortizes GPU overhead across
/// many positions — use during move scoring and rollouts.
/// </summary>
public sealed class GpuNeuralNetwork : IBatchNeuralNetwork, IDisposable
{
    private readonly InferenceSession _session;

    public int InputCount { get; }
    public int HiddenCount { get; }
    public int OutputCount { get; }
    public float BetaHidden { get; }
    public float BetaOutput { get; }
    public bool Trained { get; }

    /// <summary>
    /// Create a GPU-accelerated network from an existing CPU network.
    /// Exports weights to ONNX and loads with DirectML execution provider.
    /// Falls back to CPU ONNX Runtime if DirectML is unavailable.
    /// </summary>
    public GpuNeuralNetwork(NeuralNetwork source, bool preferCpu = false)
    {
        InputCount = source.InputCount;
        HiddenCount = source.HiddenCount;
        OutputCount = source.OutputCount;
        BetaHidden = source.BetaHidden;
        BetaOutput = source.BetaOutput;
        Trained = source.Trained;

        byte[] modelBytes = OnnxModelBuilder.Build(source);
        var options = new SessionOptions();

        if (!preferCpu)
        {
            try
            {
                options.AppendExecutionProvider_DML(0);
            }
            catch (Exception)
            {
                // DirectML unavailable — fall back to CPU
            }
        }

        _session = new InferenceSession(modelBytes, options);
    }

    /// <summary>
    /// Single-position evaluation. API-compatible but slower than CPU for small networks.
    /// Prefer <see cref="EvaluateBatch"/> for performance.
    /// </summary>
    public void Evaluate(ReadOnlySpan<float> input, Span<float> output, NNState? state = null)
    {
        // NNState save/restore not supported on GPU — always do full forward pass
        var inputTensor = new DenseTensor<float>(input.ToArray(), [1, InputCount]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor),
        };

        using var results = _session.Run(inputs);
        var outputTensor = results[0].AsTensor<float>();

        int count = Math.Min(OutputCount, output.Length);
        for (int i = 0; i < count; i++)
            output[i] = outputTensor[0, i];
    }

    /// <summary>
    /// Batched evaluation — the primary GPU advantage.
    /// Evaluates <paramref name="batchSize"/> positions in one GPU kernel launch.
    /// </summary>
    /// <param name="batchedInput">Flat array of batchSize × InputCount floats.</param>
    /// <param name="batchedOutput">Flat array of batchSize × OutputCount floats (filled on return).</param>
    /// <param name="batchSize">Number of positions in the batch.</param>
    public void EvaluateBatch(ReadOnlySpan<float> batchedInput, Span<float> batchedOutput, int batchSize)
    {
        var inputTensor = new DenseTensor<float>(batchedInput.ToArray(), [batchSize, InputCount]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor),
        };

        using var results = _session.Run(inputs);
        var outputTensor = results[0].AsTensor<float>();

        int outputStride = OutputCount;
        for (int b = 0; b < batchSize; b++)
        for (int o = 0; o < outputStride; o++)
            batchedOutput[b * outputStride + o] = outputTensor[b, o];
    }

    public void Dispose()
    {
        _session.Dispose();
    }
}
