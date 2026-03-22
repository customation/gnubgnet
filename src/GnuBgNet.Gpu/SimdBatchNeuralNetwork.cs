// SPDX-License-Identifier: GPL-3.0-or-later
// CPU batch neural network using SIMD forward pass in a loop.

using GnuBgNet.NeuralNet;
using GnuBgNet.Optimized;

namespace GnuBgNet.Gpu;

/// <summary>
/// CPU-based batched neural network wrapping <see cref="SimdNeuralNetwork"/>.
/// Evaluates batch by looping — no GPU overhead, useful as a comparison baseline
/// and for systems without DirectML.
/// </summary>
public sealed class SimdBatchNeuralNetwork : IBatchNeuralNetwork
{
    private readonly SimdNeuralNetwork _inner;

    public int InputCount => _inner.InputCount;
    public int HiddenCount => _inner.HiddenCount;
    public int OutputCount => _inner.OutputCount;
    public float BetaHidden => _inner.BetaHidden;
    public float BetaOutput => _inner.BetaOutput;
    public bool Trained => _inner.Trained;

    public SimdBatchNeuralNetwork(NeuralNetwork source)
    {
        _inner = new SimdNeuralNetwork(source);
    }

    public SimdBatchNeuralNetwork(SimdNeuralNetwork simd)
    {
        _inner = simd;
    }

    public void Evaluate(ReadOnlySpan<float> input, Span<float> output, NNState? state = null)
        => _inner.Evaluate(input, output, state);

    public void EvaluateBatch(ReadOnlySpan<float> batchedInput, Span<float> batchedOutput, int batchSize)
    {
        int inStride = InputCount;
        int outStride = OutputCount;

        for (int b = 0; b < batchSize; b++)
        {
            var input = batchedInput.Slice(b * inStride, inStride);
            var output = batchedOutput.Slice(b * outStride, outStride);
            _inner.Evaluate(input, output);
        }
    }
}
