// SPDX-License-Identifier: GPL-3.0-or-later

using GnuBgNet.NeuralNet;

namespace GnuBgNet.Gpu;

/// <summary>
/// Neural network that supports batched evaluation.
/// Single <see cref="INeuralNetwork.Evaluate"/> is for API compatibility;
/// the real win is <see cref="EvaluateBatch"/> which amortizes overhead.
/// </summary>
public interface IBatchNeuralNetwork : INeuralNetwork
{
    /// <summary>
    /// Evaluate <paramref name="batchSize"/> positions in one call.
    /// </summary>
    /// <param name="batchedInput">Flat: batchSize × InputCount floats.</param>
    /// <param name="batchedOutput">Flat: batchSize × OutputCount floats.</param>
    /// <param name="batchSize">Number of positions.</param>
    void EvaluateBatch(ReadOnlySpan<float> batchedInput, Span<float> batchedOutput, int batchSize);
}
