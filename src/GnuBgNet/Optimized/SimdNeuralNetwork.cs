// SPDX-License-Identifier: GPL-3.0-or-later
// Optimized neural network with TensorPrimitives and LUT sigmoid.

using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using GnuBgNet.NeuralNet;

namespace GnuBgNet.Optimized;

/// <summary>
/// Drop-in replacement for <see cref="NeuralNetwork"/> with:
/// - TensorPrimitives for weight accumulation (auto-selects AVX-512/AVX2/SSE)
/// - SIMD-vectorized sigmoid (Padé [2,2] rational approximation)
/// - TensorPrimitives.Dot for output layer
/// - stackalloc hidden buffer (zero heap allocation per eval)
/// </summary>
public sealed class SimdNeuralNetwork : INeuralNetwork
{
    public int InputCount { get; }
    public int HiddenCount { get; }
    public int OutputCount { get; }
    public float BetaHidden { get; }
    public float BetaOutput { get; }
    public bool Trained { get; }

    internal readonly float[] HiddenWeight;
    internal readonly float[] OutputWeight;
    internal readonly float[] HiddenThreshold;
    internal readonly float[] OutputThreshold;

    public SimdNeuralNetwork(NeuralNetwork source)
    {
        InputCount = source.InputCount;
        HiddenCount = source.HiddenCount;
        OutputCount = source.OutputCount;
        BetaHidden = source.BetaHidden;
        BetaOutput = source.BetaOutput;
        Trained = source.Trained;
        HiddenWeight = source.HiddenWeight;
        OutputWeight = source.OutputWeight;
        HiddenThreshold = source.HiddenThreshold;
        OutputThreshold = source.OutputThreshold;
    }

    public SimdNeuralNetwork(int inputCount, int hiddenCount, int outputCount,
                             float betaHidden, float betaOutput, bool trained,
                             float[] hiddenWeight, float[] outputWeight,
                             float[] hiddenThreshold, float[] outputThreshold)
    {
        InputCount = inputCount;
        HiddenCount = hiddenCount;
        OutputCount = outputCount;
        BetaHidden = betaHidden;
        BetaOutput = betaOutput;
        Trained = trained;
        HiddenWeight = hiddenWeight;
        OutputWeight = outputWeight;
        HiddenThreshold = hiddenThreshold;
        OutputThreshold = outputThreshold;
    }

    public void Evaluate(ReadOnlySpan<float> input, Span<float> output, NNState? state = null)
    {
        var action = state?.GetAction() ?? NNEvalType.None;

        Span<float> hidden = stackalloc float[HiddenCount];

        switch (action)
        {
            case NNEvalType.None:
                ComputeForwardPass(input, hidden, output, null);
                break;

            case NNEvalType.Save:
                state!.SavedIBaseCount = InputCount;
                input.Slice(0, InputCount).CopyTo(state.SavedIBase);
                ComputeForwardPass(input, hidden, output, state.SavedBase);
                break;

            case NNEvalType.FromBase:
                if (state!.SavedIBaseCount != InputCount)
                {
                    ComputeForwardPass(input, hidden, output, null);
                    break;
                }
                state.SavedBase.AsSpan(0, HiddenCount).CopyTo(hidden);

                Span<float> diff = stackalloc float[InputCount];
                for (int i = 0; i < InputCount; i++)
                {
                    float curr = input[i];
                    float saved = state.SavedIBase![i];
                    diff[i] = (curr != saved) ? curr - saved : 0.0f;
                }

                ComputFromBase(diff, hidden, output);
                break;
        }
    }

    private void ComputeForwardPass(ReadOnlySpan<float> input, Span<float> hidden, Span<float> output, float[]? saveHidden)
    {
        int cHidden = HiddenCount;

        HiddenThreshold.AsSpan(0, cHidden).CopyTo(hidden);

        // TensorPrimitives-based accumulation
        int weightOffset = 0;
        for (int i = 0; i < InputCount; i++)
        {
            float ari = input[i];
            if (ari == 0.0f)
            {
                weightOffset += cHidden;
            }
            else
            {
                var weights = HiddenWeight.AsSpan(weightOffset, cHidden);
                if (ari == 1.0f)
                {
                    TensorPrimitives.Add(hidden, weights, hidden);
                }
                else
                {
                    TensorPrimitives.MultiplyAdd(weights, ari, hidden, hidden);
                }
                weightOffset += cHidden;
            }
        }

        if (saveHidden != null)
            hidden.Slice(0, cHidden).CopyTo(saveHidden);

        // SIMD sigmoid using Padé approximation
        ApplySigmoid(hidden, BetaHidden);

        // TensorPrimitives.Dot for output layer
        ComputeOutputLayerSimd(hidden, output);
    }

    private void ComputFromBase(ReadOnlySpan<float> inputDiff, Span<float> hidden, Span<float> output)
    {
        int cHidden = HiddenCount;
        int weightOffset = 0;

        for (int i = 0; i < InputCount; i++)
        {
            float ari = inputDiff[i];
            if (ari == 0.0f)
            {
                weightOffset += cHidden;
            }
            else
            {
                var weights = HiddenWeight.AsSpan(weightOffset, cHidden);
                if (ari == 1.0f)
                {
                    TensorPrimitives.Add(hidden, weights, hidden);
                }
                else if (ari == -1.0f)
                {
                    TensorPrimitives.Subtract(hidden, weights, hidden);
                }
                else
                {
                    TensorPrimitives.MultiplyAdd(weights, ari, hidden, hidden);
                }
                weightOffset += cHidden;
            }
        }

        ApplySigmoid(hidden, BetaHidden);
        ComputeOutputLayerSimd(hidden, output);
    }

    /// <summary>
    /// Apply sigmoid using LUT (same as base NeuralNetwork).
    /// Benchmarks show LUT is faster than Padé rational approximation
    /// because scalar table lookup avoids costly SIMD division.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplySigmoid(Span<float> values, float beta)
    {
        for (int i = 0; i < values.Length; i++)
            values[i] = Sigmoid.Evaluate(-beta * values[i]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ComputeOutputLayerSimd(ReadOnlySpan<float> hidden, Span<float> output)
    {
        int cHidden = HiddenCount;
        int weightOffset = 0;

        for (int i = 0; i < OutputCount; i++)
        {
            float r = OutputThreshold[i] +
                TensorPrimitives.Dot(hidden, OutputWeight.AsSpan(weightOffset, cHidden));
            output[i] = Sigmoid.Evaluate(-BetaOutput * r);
            weightOffset += cHidden;
        }
    }
}
