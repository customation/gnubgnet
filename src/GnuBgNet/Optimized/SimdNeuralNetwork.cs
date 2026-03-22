// SPDX-License-Identifier: GPL-3.0-or-later
// Optimized neural network with TensorPrimitives and vectorized sigmoid.

using System.Numerics;
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
        ApplySigmoidSimd(hidden, BetaHidden);

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

        ApplySigmoidSimd(hidden, BetaHidden);
        ComputeOutputLayerSimd(hidden, output);
    }

    /// <summary>
    /// SIMD sigmoid using Padé [2,2] rational approximation for tanh.
    /// σ(β·h) = 0.5 + 0.5·tanh(β·h/2), with tanh(y) ≈ y·(15+y²)/(15+6y²).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplySigmoidSimd(Span<float> values, float beta)
    {
        int vecSize = Vector<float>.Count;
        var halfBeta = new Vector<float>(0.5f * beta);
        var half = new Vector<float>(0.5f);
        var fifteen = new Vector<float>(15f);
        var six = new Vector<float>(6f);
        var one = new Vector<float>(1f);
        var zero = Vector<float>.Zero;

        int i = 0;
        for (; i + vecSize <= values.Length; i += vecSize)
        {
            var h = new Vector<float>(values.Slice(i, vecSize));

            var y = h * halfBeta;
            var y2 = y * y;

            var num = y * (fifteen + y2);
            var den = fifteen + six * y2;
            var tanh_y = num / den;

            var result = half + half * tanh_y;
            result = Vector.Max(zero, Vector.Min(one, result));
            result.CopyTo(values.Slice(i, vecSize));
        }

        // Scalar remainder using original LUT for accuracy
        for (; i < values.Length; i++)
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
