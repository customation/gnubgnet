// SPDX-License-Identifier: GPL-3.0-or-later
// Optimized neural network with SIMD-vectorized sigmoid and output layer dot product.

using System.Numerics;
using System.Runtime.CompilerServices;
using GnuBgNet.NeuralNet;

namespace GnuBgNet.Optimized;

/// <summary>
/// Drop-in replacement for <see cref="NeuralNetwork"/> with:
/// - SIMD-vectorized sigmoid (Padé [2,2] rational approximation for tanh)
/// - SIMD-vectorized output layer dot product
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

        // stackalloc: zero heap allocation for hidden layer
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

        int weightOffset = 0;
        for (int i = 0; i < InputCount; i++)
        {
            float ari = input[i];
            if (ari == 0.0f)
            {
                weightOffset += cHidden;
            }
            else if (ari == 1.0f)
            {
                AccumulateWeightsAdd(hidden, HiddenWeight.AsSpan(weightOffset, cHidden));
                weightOffset += cHidden;
            }
            else
            {
                AccumulateWeightsMul(hidden, HiddenWeight.AsSpan(weightOffset, cHidden), ari);
                weightOffset += cHidden;
            }
        }

        if (saveHidden != null)
            hidden.Slice(0, cHidden).CopyTo(saveHidden);

        // SIMD sigmoid: process Vector<float>.Count elements at a time
        ApplySigmoidSimd(hidden, BetaHidden);

        // SIMD output layer
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
            else if (ari == 1.0f)
            {
                AccumulateWeightsAdd(hidden, HiddenWeight.AsSpan(weightOffset, cHidden));
                weightOffset += cHidden;
            }
            else if (ari == -1.0f)
            {
                AccumulateWeightsSub(hidden, HiddenWeight.AsSpan(weightOffset, cHidden));
                weightOffset += cHidden;
            }
            else
            {
                AccumulateWeightsMul(hidden, HiddenWeight.AsSpan(weightOffset, cHidden), ari);
                weightOffset += cHidden;
            }
        }

        ApplySigmoidSimd(hidden, BetaHidden);
        ComputeOutputLayerSimd(hidden, output);
    }

    /// <summary>
    /// SIMD sigmoid using Padé [2,2] rational approximation for tanh.
    /// gnubg sigmoid(x) = 1/(1+exp(x)). Called as sigmoid(-beta*h) = σ(beta*h).
    /// Identity: σ(beta*h) = 0.5 + 0.5 * tanh(beta*h/2).
    /// Padé [2,2]: tanh(y) ≈ y*(15+y²)/(15+6y²), accurate for |y| &lt; 3.
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

            // y = beta*h/2
            var y = h * halfBeta;
            var y2 = y * y;

            // tanh(y) ≈ y*(15+y²)/(15+6y²)
            var num = y * (fifteen + y2);
            var den = fifteen + six * y2;
            var tanh_y = num / den;

            // sigmoid = 0.5 + 0.5 * tanh, clamped to [0,1]
            var result = half + half * tanh_y;
            result = Vector.Max(zero, Vector.Min(one, result));
            result.CopyTo(values.Slice(i, vecSize));
        }

        // Scalar remainder using original LUT for accuracy
        for (; i < values.Length; i++)
            values[i] = Sigmoid.Evaluate(-beta * values[i]);
    }

    /// <summary>
    /// SIMD-vectorized output layer: dot product of hidden × weights + threshold, then sigmoid.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ComputeOutputLayerSimd(ReadOnlySpan<float> hidden, Span<float> output)
    {
        int cHidden = HiddenCount;
        int vecSize = Vector<float>.Count;
        int weightOffset = 0;

        for (int i = 0; i < OutputCount; i++)
        {
            var sum = Vector<float>.Zero;
            int j = 0;

            for (; j + vecSize <= cHidden; j += vecSize)
            {
                var h = new Vector<float>(hidden.Slice(j, vecSize));
                var w = new Vector<float>(OutputWeight.AsSpan(weightOffset + j, vecSize));
                sum += h * w;
            }

            // Horizontal sum
            float r = OutputThreshold[i] + Vector.Dot(sum, Vector<float>.One);

            // Scalar remainder
            for (; j < cHidden; j++)
                r += hidden[j] * OutputWeight[weightOffset + j];

            output[i] = Sigmoid.Evaluate(-BetaOutput * r);
            weightOffset += cHidden;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateWeightsAdd(Span<float> hidden, ReadOnlySpan<float> weights)
    {
        int vecSize = Vector<float>.Count;
        int i = 0;
        for (; i + vecSize <= hidden.Length; i += vecSize)
        {
            var h = new Vector<float>(hidden.Slice(i, vecSize));
            var w = new Vector<float>(weights.Slice(i, vecSize));
            (h + w).CopyTo(hidden.Slice(i, vecSize));
        }
        for (; i < hidden.Length; i++)
            hidden[i] += weights[i];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateWeightsSub(Span<float> hidden, ReadOnlySpan<float> weights)
    {
        int vecSize = Vector<float>.Count;
        int i = 0;
        for (; i + vecSize <= hidden.Length; i += vecSize)
        {
            var h = new Vector<float>(hidden.Slice(i, vecSize));
            var w = new Vector<float>(weights.Slice(i, vecSize));
            (h - w).CopyTo(hidden.Slice(i, vecSize));
        }
        for (; i < hidden.Length; i++)
            hidden[i] -= weights[i];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateWeightsMul(Span<float> hidden, ReadOnlySpan<float> weights, float scalar)
    {
        int vecSize = Vector<float>.Count;
        var vScalar = new Vector<float>(scalar);
        int i = 0;
        for (; i + vecSize <= hidden.Length; i += vecSize)
        {
            var h = new Vector<float>(hidden.Slice(i, vecSize));
            var w = new Vector<float>(weights.Slice(i, vecSize));
            (h + w * vScalar).CopyTo(hidden.Slice(i, vecSize));
        }
        for (; i < hidden.Length; i++)
            hidden[i] += weights[i] * scalar;
    }
}
