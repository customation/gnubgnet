// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2019 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Ported from neuralnet.c

using System.Numerics;
using System.Runtime.CompilerServices;

namespace GnuBgNet.NeuralNet;

/// <summary>
/// A 3-layer feedforward neural network: input → hidden (sigmoid) → output (sigmoid).
/// Port of the neuralnet struct and evaluation functions from neuralnet.c.
/// </summary>
public sealed class NeuralNetwork : INeuralNetwork
{
    public int InputCount { get; }
    public int HiddenCount { get; }
    public int OutputCount { get; }
    public float BetaHidden { get; }
    public float BetaOutput { get; }
    public bool Trained { get; }

    // Weight arrays (read-only after loading)
    internal readonly float[] HiddenWeight;   // [InputCount * HiddenCount]
    internal readonly float[] OutputWeight;    // [HiddenCount * OutputCount]
    internal readonly float[] HiddenThreshold; // [HiddenCount]
    internal readonly float[] OutputThreshold; // [OutputCount]

    public NeuralNetwork(int inputCount, int hiddenCount, int outputCount,
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

    /// <summary>
    /// Evaluate the neural network. Port of NeuralNetEvaluate / Evaluate from neuralnet.c.
    /// </summary>
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
                // Copy saved base and compute difference
                state.SavedBase.AsSpan(0, HiddenCount).CopyTo(hidden);

                // Compute input diff in-place on stack
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

    /// <summary>
    /// Full forward pass: hidden layer from thresholds + input*weights, then sigmoid, then output layer.
    /// Port of Evaluate() from neuralnet.c.
    /// </summary>
    private void ComputeForwardPass(ReadOnlySpan<float> input, Span<float> hidden, Span<float> output, float[]? saveHidden)
    {
        int cHidden = HiddenCount;

        // Initialize hidden layer from thresholds
        HiddenThreshold.AsSpan(0, cHidden).CopyTo(hidden);

        // Accumulate input * weights into hidden layer
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

        // Save pre-sigmoid hidden layer if requested
        if (saveHidden != null)
            hidden.Slice(0, cHidden).CopyTo(saveHidden);

        // Apply sigmoid to hidden layer
        for (int i = 0; i < cHidden; i++)
            hidden[i] = Sigmoid.Evaluate(-BetaHidden * hidden[i]);

        // Compute output layer
        ComputeOutputLayer(hidden, output);
    }

    /// <summary>
    /// Incremental forward pass from saved base.
    /// Port of EvaluateFromBase() from neuralnet.c.
    /// </summary>
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

        for (int i = 0; i < cHidden; i++)
            hidden[i] = Sigmoid.Evaluate(-BetaHidden * hidden[i]);

        ComputeOutputLayer(hidden, output);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ComputeOutputLayer(ReadOnlySpan<float> hidden, Span<float> output)
    {
        int cHidden = HiddenCount;
        int weightOffset = 0;

        for (int i = 0; i < OutputCount; i++)
        {
            float r = OutputThreshold[i];
            for (int j = 0; j < cHidden; j++)
                r += hidden[j] * OutputWeight[weightOffset + j];
            output[i] = Sigmoid.Evaluate(-BetaOutput * r);
            weightOffset += cHidden;
        }
    }

    /// <summary>hidden[j] += weights[j] using SIMD.</summary>
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

    /// <summary>hidden[j] -= weights[j] using SIMD.</summary>
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

    /// <summary>hidden[j] += weights[j] * scalar using SIMD.</summary>
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
