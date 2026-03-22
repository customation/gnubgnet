// SPDX-License-Identifier: GPL-3.0-or-later

namespace GnuBgNet.NeuralNet;

/// <summary>
/// Interface for neural network evaluation.
/// Allows plugging in alternative implementations (ONNX, TorchSharp, etc.)
/// in place of the built-in 3-layer MLP.
/// </summary>
public interface INeuralNetwork
{
    int InputCount { get; }
    int HiddenCount { get; }
    int OutputCount { get; }
    bool Trained { get; }
    void Evaluate(ReadOnlySpan<float> input, Span<float> output, NNState? state = null);
}
