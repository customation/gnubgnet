// SPDX-License-Identifier: GPL-3.0-or-later

namespace GnuBgNet.NeuralNet;

/// <summary>
/// Interface for converting board positions to neural network input vectors.
/// Typically paired with a specific neural network architecture — a different
/// <see cref="INeuralNetwork"/> likely needs a different IInputCalculator.
/// </summary>
public interface IInputCalculator
{
    void BaseInputs(Board board, Span<float> arInput);
    void CalculateRaceInputs(Board board, Span<float> inputs);
    void CalculateContactInputs(Board board, Span<float> arInput);
    void CalculateCrashedInputs(Board board, Span<float> arInput);
}
