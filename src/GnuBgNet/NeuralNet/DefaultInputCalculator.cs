// SPDX-License-Identifier: GPL-3.0-or-later

namespace GnuBgNet.NeuralNet;

/// <summary>
/// Default <see cref="IInputCalculator"/> implementation that delegates to
/// the static <see cref="InputCalculator"/> methods (gnubg standard features).
/// </summary>
public sealed class DefaultInputCalculator : IInputCalculator
{
    public static readonly DefaultInputCalculator Instance = new();

    public void BaseInputs(Board board, Span<float> arInput)
        => InputCalculator.BaseInputs(board, arInput);

    public void CalculateRaceInputs(Board board, Span<float> inputs)
        => InputCalculator.CalculateRaceInputs(board, inputs);

    public void CalculateContactInputs(Board board, Span<float> arInput)
        => InputCalculator.CalculateContactInputs(board, arInput);

    public void CalculateCrashedInputs(Board board, Span<float> arInput)
        => InputCalculator.CalculateCrashedInputs(board, arInput);
}
