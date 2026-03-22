// SPDX-License-Identifier: GPL-3.0-or-later
// Optimized input calculator that skips empty board points.

using GnuBgNet.NeuralNet;

namespace GnuBgNet.Optimized;

/// <summary>
/// Drop-in replacement for <see cref="InputCalculator"/> that skips empty board points
/// during feature extraction. Reduces writes from 200 (always) to ~80-100 (typical).
/// Uses Span.Clear() for fast zeroing instead of writing zero vectors per-point.
/// </summary>
public sealed class SparseInputCalculator : IInputCalculator
{
    public static readonly SparseInputCalculator Instance = new();

    private static readonly float[][] InputVec =
    [
        [0f, 0f, 0f, 0f],
        [1f, 0f, 0f, 0f],
        [0f, 1f, 0f, 0f],
        [0f, 0f, 1f, 0f],
        [0f, 0f, 1f, 0.5f],
        [0f, 0f, 1f, 1.0f],
        [0f, 0f, 1f, 1.5f],
        [0f, 0f, 1f, 2.0f],
        [0f, 0f, 1f, 2.5f],
        [0f, 0f, 1f, 3.0f],
        [0f, 0f, 1f, 3.5f],
        [0f, 0f, 1f, 4.0f],
        [0f, 0f, 1f, 4.5f],
        [0f, 0f, 1f, 5.0f],
        [0f, 0f, 1f, 5.5f],
        [0f, 0f, 1f, 6.0f],
    ];

    private static readonly float[][] InputVecBar =
    [
        [0f, 0f, 0f, 0f],
        [1f, 0f, 0f, 0f],
        [1f, 1f, 0f, 0f],
        [1f, 1f, 1f, 0f],
        [1f, 1f, 1f, 0.5f],
        [1f, 1f, 1f, 1.0f],
        [1f, 1f, 1f, 1.5f],
        [1f, 1f, 1f, 2.0f],
        [1f, 1f, 1f, 2.5f],
        [1f, 1f, 1f, 3.0f],
        [1f, 1f, 1f, 3.5f],
        [1f, 1f, 1f, 4.0f],
        [1f, 1f, 1f, 4.5f],
        [1f, 1f, 1f, 5.0f],
        [1f, 1f, 1f, 5.5f],
        [1f, 1f, 1f, 6.0f],
    ];

    public void BaseInputs(Board board, Span<float> arInput)
    {
        // Clear entire span once (fast memset) — then only write non-zero entries
        arInput.Slice(0, 200).Clear();

        WriteSideInputsSparse(board.Player, arInput, 0);
        WriteSideInputsSparse(board.Opponent, arInput, 100);
    }

    private static void WriteSideInputsSparse(uint[] side, Span<float> arInput, int baseOffset)
    {
        // Only write features for occupied points (skip empties)
        for (int i = 0; i < 24; i++)
        {
            uint nc = side[i];
            if (nc == 0) continue;

            int idx = Math.Min((int)nc, 15);
            var vec = InputVec[idx];
            int offset = baseOffset + i * 4;
            arInput[offset] = vec[0];
            arInput[offset + 1] = vec[1];
            arInput[offset + 2] = vec[2];
            arInput[offset + 3] = vec[3];
        }

        // Bar (point 24)
        uint bar = side[24];
        if (bar > 0)
        {
            int idx = Math.Min((int)bar, 15);
            var vec = InputVecBar[idx];
            int offset = baseOffset + 96;
            arInput[offset] = vec[0];
            arInput[offset + 1] = vec[1];
            arInput[offset + 2] = vec[2];
            arInput[offset + 3] = vec[3];
        }
    }

    // Race, contact, and crashed inputs delegate to the original static methods
    // since they have complex logic beyond base inputs that isn't worth duplicating.
    // The base inputs portion is the hot path that benefits from sparse optimization.

    public void CalculateRaceInputs(Board board, Span<float> inputs)
        => InputCalculator.CalculateRaceInputs(board, inputs);

    public void CalculateContactInputs(Board board, Span<float> arInput)
        => InputCalculator.CalculateContactInputs(board, arInput);

    public void CalculateCrashedInputs(Board board, Span<float> arInput)
        => InputCalculator.CalculateCrashedInputs(board, arInput);
}
