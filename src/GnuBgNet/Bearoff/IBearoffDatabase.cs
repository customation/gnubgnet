// SPDX-License-Identifier: GPL-3.0-or-later

namespace GnuBgNet.Bearoff;

/// <summary>
/// Interface for bearoff database lookup.
/// Allows plugging in alternative endgame databases (computed, compressed, remote, etc.).
/// </summary>
public interface IBearoffDatabase
{
    BearoffType Type { get; }
    int Points { get; }
    int Chequers { get; }
    bool Cubeful { get; }
    bool Gammon { get; }
    uint NumPositions { get; }
    void Evaluate(Board board, Span<float> output);
    void GetCubefulEquities(Board board, Span<float> equities);
    void GetDistribution(uint posId, Span<float> probs);
    void GetGammonDistribution(uint posId, Span<float> probs);
}
