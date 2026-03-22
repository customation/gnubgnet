// SPDX-License-Identifier: GPL-3.0-or-later

namespace GnuBgNet.MoveGeneration;

/// <summary>
/// Interface for legal move generation and application.
/// Allows plugging in move generators for backgammon variants
/// (nackgammon, acey-deucey, tavla, etc.).
/// </summary>
public interface IMoveGenerator
{
    MoveList GenerateMoves(Board board, int n0, int n1);
    void GenerateMovesInto(MoveList ml, Board board, int n0, int n1);
    Board ApplyMove(Board board, Move move);
    Board ApplyMoveAndSwap(Board board, Move move);
}
