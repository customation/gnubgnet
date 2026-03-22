// SPDX-License-Identifier: GPL-3.0-or-later

namespace GnuBgNet.MoveGeneration;

/// <summary>
/// Default <see cref="IMoveGenerator"/> implementation that delegates to
/// the static <see cref="MoveGenerator"/> methods (standard backgammon rules).
/// </summary>
public sealed class DefaultMoveGenerator : IMoveGenerator
{
    public static readonly DefaultMoveGenerator Instance = new();

    public MoveList GenerateMoves(Board board, int n0, int n1)
        => MoveGenerator.GenerateMoves(board, n0, n1);

    public void GenerateMovesInto(MoveList ml, Board board, int n0, int n1)
        => MoveGenerator.GenerateMovesInto(ml, board, n0, n1);

    public Board ApplyMove(Board board, Move move)
        => MoveGenerator.ApplyMove(board, move);

    public Board ApplyMoveAndSwap(Board board, Move move)
        => MoveGenerator.ApplyMoveAndSwap(board, move);
}
