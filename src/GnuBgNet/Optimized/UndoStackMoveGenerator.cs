// SPDX-License-Identifier: GPL-3.0-or-later
// Optimized move generator using in-place apply/undo instead of Board.Clone().

using GnuBgNet.Encoding;
using GnuBgNet.MoveGeneration;

namespace GnuBgNet.Optimized;

/// <summary>
/// Drop-in replacement for <see cref="MoveGenerator"/> that eliminates Board.Clone()
/// during recursive move generation by using in-place apply/undo.
/// Reduces heap allocations from O(moves × depth) to O(1) during generation.
/// </summary>
public sealed class UndoStackMoveGenerator : IMoveGenerator
{
    public static readonly UndoStackMoveGenerator Instance = new();

    public MoveList GenerateMoves(Board board, int n0, int n1)
    {
        var ml = new MoveList();
        GenerateMovesInto(ml, board, n0, n1);
        return ml;
    }

    public void GenerateMovesInto(MoveList ml, Board board, int n0, int n1)
    {
        ml.Moves.Clear();
        ml.MaxMoves = 0;
        ml.MaxPips = 0;

        Span<int> anMoves = stackalloc int[8];
        anMoves.Fill(-1);

        Span<int> anRoll = stackalloc int[4];
        if (n0 == n1)
        {
            anRoll[0] = anRoll[1] = anRoll[2] = anRoll[3] = n0;
            GenerateMovesSub(ml, anRoll, 0, 23, 0, ref board, anMoves);
        }
        else
        {
            anRoll[0] = n0; anRoll[1] = n1; anRoll[2] = 0; anRoll[3] = 0;
            GenerateMovesSub(ml, anRoll, 0, 23, 0, ref board, anMoves);

            anRoll[0] = n1; anRoll[1] = n0;
            GenerateMovesSub(ml, anRoll, 0, 23, 0, ref board, anMoves);
        }
    }

    public Board ApplyMove(Board board, Move move)
        => MoveGenerator.ApplyMove(board, move);

    public Board ApplyMoveAndSwap(Board board, Move move)
        => MoveGenerator.ApplyMoveAndSwap(board, move);

    /// <summary>
    /// Undo information for a single sub-move.
    /// </summary>
    private struct UndoInfo
    {
        public int Src;
        public int Dest;   // -1 if bore off
        public bool WasHit;
    }

    /// <summary>
    /// Apply a sub-move in-place, returning undo information.
    /// </summary>
    private static UndoInfo ApplySubMoveInPlace(ref Board board, int iSrc, int nRoll)
    {
        int iDest = iSrc - nRoll;
        var undo = new UndoInfo { Src = iSrc, Dest = iDest, WasHit = false };

        board.Player[iSrc]--;

        if (iDest < 0)
            return undo; // bore off

        if (board.Opponent[23 - iDest] == 1)
        {
            // Hit
            board.Opponent[23 - iDest] = 0;
            board.Opponent[24]++;
            undo.WasHit = true;
        }

        board.Player[iDest]++;
        return undo;
    }

    /// <summary>
    /// Undo a sub-move, restoring the board to its previous state.
    /// </summary>
    private static void UndoSubMove(ref Board board, in UndoInfo undo)
    {
        board.Player[undo.Src]++;

        if (undo.Dest < 0)
            return; // was bear-off

        board.Player[undo.Dest]--;

        if (undo.WasHit)
        {
            board.Opponent[24]--;
            board.Opponent[23 - undo.Dest] = 1;
        }
    }

    private static bool LegalMove(Board board, int iSrc, int nPips)
    {
        int iDest = iSrc - nPips;

        if (iDest >= 0)
            return board.Opponent[23 - iDest] < 2;

        // Bearing off
        int nBack;
        for (nBack = 24; nBack > 0; nBack--)
            if (board.Player[nBack] > 0) break;

        return nBack <= 5 && (iSrc == nBack || iDest == -1);
    }

    /// <summary>
    /// Recursive move generation using in-place apply/undo instead of Board.Clone().
    /// </summary>
    private static bool GenerateMovesSub(MoveList ml, Span<int> anRoll, int nMoveDepth,
        int iPip, int cPip, ref Board board, Span<int> anMoves)
    {
        if (nMoveDepth > 3 || anRoll[nMoveDepth] == 0)
            return true;

        // If checker on bar, must enter
        if (board.Player[24] > 0)
        {
            int entryPoint = anRoll[nMoveDepth] - 1;
            if (board.Opponent[entryPoint] >= 2)
                return true; // blocked

            anMoves[nMoveDepth * 2] = 24;
            anMoves[nMoveDepth * 2 + 1] = 24 - anRoll[nMoveDepth];

            // Apply in-place instead of Clone
            var undo = ApplySubMoveInPlace(ref board, 24, anRoll[nMoveDepth]);

            if (GenerateMovesSub(ml, anRoll, nMoveDepth + 1, 23,
                    cPip + anRoll[nMoveDepth], ref board, anMoves))
                SaveMoves(ml, nMoveDepth + 1, (uint)(cPip + anRoll[nMoveDepth]), anMoves, board);

            // Undo
            UndoSubMove(ref board, in undo);

            return false;
        }

        bool fUsed = false;
        for (int i = iPip; i >= 0; i--)
        {
            if (board.Player[i] == 0 || !LegalMove(board, i, anRoll[nMoveDepth]))
                continue;

            anMoves[nMoveDepth * 2] = i;
            anMoves[nMoveDepth * 2 + 1] = i - anRoll[nMoveDepth];

            // Apply in-place instead of Clone
            var undo = ApplySubMoveInPlace(ref board, i, anRoll[nMoveDepth]);

            int nextIPip = (anRoll[0] == anRoll[1]) ? i : 23;

            if (GenerateMovesSub(ml, anRoll, nMoveDepth + 1, nextIPip,
                    cPip + anRoll[nMoveDepth], ref board, anMoves))
                SaveMoves(ml, nMoveDepth + 1, (uint)(cPip + anRoll[nMoveDepth]), anMoves, board);

            // Undo
            UndoSubMove(ref board, in undo);

            fUsed = true;
        }

        return !fUsed;
    }

    private static void SaveMoves(MoveList ml, int cMoves, uint cPip, Span<int> anMoves, Board board)
    {
        uint cMovesU = (uint)cMoves;

        if (cMovesU < ml.MaxMoves || cPip < ml.MaxPips)
            return;

        if (cMovesU > ml.MaxMoves || cPip > ml.MaxPips)
            ml.Moves.Clear();

        ml.MaxMoves = cMovesU;
        ml.MaxPips = cPip;

        var key = PositionId.ToKey(board);

        for (int idx = 0; idx < ml.Moves.Count; idx++)
        {
            var existing = ml.Moves[idx];
            if (existing.Key.Equals(key))
            {
                if (cMovesU > existing.SubMoveCount || cPip > existing.Pips)
                {
                    var updated = new Move();
                    CopyAnMoves(anMoves, cMoves, updated);
                    updated.Key = key;
                    updated.SubMoveCount = cMovesU;
                    updated.Pips = cPip;
                    ml.Moves[idx] = updated;
                }
                return;
            }
        }

        var move = new Move();
        CopyAnMoves(anMoves, cMoves, move);
        move.Key = key;
        move.SubMoveCount = cMovesU;
        move.Pips = cPip;
        ml.Moves.Add(move);
    }

    private static void CopyAnMoves(Span<int> src, int cMoves, Move dest)
    {
        for (int i = 0; i < cMoves * 2; i++)
            dest.AnMove[i] = src[i];
        for (int i = cMoves * 2; i < 8; i++)
            dest.AnMove[i] = -1;
    }
}
