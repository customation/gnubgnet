// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2019 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Ported from eval.c (GenerateMoves, GenerateMovesSub, LegalMove, ApplySubMove, SaveMoves)

using GnuBgNet.Encoding;

namespace GnuBgNet.MoveGeneration;

/// <summary>
/// Generates all legal moves for a given board position and dice roll.
/// Port of GenerateMoves/GenerateMovesSub from eval.c.
/// </summary>
public static class MoveGenerator
{
    /// <summary>
    /// Generate all legal moves for the player (anBoard[1]) given dice n0, n1.
    /// Port of GenerateMoves() from eval.c.
    /// </summary>
    public static MoveList GenerateMoves(Board board, int n0, int n1)
    {
        var ml = new MoveList();
        GenerateMovesInto(ml, board, n0, n1);
        return ml;
    }

    /// <summary>
    /// Generate moves into an existing MoveList (clears it first).
    /// </summary>
    public static void GenerateMovesInto(MoveList ml, Board board, int n0, int n1)
    {
        ml.Moves.Clear();
        ml.BestIndex = -1;
        ml.BestScore = 0;
        ml.MaxPips = 0;
        ml.MaxMoves = 0;

        int[] anRoll = new int[4];
        anRoll[0] = n0;
        anRoll[1] = n1;
        anRoll[2] = anRoll[3] = (n0 == n1) ? n0 : 0;

        int[] anMoves = new int[8];

        GenerateMovesSub(ml, anRoll, 0, 23, 0, board, anMoves);

        if (anRoll[0] != anRoll[1])
        {
            (anRoll[0], anRoll[1]) = (anRoll[1], anRoll[0]);
            GenerateMovesSub(ml, anRoll, 0, 23, 0, board, anMoves);
        }
    }

    /// <summary>
    /// Apply a move to a board (for use by evaluator).
    /// Returns a new board with the move applied (sides NOT swapped).
    /// </summary>
    public static Board ApplyMove(Board board, Move move)
    {
        var newBoard = board.Clone();
        for (int i = 0; i < move.SubMoveCount; i++)
        {
            int src = move.AnMove[i * 2];
            int nRoll = src - move.AnMove[i * 2 + 1];
            ApplySubMove(newBoard, src, nRoll);
        }
        return newBoard;
    }

    /// <summary>
    /// Apply a move and swap sides, returning the board ready for the next player.
    /// Port of gnubgapi_apply_move from gnubgapi.c: applies the move then calls SwapSides,
    /// so the result has the next-to-move player in Player (anBoard[1]).
    /// </summary>
    public static Board ApplyMoveAndSwap(Board board, Move move)
    {
        return ApplyMove(board, move).Swapped();
    }

    /// <summary>
    /// Apply a raw move (from/to pairs in int[8]) and swap sides.
    /// Returns the board ready for the next player's evaluation.
    /// Port of gnubgapi_apply_move semantics from gnubgapi.c.
    /// </summary>
    public static Board ApplyMoveRawAndSwap(Board board, int[] anMove)
    {
        return ApplyMoveRaw(board, anMove).Swapped();
    }

    /// <summary>
    /// Apply a raw move (from/to pairs in int[8]) to a board.
    /// Returns a new board with the move applied (sides NOT swapped).
    /// Port of ApplyMove() from eval.c for use with MAT parsed moves.
    /// </summary>
    public static Board ApplyMoveRaw(Board board, int[] anMove)
    {
        var newBoard = board.Clone();
        for (int i = 0; i < 8; i += 2)
        {
            int src = anMove[i];
            int dest = anMove[i + 1];
            if (src < 0) break; // -1 terminated

            int nRoll;
            if (dest < 0)
                nRoll = src + 1; // bear off: roll = src + 1 (e.g., point 5 → off = roll of 6)
            else
                nRoll = src - dest;

            ApplySubMove(newBoard, src, nRoll);
        }
        return newBoard;
    }

    private static bool GenerateMovesSub(MoveList ml, int[] anRoll, int nMoveDepth,
        int iPip, int cPip, Board board, int[] anMoves)
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

            var newBoard = board.Clone();
            ApplySubMove(newBoard, 24, anRoll[nMoveDepth]);

            if (GenerateMovesSub(ml, anRoll, nMoveDepth + 1, 23,
                    cPip + anRoll[nMoveDepth], newBoard, anMoves))
                SaveMoves(ml, nMoveDepth + 1, (uint)(cPip + anRoll[nMoveDepth]), anMoves, newBoard);

            return false; // partial not supported, always save complete
        }

        bool fUsed = false;
        for (int i = iPip; i >= 0; i--)
        {
            if (board.Player[i] == 0 || !LegalMove(board, i, anRoll[nMoveDepth]))
                continue;

            anMoves[nMoveDepth * 2] = i;
            anMoves[nMoveDepth * 2 + 1] = i - anRoll[nMoveDepth];

            var newBoard = board.Clone();
            ApplySubMove(newBoard, i, anRoll[nMoveDepth]);

            int nextIPip = (anRoll[0] == anRoll[1]) ? i : 23;

            if (GenerateMovesSub(ml, anRoll, nMoveDepth + 1, nextIPip,
                    cPip + anRoll[nMoveDepth], newBoard, anMoves))
                SaveMoves(ml, nMoveDepth + 1, (uint)(cPip + anRoll[nMoveDepth]), anMoves, newBoard);

            fUsed = true;
        }

        return !fUsed;
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

    private static void ApplySubMove(Board board, int iSrc, int nRoll)
    {
        int iDest = iSrc - nRoll;
        board.Player[iSrc]--;

        if (iDest < 0)
            return; // borne off

        if (board.Opponent[23 - iDest] > 0)
        {
            if (board.Opponent[23 - iDest] == 1)
            {
                // Hit
                board.Opponent[23 - iDest] = 0;
                board.Opponent[24]++;
            }
            // If > 1, this shouldn't happen (LegalMove prevents it)
        }
        board.Player[iDest]++;
    }

    private static void SaveMoves(MoveList ml, int cMoves, uint cPip, int[] anMoves, Board board)
    {
        uint cMovesU = (uint)cMoves;

        // Enforce max moves/pips rule
        if (cMovesU < ml.MaxMoves || cPip < ml.MaxPips)
            return;

        if (cMovesU > ml.MaxMoves || cPip > ml.MaxPips)
            ml.Moves.Clear();

        ml.MaxMoves = cMovesU;
        ml.MaxPips = cPip;

        // Generate position key for dedup
        var key = PositionId.ToKey(board);

        // Check for duplicate position
        for (int idx = 0; idx < ml.Moves.Count; idx++)
        {
            var existing = ml.Moves[idx];
            if (existing.Key.Equals(key))
            {
                // Update if this move uses more sub-moves or pips
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

        // New move
        var move = new Move();
        CopyAnMoves(anMoves, cMoves, move);
        move.Key = key;
        move.SubMoveCount = cMovesU;
        move.Pips = cPip;
        ml.Moves.Add(move);
    }

    private static void CopyAnMoves(int[] src, int cMoves, Move dest)
    {
        for (int i = 0; i < cMoves * 2; i++)
            dest.AnMove[i] = src[i] > -1 ? src[i] : -1;
        if (cMoves < 4)
            dest.AnMove[cMoves * 2] = -1;
    }
}
