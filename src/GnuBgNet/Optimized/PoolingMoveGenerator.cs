// SPDX-License-Identifier: GPL-3.0-or-later

using GnuBgNet.Encoding;
using GnuBgNet.MoveGeneration;

namespace GnuBgNet.Optimized;

/// <summary>
/// Move generator that pools Move and MoveList objects to avoid GC pressure.
/// Uses in-place apply/undo (like <see cref="UndoStackMoveGenerator"/>) and
/// thread-local object pools for Move/MoveList reuse.
/// </summary>
public sealed class PoolingMoveGenerator : IMoveGenerator
{
    public static readonly PoolingMoveGenerator Instance = new();

    [ThreadStatic] private static Stack<Move>? t_movePool;
    [ThreadStatic] private static Stack<MoveList>? t_listPool;

    public MoveList GenerateMoves(Board board, int n0, int n1)
    {
        var ml = RentList();
        GenerateMovesInto(ml, board, n0, n1);
        return ml;
    }

    public void GenerateMovesInto(MoveList ml, Board board, int n0, int n1)
    {
        // Return existing moves to pool before clearing
        ReturnMovesToPool(ml);
        ml.Moves.Clear();
        ml.MaxMoves = 0;
        ml.MaxPips = 0;

        Span<int> anMoves = stackalloc int[8];
        anMoves.Fill(-1);

        if (n0 == n1)
        {
            int[] anRoll = [n0, n0, n0, n0];
            GenerateMovesSub(ml, anRoll, 0, 23, 0, ref board, anMoves);
        }
        else
        {
            int[] anRoll1 = [n0, n1, 0, 0];
            GenerateMovesSub(ml, anRoll1, 0, 23, 0, ref board, anMoves);

            int[] anRoll2 = [n1, n0, 0, 0];
            GenerateMovesSub(ml, anRoll2, 0, 23, 0, ref board, anMoves);
        }
    }

    public Board ApplyMove(Board board, Move move)
        => MoveGenerator.ApplyMove(board, move);

    public Board ApplyMoveAndSwap(Board board, Move move)
        => MoveGenerator.ApplyMoveAndSwap(board, move);

    /// <summary>
    /// Return a MoveList and all its Move objects to the pool for reuse.
    /// </summary>
    public void ReturnMoves(MoveList ml)
    {
        ReturnMovesToPool(ml);
        ml.Moves.Clear();
        ml.MaxMoves = 0;
        ml.MaxPips = 0;
        ml.BestIndex = -1;
        ml.BestScore = 0;

        t_listPool ??= new Stack<MoveList>();
        t_listPool.Push(ml);
    }

    private static MoveList RentList()
    {
        t_listPool ??= new Stack<MoveList>();
        if (t_listPool.Count > 0)
        {
            var ml = t_listPool.Pop();
            ml.BestIndex = -1;
            ml.BestScore = 0;
            ml.MaxPips = 0;
            ml.MaxMoves = 0;
            ml.Moves.Clear();
            return ml;
        }
        return new MoveList();
    }

    private static Move RentMove()
    {
        t_movePool ??= new Stack<Move>();
        if (t_movePool.Count > 0)
        {
            var m = t_movePool.Pop();
            m.AnMove[0] = -1;
            m.AnMove[1] = -1;
            m.AnMove[2] = -1;
            m.AnMove[3] = -1;
            m.AnMove[4] = -1;
            m.AnMove[5] = -1;
            m.AnMove[6] = -1;
            m.AnMove[7] = -1;
            m.Key = default;
            m.SubMoveCount = 0;
            m.Pips = 0;
            m.Score = 0;
            m.Score2 = 0;
            return m;
        }
        return new Move();
    }

    private static void ReturnMovesToPool(MoveList ml)
    {
        t_movePool ??= new Stack<Move>();
        for (int i = 0; i < ml.Moves.Count; i++)
            t_movePool.Push(ml.Moves[i]);
    }

    // --- Undo-stack move generation (same as UndoStackMoveGenerator) ---

    private struct UndoInfo
    {
        public int Src;
        public int Dest;
        public bool WasHit;
    }

    private static UndoInfo ApplySubMoveInPlace(ref Board board, int iSrc, int nRoll)
    {
        int iDest = iSrc - nRoll;
        var undo = new UndoInfo { Src = iSrc, Dest = iDest, WasHit = false };

        board.Player[iSrc]--;

        if (iDest < 0)
            return undo;

        if (board.Opponent[23 - iDest] == 1)
        {
            board.Opponent[23 - iDest] = 0;
            board.Opponent[24]++;
            undo.WasHit = true;
        }

        board.Player[iDest]++;
        return undo;
    }

    private static void UndoSubMove(ref Board board, in UndoInfo undo)
    {
        board.Player[undo.Src]++;

        if (undo.Dest < 0)
            return;

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

        int nBack;
        for (nBack = 24; nBack > 0; nBack--)
            if (board.Player[nBack] > 0) break;

        return nBack <= 5 && (iSrc == nBack || iDest == -1);
    }

    private static bool GenerateMovesSub(MoveList ml, int[] anRoll, int nMoveDepth,
        int iPip, int cPip, ref Board board, Span<int> anMoves)
    {
        if (nMoveDepth > 3 || anRoll[nMoveDepth] == 0)
            return true;

        if (board.Player[24] > 0)
        {
            int entryPoint = anRoll[nMoveDepth] - 1;
            if (board.Opponent[entryPoint] >= 2)
                return true;

            anMoves[nMoveDepth * 2] = 24;
            anMoves[nMoveDepth * 2 + 1] = 24 - anRoll[nMoveDepth];

            var undo = ApplySubMoveInPlace(ref board, 24, anRoll[nMoveDepth]);

            if (GenerateMovesSub(ml, anRoll, nMoveDepth + 1, 23,
                    cPip + anRoll[nMoveDepth], ref board, anMoves))
                SaveMoves(ml, nMoveDepth + 1, (uint)(cPip + anRoll[nMoveDepth]), anMoves, board);

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

            var undo = ApplySubMoveInPlace(ref board, i, anRoll[nMoveDepth]);

            int nextIPip = (anRoll[0] == anRoll[1]) ? i : 23;

            if (GenerateMovesSub(ml, anRoll, nMoveDepth + 1, nextIPip,
                    cPip + anRoll[nMoveDepth], ref board, anMoves))
                SaveMoves(ml, nMoveDepth + 1, (uint)(cPip + anRoll[nMoveDepth]), anMoves, board);

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
        {
            // Return displaced moves to pool before clearing
            ReturnMovesToPool(ml);
            ml.Moves.Clear();
        }

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
                    var updated = RentMove();
                    CopyAnMoves(anMoves, cMoves, updated);
                    updated.Key = key;
                    updated.SubMoveCount = cMovesU;
                    updated.Pips = cPip;
                    // Return the displaced move to pool
                    t_movePool!.Push(existing);
                    ml.Moves[idx] = updated;
                }
                return;
            }
        }

        var move = RentMove();
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
