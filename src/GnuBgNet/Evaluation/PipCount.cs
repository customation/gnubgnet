// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2019 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Ported from eval.c (PipCount, KleinmanCount, KeithCount, IsightCount, ThorpCount)

namespace GnuBgNet.Evaluation;

/// <summary>
/// Pip counting and race formula calculations.
/// Port of PipCount/KleinmanCount/KeithCount/IsightCount/ThorpCount from eval.c.
/// </summary>
public static class PipCount
{
    /// <summary>
    /// Count total pips for each side.
    /// Returns (playerPips, opponentPips) where player = board.Player (on roll).
    /// Port of PipCount() from eval.c.
    /// </summary>
    public static (int Player, int Opponent) Count(Board board)
    {
        int pip0 = 0, pip1 = 0;
        for (int i = 0; i < 25; i++)
        {
            pip0 += (int)board.Opponent[i] * (i + 1);
            pip1 += (int)board.Player[i] * (i + 1);
        }
        // In gnubg: anBoard[0] = opponent, anBoard[1] = player
        // anPips[0] = opponent pips, anPips[1] = player pips
        return (Player: pip1, Opponent: pip0);
    }

    /// <summary>
    /// Kleinman race formula: probability of winning based on pip counts.
    /// Port of KleinmanCount() from eval.c.
    /// </summary>
    public static float KleinmanCount(int pipOnRoll, int pipNotOnRoll)
    {
        int diff = pipNotOnRoll - pipOnRoll;
        int sum = pipNotOnRoll + pipOnRoll;

        if (sum > 4)
        {
            float rK = (diff + 4) / (2.0f * MathF.Sqrt(sum - 4));
            return 0.5f * (1.0f + Erf(rK));
        }

        return 0.0f;
    }

    /// <summary>
    /// Keith race formula: adjusted pip count with wastage.
    /// Returns (playerCount, opponentCount).
    /// Port of KeithCount() from eval.c.
    /// </summary>
    public static (int Player, int Opponent) KeithCount(Board board)
    {
        var (playerPips, opponentPips) = Count(board);
        int[] pn = new int[2];

        // side 0 = opponent (anBoard[0]), side 1 = player (anBoard[1])
        for (int side = 0; side < 2; side++)
        {
            uint[] b = side == 0 ? board.Opponent : board.Player;
            pn[side] = side == 0 ? opponentPips : playerPips;
            pn[side] += (Math.Max(1, (int)b[0]) - 1) * 2;
            pn[side] += Math.Max(1, (int)b[1]) - 1;
            pn[side] += Math.Max(3, (int)b[2]) - 3;
            for (int x = 3; x < 6; x++)
                if (b[x] == 0)
                    pn[side]++;
        }

        return (Player: pn[1], Opponent: pn[0]);
    }

    /// <summary>
    /// Isight race formula: adjusted pip count with crossovers and men left.
    /// Returns (playerCount, opponentCount).
    /// Port of IsightCount() from eval.c.
    /// </summary>
    public static (int Player, int Opponent) IsightCount(Board board)
    {
        var (playerPips, opponentPips) = Count(board);
        int[] pn = new int[2];
        int[] menLeft = new int[2];
        int[] crossOver = new int[2];

        for (int x = 0; x < 25; x++)
        {
            menLeft[0] += (int)board.Opponent[x];
            menLeft[1] += (int)board.Player[x];
            crossOver[0] += (int)board.Opponent[x] * (x / 6);
            crossOver[1] += (int)board.Player[x] * (x / 6);
        }

        for (int side = 0; side < 2; side++)
        {
            uint[] b = side == 0 ? board.Opponent : board.Player;
            pn[side] = side == 0 ? opponentPips : playerPips;
            if (menLeft[side] > menLeft[1 - side])
                pn[side] += menLeft[side] - menLeft[1 - side];
            pn[side] += (Math.Max(2, (int)b[0]) - 2) * 2;
            pn[side] += Math.Max(2, (int)b[1]) - 2;
            pn[side] += Math.Max(3, (int)b[2]) - 3;
            uint[] oppB = side == 0 ? board.Player : board.Opponent;
            for (int x = 3; x < 6; x++)
                if (b[x] == 0 && oppB[x] != 0)
                    pn[side]++;
            if (crossOver[side] > crossOver[1 - side])
                pn[side] += crossOver[side] - crossOver[1 - side];
        }

        return (Player: pn[1], Opponent: pn[0]);
    }

    /// <summary>
    /// Thorp race formula.
    /// Returns (leaderCount, leaderAdjusted, trailerCount).
    /// Port of ThorpCount() from eval.c.
    /// </summary>
    public static (int Leader, float Adjusted, int Trailer) ThorpCount(Board board)
    {
        var (playerPips, opponentPips) = Count(board);
        int[] menLeft = new int[2];
        int[] covered = new int[2];

        for (int x = 0; x < 25; x++)
        {
            menLeft[0] += (int)board.Opponent[x];
            menLeft[1] += (int)board.Player[x];
        }

        for (int x = 0; x < 6; x++)
        {
            if (board.Opponent[x] > 0) covered[0]++;
            if (board.Player[x] > 0) covered[1]++;
        }

        // In C: pnLeader = anPips[1] (player), pnTrailer = anPips[0] (opponent)
        int leader = playerPips + 2 * menLeft[1] + (int)board.Player[0] - covered[1];
        float adjusted = leader > 30 ? leader * 1.1f : leader;
        int trailer = opponentPips + 2 * menLeft[0] + (int)board.Opponent[0] - covered[0];

        return (Leader: leader, Adjusted: adjusted, Trailer: trailer);
    }

    /// <summary>
    /// Error function approximation using Abramowitz and Stegun formula 7.1.26.
    /// Maximum error: 1.5e-7.
    /// </summary>
    private static float Erf(float x)
    {
        float sign = x < 0 ? -1.0f : 1.0f;
        x = MathF.Abs(x);

        const float a1 = 0.254829592f;
        const float a2 = -0.284496736f;
        const float a3 = 1.421413741f;
        const float a4 = -1.453152027f;
        const float a5 = 1.061405429f;
        const float p = 0.3275911f;

        float t = 1.0f / (1.0f + p * x);
        float y = 1.0f - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * MathF.Exp(-x * x);

        return sign * y;
    }
}
