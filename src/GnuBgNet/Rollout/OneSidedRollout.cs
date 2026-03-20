// Copyright (C) 2002-2003 Joseph Heled <joseph@gnubg.org>
// Copyright (C) 2003-2019 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Port of osr.c (raceProbs, rollOSR, one-sided rollout)

using GnuBgNet.Bearoff;
using GnuBgNet.Encoding;
using GnuBgNet.MoveGeneration;
using GnuBgNet.Random;

namespace GnuBgNet.Rollout;

/// <summary>
/// One-sided rollout: fast race evaluation using Monte Carlo simulation
/// combined with bearoff database lookups. Used when a position is a race
/// but outside the bearoff database range.
/// Port of raceProbs / rollOSR from osr.c.
/// </summary>
public sealed class OneSidedRollout
{
    private readonly BearoffDatabase? _osBearoff;

    public OneSidedRollout(BearoffDatabase? osBearoff)
    {
        _osBearoff = osBearoff;
    }

    /// <summary>
    /// Compute race probabilities using one-sided rollout + bearoff convolution.
    /// Returns 5 output probabilities (win, wg, wbg, lg, lbg) and average rolls
    /// for each side.
    /// Port of raceProbs() from osr.c.
    /// </summary>
    public void RaceProbs(Board board, uint nGames, Span<float> output, out float muPlayer, out float muOpponent)
    {
        muPlayer = 0;
        muOpponent = 0;

        // Get per-side distribution: P(bear off in exactly i rolls)
        Span<float> distPlayer = stackalloc float[32];
        Span<float> distOpponent = stackalloc float[32];

        SimulateSide(board.Player, nGames, distPlayer, out muPlayer);
        SimulateSide(board.Opponent, nGames, distOpponent, out muOpponent);

        // Combine distributions to get win/gammon/backgammon probabilities
        // P(win) = sum over i: P_player(i) * P_opponent_not_finished(i)
        // Player moves first (on roll advantage)
        float pWin = 0;
        float pOppStillOn = 1.0f;
        for (int i = 1; i < 32; i++)
        {
            pWin += distPlayer[i] * pOppStillOn;
            pOppStillOn -= distOpponent[i];
        }

        // Gammon: opponent has all 15 checkers = never bore off
        // Approximate: P(gammon) ≈ P(win) × P(opponent still has 15)
        // This is simplified; full gnubg uses gammon distribution from bearoff DB
        float pWinGammon = 0;
        float pLoseGammon = 0;

        if (_osBearoff is { Gammon: true })
        {
            // Use bearoff gammon distributions if available
            Span<float> gammonPlayer = stackalloc float[32];
            Span<float> gammonOpponent = stackalloc float[32];

            uint posPlayer = PositionId.PositionBearoff(board.Player, _osBearoff.Points, _osBearoff.Chequers);
            uint posOpponent = PositionId.PositionBearoff(board.Opponent, _osBearoff.Points, _osBearoff.Chequers);

            if (posPlayer < _osBearoff.NumPositions && posOpponent < _osBearoff.NumPositions)
            {
                _osBearoff.GetGammonDistribution(posPlayer, gammonPlayer);
                _osBearoff.GetGammonDistribution(posOpponent, gammonOpponent);

                // P(win gammon) = sum P_player(i) * P_opponent_gammon_still_on(i)
                float pOppGammonOn = 1.0f;
                for (int i = 1; i < 32; i++)
                {
                    pWinGammon += distPlayer[i] * pOppGammonOn;
                    pOppGammonOn -= gammonOpponent[i];
                }

                float pPlayerGammonOn = 1.0f;
                for (int i = 1; i < 32; i++)
                {
                    pLoseGammon += distOpponent[i] * pPlayerGammonOn;
                    pPlayerGammonOn -= gammonPlayer[i];
                }
            }
        }

        output[Constants.OutputWin] = Math.Clamp(pWin, 0, 1);
        output[Constants.OutputWinGammon] = Math.Clamp(pWinGammon, 0, 1);
        output[Constants.OutputWinBackgammon] = 0; // race positions rarely have backgammons
        output[Constants.OutputLoseGammon] = Math.Clamp(pLoseGammon, 0, 1);
        output[Constants.OutputLoseBackgammon] = 0;
    }

    /// <summary>
    /// Simulate one side's bearing off to get distribution P(off in i rolls).
    /// If all checkers are in home board and within bearoff DB range, use DB lookup.
    /// Otherwise, do Monte Carlo simulation.
    /// </summary>
    private void SimulateSide(uint[] side, uint nGames, Span<float> dist, out float mu)
    {
        mu = 0;
        for (int i = 0; i < 32; i++) dist[i] = 0;

        // Check if all checkers are in home board (points 0-5)
        bool allHome = true;
        for (int i = 6; i < 25; i++)
        {
            if (side[i] > 0) { allHome = false; break; }
        }

        if (allHome && _osBearoff != null)
        {
            // Use bearoff database directly
            uint posId = PositionId.PositionBearoff(side, _osBearoff.Points, _osBearoff.Chequers);
            if (posId < _osBearoff.NumPositions)
            {
                _osBearoff.GetDistribution(posId, dist);
                for (int i = 0; i < 32; i++)
                    mu += i * dist[i];
                return;
            }
        }

        // Monte Carlo simulation: play out until all checkers are home,
        // then look up bearoff distribution for the final position
        uint[] counts = new uint[32];
        float[] boDist = new float[32];
        for (uint game = 0; game < nGames; game++)
        {
            var rng = new MersenneTwister(game);
            uint[] working = (uint[])side.Clone();
            int rolls = 0;

            // Roll until all checkers in home board
            while (rolls < 30)
            {
                bool outOfHome = false;
                for (int i = 6; i < 25; i++)
                {
                    if (working[i] > 0) { outOfHome = true; break; }
                }
                if (!outOfHome) break;

                (int d0, int d1) = rng.NextDiceRoll();
                PlayRollForSide(working, d0, d1);
                rolls++;
            }

            // Now all in home board: get bearoff distribution
            if (_osBearoff != null)
            {
                uint posId = PositionId.PositionBearoff(working, _osBearoff.Points, _osBearoff.Chequers);
                if (posId < _osBearoff.NumPositions)
                {
                    _osBearoff.GetDistribution(posId, boDist);

                    for (int i = 0; i < 32; i++)
                    {
                        int totalRolls = rolls + i;
                        if (totalRolls < 32)
                            dist[totalRolls] += boDist[i];
                    }
                }
            }
            else
            {
                // No bearoff DB: just record the rolls to reach home
                if (rolls < 32) counts[rolls]++;
            }
        }

        // Normalize
        if (_osBearoff != null)
        {
            float total = 0;
            for (int i = 0; i < 32; i++) total += dist[i];
            if (total > 0)
            {
                for (int i = 0; i < 32; i++)
                    dist[i] /= total;
            }
        }
        else
        {
            for (int i = 0; i < 32; i++)
                dist[i] = (float)counts[i] / nGames;
        }

        for (int i = 0; i < 32; i++)
            mu += i * dist[i];
    }

    /// <summary>
    /// Play a dice roll for one side (move checkers towards bearing off).
    /// Simple heuristic: move from back first, prioritize entering home.
    /// Port of FindBestMoveOSR from osr.c (simplified).
    /// </summary>
    private static void PlayRollForSide(uint[] board, int d0, int d1)
    {
        int[] dice = d0 == d1 ? [d0, d0, d0, d0] : [d0, d1];

        foreach (int die in dice)
        {
            // Try to move from the farthest back point
            bool moved = false;

            // Check bar first
            if (board[24] > 0)
            {
                int target = 24 - die;
                if (target >= 0)
                {
                    board[24]--;
                    board[target]++;
                    continue;
                }
            }

            // Move from farthest back
            for (int from = 23; from >= 0; from--)
            {
                if (board[from] == 0) continue;
                int to = from - die;
                if (to >= 0)
                {
                    board[from]--;
                    board[to]++;
                    moved = true;
                    break;
                }
                else if (from < 6)
                {
                    // Bear off: can bear off if no checker farther back
                    bool canBearOff = true;
                    for (int k = from + 1; k < 6; k++)
                    {
                        if (board[k] > 0) { canBearOff = false; break; }
                    }
                    if (canBearOff || from == die - 1)
                    {
                        board[from]--;
                        moved = true;
                        break;
                    }
                }
            }

            if (!moved)
            {
                // Try bearing off the highest point
                for (int from = 5; from >= 0; from--)
                {
                    if (board[from] > 0 && from < die)
                    {
                        board[from]--;
                        break;
                    }
                }
            }
        }
    }
}
