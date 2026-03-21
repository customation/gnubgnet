// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2019 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Port of QuasiRandomSeed / dicePerms from rollout.c

namespace GnuBgNet.Random;

/// <summary>
/// Quasi-random dice permutation tables for rollout variance reduction.
/// Port of QuasiRandomSeed() and aaanPermutation from rollout.c.
/// Uses seed-based Fisher-Yates shuffles so that each trial gets a
/// different but deterministic permutation of all 36 dice outcomes.
/// </summary>
public sealed class DicePermutations
{
    private const int QrLen = 6;    // dice pair depth
    private const int NumRolls = 36; // 6×6

    // aaanPermutation[diePairIdx][turnDepth][rollIdx] → encoded roll (0..35)
    private readonly int[,,] _perms = new int[QrLen, QrLen, NumRolls];

    public DicePermutations(uint seed)
    {
        // Use a simple LCG seeded with the given seed for shuffle randomness.
        // gnubg uses ISAAC; we use a deterministic LCG that's fast and sufficient
        // for permutation generation. The important property is determinism + coverage.
        uint state = seed;

        for (int i = 0; i < QrLen; i++)
        {
            for (int j = i; j < QrLen; j++)
            {
                // Initialize identity permutation
                for (int k = 0; k < NumRolls; k++)
                    _perms[i, j, k] = k;

                // Fisher-Yates shuffle
                for (int k = 0; k < NumRolls - 1; k++)
                {
                    state = NextLcg(state);
                    int r = k + (int)(state % (uint)(NumRolls - k));
                    (_perms[i, j, k], _perms[i, j, r]) = (_perms[i, j, r], _perms[i, j, k]);
                }
            }
        }
    }

    /// <summary>
    /// Get the dice roll for a given game index and turn.
    /// Returns (d0, d1) where d0 >= d1, encoded from the permutation table.
    /// Port of RolloutDice() from rollout.c.
    /// </summary>
    public (int D0, int D1) GetRoll(int gameIndex, int turn, ref int skip, bool skipDoubles)
    {
        if (turn == 0)
        {
            // First roll: single-level permutation lookup
            int j;
            do
            {
                j = _perms[0, 0, (gameIndex + skip) % NumRolls];
                skip++;
            } while (skipDoubles && (j / 6) == (j % 6)); // skip doubles for initial roll

            int d0 = j / 6 + 1;
            int d1 = j % 6 + 1;
            return d0 >= d1 ? (d0, d1) : (d1, d0);
        }
        else
        {
            // Subsequent turns: multi-level permutation (hierarchical quasi-random)
            int depth = Math.Min(turn, QrLen - 1);
            int j = 0;
            int k = 1;
            for (int i = 0; i <= depth; i++)
            {
                j = _perms[i, Math.Min(turn, QrLen - 1),
                    ((gameIndex + skip) / k + j) % NumRolls];
                k *= NumRolls;
            }

            int d0 = j / 6 + 1;
            int d1 = j % 6 + 1;
            return d0 >= d1 ? (d0, d1) : (d1, d0);
        }
    }

    private static uint NextLcg(uint state)
    {
        return state * 1103515245U + 12345U;
    }
}
