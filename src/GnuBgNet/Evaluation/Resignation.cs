// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2019 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Port of getResignation() and getResignEquities() from rollout.c

using GnuBgNet.MatchEquity;

namespace GnuBgNet.Evaluation;

/// <summary>
/// Result of resignation analysis.
/// </summary>
public readonly record struct ResignationResult(
    int RecommendedLevel,
    double EquityBeforeResign,
    double EquityAfterResign);

/// <summary>
/// Resignation analysis logic.
/// Port of getResignation() and getResignEquities() from rollout.c.
/// </summary>
public static class Resignation
{
    /// <summary>
    /// Determine whether a player should resign and at what level.
    /// Port of getResignation() from rollout.c lines 1755-1797.
    /// Returns 0 (no resign), 1 (normal), 2 (gammon), or 3 (backgammon).
    /// </summary>
    public static int GetResignation(ReadOnlySpan<float> arResign, CubeInfo ci,
        MatchEquityTable? met = null)
    {
        float rPlay = Utility(arResign, ci, met);

        // Construct baseline: all-loss scenario
        float[] ar = [0.0f, 0.0f, 0.0f, 1.0f, 1.0f];

        // Check backgammon resign
        if (arResign[Constants.OutputLoseBackgammon] > 0.0f)
        {
            float rBG = Utility(ar, ci, met);
            if (MathF.Abs(rBG - rPlay) < 1e-6f)
            {
                // Money game with Jacoby and centered cube → resign normal
                if (ci.MatchTo == 0 && ci.Jacoby && ci.CubeOwner == -1)
                    return 1;
                return 3; // resign backgammon
            }
        }

        // Check gammon resign
        if (arResign[Constants.OutputLoseGammon] > 0.0f)
        {
            ar[Constants.OutputLoseBackgammon] = 0.0f;
            float rG = Utility(ar, ci, met);
            if (MathF.Abs(rG - rPlay) < 1e-6f)
            {
                if (ci.MatchTo == 0 && ci.Jacoby && ci.CubeOwner == -1)
                    return 1;
                return 2; // resign gammon
            }
        }

        // Check normal resign
        ar[Constants.OutputLoseGammon] = 0.0f;
        ar[Constants.OutputLoseBackgammon] = 0.0f;
        float rNormal = Utility(ar, ci, met);
        if (MathF.Abs(rNormal - rPlay) < 1e-6f)
            return 1;

        return 0; // don't resign
    }

    /// <summary>
    /// Calculate equity before and after resignation at a specific level.
    /// Port of getResignEquities() from rollout.c lines 1801-1815.
    /// </summary>
    public static void GetResignEquities(ReadOnlySpan<float> arResign, CubeInfo ci,
        int nResigned, out float equityBefore, out float equityAfter,
        MatchEquityTable? met = null)
    {
        equityBefore = Utility(arResign, ci, met);

        // Construct the resignation outcome
        float[] ar = [0.0f, 0.0f, 0.0f, 0.0f, 0.0f];

        if (nResigned > 1)
            ar[Constants.OutputLoseGammon] = 1.0f;
        if (nResigned > 2)
            ar[Constants.OutputLoseBackgammon] = 1.0f;

        equityAfter = Utility(ar, ci, met);
    }

    /// <summary>
    /// Check if an opponent's resignation should be accepted.
    /// Port of check_resigns() logic from play.c.
    /// Returns the acceptable resignation level (1-3) or 0 if resignation should be rejected.
    /// </summary>
    public static int CheckResignation(ReadOnlySpan<float> arResign, CubeInfo ci,
        int nResigned, MatchEquityTable? met = null, float maxCost = 0.05f)
    {
        float equityBefore = Utility(arResign, ci, met);

        for (int level = nResigned; level >= 1; level--)
        {
            GetResignEquities(arResign, ci, level, out _, out float eqAfter, met);

            float cost = equityBefore - eqAfter;
            if (cost <= maxCost)
                return level;
        }

        return 0; // reject resignation
    }

    /// <summary>
    /// Compute utility (equity) from output probabilities.
    /// Port of Utility() from eval.c lines 2445-2471.
    /// For money game: equity = 2*P(win) - 1 + gammon/backgammon adjustments
    /// For match play: uses gammon prices from CubeInfo.
    /// </summary>
    internal static float Utility(ReadOnlySpan<float> output, CubeInfo ci,
        MatchEquityTable? met = null)
    {
        float eq = output[Constants.OutputWin] * 2.0f - 1.0f;

        if (ci.MatchTo == 0)
        {
            // Money game
            eq += output[Constants.OutputWinGammon]
                + output[Constants.OutputWinBackgammon]
                - output[Constants.OutputLoseGammon]
                - output[Constants.OutputLoseBackgammon];

            if (ci.Jacoby && ci.CubeOwner == -1)
            {
                // Jacoby rule: gammons/backgammons don't count with centered cube
                eq = output[Constants.OutputWin] * 2.0f - 1.0f;
            }
        }
        else if (met != null)
        {
            // Match play: use gammon prices
            CubeDecision.SetGammonPrices(ci, met);
            eq += output[Constants.OutputWinGammon] * ci.GammonPrice[ci.Move]
                - output[Constants.OutputLoseGammon] * ci.GammonPrice[1 - ci.Move]
                + output[Constants.OutputWinBackgammon] * ci.GammonPrice[2 + ci.Move]
                - output[Constants.OutputLoseBackgammon] * ci.GammonPrice[2 + (1 - ci.Move)];
        }
        else
        {
            // No MET: use simple money-style equity
            eq += output[Constants.OutputWinGammon]
                + output[Constants.OutputWinBackgammon]
                - output[Constants.OutputLoseGammon]
                - output[Constants.OutputLoseBackgammon];
        }

        return eq;
    }
}
