// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2019 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Ported from eval.c (Cl2CfMoney, Cl2CfMatch, MoneyLive, GetDPEq, FindCubeDecision)

using GnuBgNet.MatchEquity;

namespace GnuBgNet.Evaluation;

/// <summary>
/// Result of a cube decision analysis.
/// </summary>
public readonly record struct CubeDecisionResult(
    CubeAction Action,
    double OptimalEquity,
    double NoDoubleEquity,
    double DoubleTakeEquity,
    double DoublePassEquity);

/// <summary>
/// Full cube decision classification.
/// Port of cubedecision enum from eval.h.
/// </summary>
public enum CubeAction
{
    DoubleTake = 0,
    DoublePass = 1,
    NoDoubleTake = 2,
    TooGoodTake = 3,
    TooGoodPass = 4,
    DoubleBeaver = 5,
    NoDoubleBeaver = 6,
    RedoubleTake = 7,
    RedoublePass = 8,
    NoRedoubleTake = 9,
    TooGoodRedoubleTake = 10,
    TooGoodRedoublePass = 11,
    NoRedoubleBeaver = 12,
    NoDoubleDeadCube = 13,
    NoRedoubleDeadCube = 14,
    NotAvailable = 15,
    OptionalDoubleTake = 16,
    OptionalRedoubleTake = 17,
    OptionalDoubleBeaver = 18,
    OptionalDoublePass = 19,
    OptionalRedoublePass = 20,
}

/// <summary>Indices into the arDouble array.</summary>
internal static class CubeOutputIndex
{
    public const int Optimal = 0;
    public const int NoDouble = 1;
    public const int Take = 2;
    public const int Drop = 3;
}

/// <summary>
/// Cube decision logic for money and match play.
/// Port of Cl2CfMoney, Cl2CfMatch*, MoneyLive, GetDPEq, FindCubeDecision from eval.c.
/// </summary>
public static class CubeDecision
{
    private const float DefaultCubeEfficiency = 0.68f;

    /// <summary>
    /// Analyse the cube decision for a position.
    /// Dispatches to money or match play depending on CubeInfo.MatchTo.
    /// </summary>
    public static CubeDecisionResult Analyse(
        ReadOnlySpan<float> output,
        CubeInfo ci,
        MatchEquityTable met,
        float cubeEfficiency = DefaultCubeEfficiency)
    {
        if (ci.MatchTo == 0)
            return AnalyseMoney(output, ci, cubeEfficiency);
        else
            return AnalyseMatch(output, ci, met, cubeEfficiency);
    }

    /// <summary>
    /// Analyse the cube decision for a money game position (simple overload).
    /// </summary>
    public static CubeDecisionResult AnalyseMoney(
        ReadOnlySpan<float> output,
        int cubeOwner = -1,
        bool jacoby = false)
    {
        var ci = new CubeInfo
        {
            CubeOwner = cubeOwner,
            Move = 0,
            MatchTo = 0,
            Jacoby = jacoby,
            Beavers = true,
        };
        return AnalyseMoney(output, ci);
    }

    /// <summary>
    /// Analyse the cube decision for a money game position with full CubeInfo.
    /// </summary>
    public static CubeDecisionResult AnalyseMoney(
        ReadOnlySpan<float> output,
        CubeInfo ci,
        float cubeEfficiency = DefaultCubeEfficiency)
    {
        float cubeEff = cubeEfficiency;

        // No-double: cubeful equity keeping the cube as-is
        float noDoubleEquity = CubelessToCubefulMoney(
            output, ci.CubeOwner, ci.Jacoby, cubeEff, ci.Move);

        // Double-take: after doubling, opponent of doubler owns cube.
        // The doubler is ci.Move, so opponent is 1 - ci.Move.
        int oppOwner = 1 - ci.Move;
        float doubleTakeEquity = 2.0f * CubelessToCubefulMoney(
            output, oppOwner, ci.Jacoby, cubeEff, ci.Move);

        // Double-pass: +1 normalized
        float doublePassEquity = 1.0f;

        // Build aarOutput for FindBestCubeDecision
        float[][] aarOutput = new float[2][];
        aarOutput[0] = new float[Constants.NumRolloutOutputs];
        aarOutput[1] = new float[Constants.NumRolloutOutputs];
        for (int i = 0; i < Constants.NumOutputs; i++)
            aarOutput[0][i] = output[i];
        aarOutput[0][Constants.OutputEquity] = noDoubleEquity;
        aarOutput[0][Constants.OutputCubefulEquity] = noDoubleEquity;
        aarOutput[1][Constants.OutputEquity] = doubleTakeEquity;
        aarOutput[1][Constants.OutputCubefulEquity] = doubleTakeEquity;

        float[] arDouble = new float[4];
        arDouble[CubeOutputIndex.Drop] = doublePassEquity;
        arDouble[CubeOutputIndex.NoDouble] = noDoubleEquity;
        arDouble[CubeOutputIndex.Take] = doubleTakeEquity;

        var action = FindBestCubeDecision(arDouble, aarOutput, ci);

        return new CubeDecisionResult(
            Action: action,
            OptimalEquity: arDouble[CubeOutputIndex.Optimal],
            NoDoubleEquity: noDoubleEquity,
            DoubleTakeEquity: doubleTakeEquity,
            DoublePassEquity: doublePassEquity);
    }

    /// <summary>
    /// Analyse the cube decision for match play.
    /// Port of GeneralCubeDecisionE + FindCubeDecision from eval.c.
    /// </summary>
    public static CubeDecisionResult AnalyseMatch(
        ReadOnlySpan<float> output,
        CubeInfo ci,
        MatchEquityTable met,
        float cubeEfficiency = DefaultCubeEfficiency)
    {
        bool cubeDead = IsCubeDead(ci);
        float cubeEff = cubeEfficiency;

        // No-double: cubeful equity with current cube state
        float ndMwc = cubeDead
            ? Eq2Mwc(UtilityMatch(output, ci, met), ci, met)
            : Cl2CfMatch(output, ci, met, cubeEff);

        // Double-pass: MWC for cashing
        float dpMwc = GetDoublePassMwc(ci, met);

        // Double-take: cubeful equity after opponent takes (opponent owns cube at 2x)
        var ciOppCube = new CubeInfo
        {
            Cube = ci.Cube * 2,
            CubeOwner = 1 - ci.Move,
            Move = ci.Move,
            MatchTo = ci.MatchTo,
            Score = (int[])ci.Score.Clone(),
            Crawford = ci.Crawford,
            Jacoby = ci.Jacoby,
            Beavers = ci.Beavers,
            Variation = ci.Variation,
        };
        SetGammonPrices(ciOppCube, met);

        float dtMwc = IsCubeDead(ciOppCube)
            ? Eq2Mwc(UtilityMatch(output, ciOppCube, met), ciOppCube, met)
            : Cl2CfMatch(output, ciOppCube, met, cubeEff);

        // Build aarOutput for FindBestCubeDecision (MWC values)
        float[][] aarOutput = new float[2][];
        aarOutput[0] = new float[Constants.NumRolloutOutputs];
        aarOutput[1] = new float[Constants.NumRolloutOutputs];
        for (int i = 0; i < Constants.NumOutputs; i++)
            aarOutput[0][i] = output[i];
        aarOutput[0][Constants.OutputCubefulEquity] = ndMwc;
        aarOutput[1][Constants.OutputCubefulEquity] = dtMwc;

        // FindCubeDecision converts MWC to normalized equity
        float[] arDouble = new float[4];
        arDouble[CubeOutputIndex.Drop] = dpMwc;
        arDouble[CubeOutputIndex.NoDouble] = ndMwc;
        arDouble[CubeOutputIndex.Take] = dtMwc;

        // Convert to normalized money equity for the decision
        arDouble[CubeOutputIndex.NoDouble] = Mwc2Eq(arDouble[CubeOutputIndex.NoDouble], ci, met);
        arDouble[CubeOutputIndex.Take] = Mwc2Eq(arDouble[CubeOutputIndex.Take], ci, met);
        arDouble[CubeOutputIndex.Drop] = Mwc2Eq(arDouble[CubeOutputIndex.Drop], ci, met);

        var action = FindBestCubeDecision(arDouble, aarOutput, ci);

        return new CubeDecisionResult(
            Action: action,
            OptimalEquity: arDouble[CubeOutputIndex.Optimal],
            NoDoubleEquity: arDouble[CubeOutputIndex.NoDouble],
            DoubleTakeEquity: arDouble[CubeOutputIndex.Take],
            DoublePassEquity: arDouble[CubeOutputIndex.Drop]);
    }

    /// <summary>
    /// Determine if the cube can be used and get the double/pass equity.
    /// Port of GetDPEq() from eval.c.
    /// Returns true if doubling is possible.
    /// </summary>
    public static bool GetDPEq(CubeInfo ci, MatchEquityTable? met, out float dpEquity)
    {
        if (ci.MatchTo == 0)
        {
            dpEquity = 1.0f;
            return ci.CubeOwner == -1 || ci.CubeOwner == ci.Move;
        }

        bool postCrawford = !ci.Crawford &&
            (ci.Score[0] == ci.MatchTo - 1 || ci.Score[1] == ci.MatchTo - 1);

        bool canDouble = !ci.Crawford &&
            ci.Score[ci.Move] + ci.Cube < ci.MatchTo &&
            !(postCrawford && ci.Score[ci.Move] == ci.MatchTo - 1) &&
            (ci.CubeOwner == -1 || ci.CubeOwner == ci.Move);

        dpEquity = met != null
            ? GetME(ci.Score[0], ci.Score[1], ci.MatchTo,
                ci.Move, ci.Cube, ci.Move, ci.Crawford, met)
            : 1.0f;

        return canDouble;
    }

    /// <summary>
    /// Find the best cube decision given equities and outputs.
    /// Port of FindBestCubeDecision() from eval.c.
    /// </summary>
    internal static CubeAction FindBestCubeDecision(
        float[] arDouble, float[][] aarOutput, CubeInfo ci)
    {
        // Check if cube is available
        bool canDouble = GetDPEq(ci, null, out _);
        if (!canDouble)
        {
            arDouble[CubeOutputIndex.Optimal] = arDouble[CubeOutputIndex.NoDouble];
            if (ci.MatchTo > 0 && (ci.CubeOwner < 0 || ci.CubeOwner == ci.Move))
                return ci.CubeOwner == -1 ? CubeAction.NoDoubleDeadCube : CubeAction.NoRedoubleDeadCube;
            return CubeAction.NotAvailable;
        }

        float nd = arDouble[CubeOutputIndex.NoDouble];
        float dt = arDouble[CubeOutputIndex.Take];
        float dp = arDouble[CubeOutputIndex.Drop];

        if (dt >= nd && dp >= nd)
        {
            // We have a double
            if (dp > dt)
            {
                // DP > DT >= ND: Double, take
                bool optional = IsOptional(dt, nd);
                arDouble[CubeOutputIndex.Optimal] = dt;

                if (ci.MatchTo == 0 && dt >= -2.0f && dt <= 0.0f && ci.Beavers)
                {
                    if (dt * 2.0f < nd)
                        return CubeAction.NoDoubleBeaver;
                    return optional ? CubeAction.OptionalDoubleBeaver : CubeAction.DoubleBeaver;
                }

                if (aarOutput[0][Constants.OutputWin] > 0.0f)
                {
                    if (optional)
                        return ci.CubeOwner == -1 ? CubeAction.OptionalDoubleTake : CubeAction.OptionalRedoubleTake;
                    return ci.CubeOwner == -1 ? CubeAction.DoubleTake : CubeAction.RedoubleTake;
                }

                return ci.CubeOwner == -1 ? CubeAction.NoDoubleTake : CubeAction.NoRedoubleTake;
            }
            else
            {
                // DT >= DP >= ND: Double, pass
                arDouble[CubeOutputIndex.Optimal] = dp;

                if (IsOptional(nd, dp) &&
                    aarOutput[0][Constants.OutputWinGammon] > 0.0f &&
                    (ci.MatchTo > 0 || ci.CubeOwner != -1 || !ci.Jacoby))
                    return ci.CubeOwner == -1 ? CubeAction.OptionalDoublePass : CubeAction.OptionalRedoublePass;

                return ci.CubeOwner == -1 ? CubeAction.DoublePass : CubeAction.RedoublePass;
            }
        }
        else
        {
            // No double: ND > DT or ND > DP
            arDouble[CubeOutputIndex.Optimal] = nd;

            if (nd > dt)
            {
                if (dt > dp)
                {
                    // ND > DT > DP: Too good, pass
                    if (aarOutput[0][Constants.OutputWinGammon] > 0.0f)
                        return ci.CubeOwner == -1 ? CubeAction.TooGoodPass : CubeAction.TooGoodRedoublePass;
                    return ci.CubeOwner == -1 ? CubeAction.DoublePass : CubeAction.RedoublePass;
                }
                else if (nd > dp)
                {
                    // ND > DP > DT: Too good, take
                    if (aarOutput[0][Constants.OutputWinGammon] > 0.0f)
                        return ci.CubeOwner == -1 ? CubeAction.TooGoodTake : CubeAction.TooGoodRedoubleTake;
                    return ci.CubeOwner == -1 ? CubeAction.NoDoubleTake : CubeAction.NoRedoubleTake;
                }
                else
                {
                    // DP > ND > DT: No double, beaver
                    if (dt >= -2.0f && dt <= 0.0f && ci.MatchTo == 0 && ci.Beavers)
                        return ci.CubeOwner == -1 ? CubeAction.NoDoubleBeaver : CubeAction.NoRedoubleBeaver;
                    return ci.CubeOwner == -1 ? CubeAction.NoDoubleTake : CubeAction.NoRedoubleTake;
                }
            }
            else
            {
                // DT >= ND > DP: Too good, pass
                if (aarOutput[0][Constants.OutputWinGammon] > 0.0f)
                    return ci.CubeOwner == -1 ? CubeAction.TooGoodPass : CubeAction.TooGoodRedoublePass;
                return ci.CubeOwner == -1 ? CubeAction.DoublePass : CubeAction.RedoublePass;
            }
        }
    }

    private static bool IsOptional(float r1, float r2)
    {
        return MathF.Abs(r1 - r2) <= 1.0e-5f;
    }

    /// <summary>
    /// Check if the cube is dead (cannot be used effectively).
    /// Returns true when cubeful evaluation is unnecessary.
    /// Port of !fDoCubeful() from eval.c.
    /// </summary>
    internal static bool IsCubeDead(CubeInfo ci)
    {
        if (ci.MatchTo == 0) return false;

        // Both players can win match with current cube value
        if (ci.Score[0] + ci.Cube >= ci.MatchTo && ci.Score[1] + ci.Cube >= ci.MatchTo)
            return true;

        // Score is -2,-2 (both 2-away)
        if (ci.Score[0] == ci.MatchTo - 2 && ci.Score[1] == ci.MatchTo - 2)
            return true;

        // Crawford game — cube is dead
        if (ci.Crawford)
            return true;

        return false;
    }

    /// <summary>
    /// Cubeless equity for match play using gammon prices.
    /// Port of Utility() for match play from eval.c.
    /// </summary>
    internal static float UtilityMatch(ReadOnlySpan<float> output, CubeInfo ci, MatchEquityTable met)
    {
        SetGammonPrices(ci, met);
        return output[Constants.OutputWin] * 2.0f - 1.0f
            + output[Constants.OutputWinGammon] * ci.GammonPrice[ci.Move]
            - output[Constants.OutputLoseGammon] * ci.GammonPrice[1 - ci.Move]
            + output[Constants.OutputWinBackgammon] * ci.GammonPrice[2 + ci.Move]
            - output[Constants.OutputLoseBackgammon] * ci.GammonPrice[2 + (1 - ci.Move)];
    }

    /// <summary>
    /// Set gammon prices in a CubeInfo based on the match equity table.
    /// The gammon price is how much a gammon win/loss is worth relative to a normal win/loss.
    /// </summary>
    internal static void SetGammonPrices(CubeInfo ci, MatchEquityTable met)
    {
        if (ci.MatchTo == 0)
        {
            // Money game: gammon = 1 point extra, backgammon = 2 points extra
            ci.GammonPrice[0] = ci.GammonPrice[1] = 1.0f;
            ci.GammonPrice[2] = ci.GammonPrice[3] = 1.0f;
            return;
        }

        // Port of getGammonPrice() from matchequity.c.
        // Single pass with player=0 computing all 4 prices:
        //   [0] = win gammon price, [1] = lose gammon price
        //   [2] = win BG price, [3] = lose BG price
        // Uses center formulation: gammon prices are "twice the usual value".
        int cube = ci.Cube;

        float rWin = GetME(ci.Score[0], ci.Score[1], ci.MatchTo,
            0, cube, 0, ci.Crawford, met);
        float rWinGammon = GetME(ci.Score[0], ci.Score[1], ci.MatchTo,
            0, 2 * cube, 0, ci.Crawford, met);
        float rWinBG = GetME(ci.Score[0], ci.Score[1], ci.MatchTo,
            0, 3 * cube, 0, ci.Crawford, met);
        float rLose = GetME(ci.Score[0], ci.Score[1], ci.MatchTo,
            0, cube, 1, ci.Crawford, met);
        float rLoseGammon = GetME(ci.Score[0], ci.Score[1], ci.MatchTo,
            0, 2 * cube, 1, ci.Crawford, met);
        float rLoseBG = GetME(ci.Score[0], ci.Score[1], ci.MatchTo,
            0, 3 * cube, 1, ci.Crawford, met);

        float rCenter = (rWin + rLose) / 2.0f;
        float halfRange = rWin - rCenter; // = (rWin - rLose) / 2

        if (MathF.Abs(halfRange) > 1e-7f)
        {
            ci.GammonPrice[0] = (rWinGammon - rCenter) / halfRange - 1.0f;
            ci.GammonPrice[1] = (rCenter - rLoseGammon) / halfRange - 1.0f;
            ci.GammonPrice[2] = (rWinBG - rCenter) / halfRange - (ci.GammonPrice[0] + 1.0f);
            ci.GammonPrice[3] = (rCenter - rLoseBG) / halfRange - (ci.GammonPrice[1] + 1.0f);
        }
        else
        {
            ci.GammonPrice[0] = ci.GammonPrice[1] = ci.GammonPrice[2] = ci.GammonPrice[3] = 0f;
        }

        // Correct numerical problems (same as C code)
        for (int i = 0; i < 4; i++)
            if (ci.GammonPrice[i] < 0.0f)
                ci.GammonPrice[i] = 0.0f;
    }

    /// <summary>
    /// Match equity lookup for specific outcome.
    /// Port of getME() from matchequity.c.
    /// </summary>
    internal static float GetME(int score0, int score1, int matchTo,
        int player, int points, int whoWins, bool crawford, MatchEquityTable met)
    {
        int n0 = matchTo - (score0 + (whoWins == 0 ? 0 : 1) * points) - 1;
        int n1 = matchTo - (score1 + (whoWins != 0 ? 0 : 1) * points) - 1;

        // Actually: whoWins=0 means player0 wins, whoWins=1 means player1 wins
        // n0 = matchTo - score0 - (if player1 wins, 0, else points) - 1
        // In C: n0 = nMatchTo - (nScore0 + (!fWhoWins) * nPoints) - 1
        //        n1 = nMatchTo - (nScore1 + fWhoWins * nPoints) - 1
        // Recompute correctly:
        n0 = matchTo - (score0 + (1 - whoWins) * points) - 1;
        n1 = matchTo - (score1 + whoWins * points) - 1;

        if (n0 < 0)
            return player != 0 ? 0.0f : 1.0f;
        if (n1 < 0)
            return player != 0 ? 1.0f : 0.0f;

        bool postCrawford = crawford ||
            (matchTo - score0 == 1) || (matchTo - score1 == 1);

        if (postCrawford)
        {
            if (n0 == 0)
                return player != 0
                    ? met.PostCrawford[1, Math.Min(n1, Constants.MaxScore - 1)]
                    : 1.0f - met.PostCrawford[1, Math.Min(n1, Constants.MaxScore - 1)];
            else
                return player != 0
                    ? 1.0f - met.PostCrawford[0, Math.Min(n0, Constants.MaxScore - 1)]
                    : met.PostCrawford[0, Math.Min(n0, Constants.MaxScore - 1)];
        }

        n0 = Math.Min(n0, Constants.MaxScore - 1);
        n1 = Math.Min(n1, Constants.MaxScore - 1);
        return player != 0 ? 1.0f - met.Met[n0, n1] : met.Met[n0, n1];
    }

    /// <summary>
    /// Get MWC for double/pass outcome.
    /// Port of GetDPEq() match play path from eval.c.
    /// </summary>
    private static float GetDoublePassMwc(CubeInfo ci, MatchEquityTable met)
    {
        return GetME(ci.Score[0], ci.Score[1], ci.MatchTo,
            ci.Move, ci.Cube, ci.Move, ci.Crawford, met);
    }

    /// <summary>
    /// Convert normalized equity to match winning chance.
    /// Port of eq2mwc() from eval.c.
    /// </summary>
    internal static float Eq2Mwc(float eq, CubeInfo ci, MatchEquityTable met)
    {
        float mwcWin = GetME(ci.Score[0], ci.Score[1], ci.MatchTo,
            ci.Move, ci.Cube, ci.Move, ci.Crawford, met);
        float mwcLose = GetME(ci.Score[0], ci.Score[1], ci.MatchTo,
            ci.Move, ci.Cube, 1 - ci.Move, ci.Crawford, met);

        return 0.5f * (eq * (mwcWin - mwcLose) + (mwcWin + mwcLose));
    }

    /// <summary>
    /// Convert match winning chance to normalized equity.
    /// Port of mwc2eq() from eval.c.
    /// </summary>
    internal static float Mwc2Eq(float mwc, CubeInfo ci, MatchEquityTable met)
    {
        float mwcWin = GetME(ci.Score[0], ci.Score[1], ci.MatchTo,
            ci.Move, ci.Cube, ci.Move, ci.Crawford, met);
        float mwcLose = GetME(ci.Score[0], ci.Score[1], ci.MatchTo,
            ci.Move, ci.Cube, 1 - ci.Move, ci.Crawford, met);

        float denom = mwcWin - mwcLose;
        if (Math.Abs(denom) < 1e-10f)
            return 0.0f;

        return (2.0f * mwc - (mwcWin + mwcLose)) / denom;
    }

    /// <summary>
    /// Cubeless-to-cubeful equity conversion for match play.
    /// Dispatches based on cube ownership.
    /// Port of Cl2CfMatch() from eval.c.
    /// </summary>
    internal static float Cl2CfMatch(
        ReadOnlySpan<float> output, CubeInfo ci, MatchEquityTable met, float cubeEfficiency)
    {
        // When the cube is dead (Crawford, DMP, both-can-win), cubeful = cubeless.
        // Port of the fDoCubeful() guard in Cl2CfMatch() from eval.c.
        if (IsCubeDead(ci))
            return Eq2Mwc(UtilityMatch(output, ci, met), ci, met);

        if (ci.CubeOwner == -1)
            return Cl2CfMatchCentered(output, ci, met, cubeEfficiency);
        else if (ci.CubeOwner == ci.Move)
            return Cl2CfMatchOwned(output, ci, met, cubeEfficiency);
        else
            return Cl2CfMatchUnavailable(output, ci, met, cubeEfficiency);
    }

    /// <summary>
    /// Centered cube match play cubeful equity.
    /// Port of Cl2CfMatchCentered() from eval.c.
    /// </summary>
    private static float Cl2CfMatchCentered(
        ReadOnlySpan<float> output, CubeInfo ci, MatchEquityTable met, float rCubeX)
    {
        ComputeGammonRatios(output, out float rG0, out float rBG0, out float rG1, out float rBG1);

        float mwcDead = Eq2Mwc(UtilityMatch(output, ci, met), ci, met);

        // Get live cube cash points
        float[] arCP = new float[2];
        GetPoints(output, ci, met, arCP);

        // Get MWC for basic outcomes at current cube level
        float[] p0 = new float[MET_DTLBP1 + 1];
        float[] p1 = new float[MET_DTLBP1 + 1];
        GetMEMultiple(ci.Score[0], ci.Score[1], ci.MatchTo,
            ci.Cube, -1, -1, ci.Crawford, met, p0, p1);

        float[] res = ci.Move == 0 ? p0 : p1;
        float mwcCash = res[MET_NDW];
        float mwcOppCash = res[MET_NDL];

        float rOppTG = 1.0f - arCP[1 - ci.Move];
        float rTG = arCP[ci.Move];

        float mwcLive;
        if (output[Constants.OutputWin] <= rOppTG)
        {
            float mwcLose = (1.0f - rG1 - rBG1) * res[MET_NDL]
                + rG1 * res[MET_NDLG]
                + rBG1 * res[MET_NDLB];
            mwcLive = rOppTG > 0.0f
                ? mwcLose + (mwcOppCash - mwcLose) * output[Constants.OutputWin] / rOppTG
                : mwcLose;
        }
        else if (output[Constants.OutputWin] < rTG)
        {
            mwcLive = mwcOppCash + (mwcCash - mwcOppCash)
                * (output[Constants.OutputWin] - rOppTG) / (rTG - rOppTG);
        }
        else
        {
            float mwcWin = (1.0f - rG0 - rBG0) * res[MET_NDW]
                + rG0 * res[MET_NDWG]
                + rBG0 * res[MET_NDWB];
            mwcLive = rTG < 1.0f
                ? mwcCash + (mwcWin - mwcCash) * (output[Constants.OutputWin] - rTG) / (1.0f - rTG)
                : mwcWin;
        }

        return mwcDead * (1.0f - rCubeX) + mwcLive * rCubeX;
    }

    /// <summary>
    /// Owned cube match play cubeful equity.
    /// Port of Cl2CfMatchOwned() from eval.c.
    /// </summary>
    private static float Cl2CfMatchOwned(
        ReadOnlySpan<float> output, CubeInfo ci, MatchEquityTable met, float rCubeX)
    {
        ComputeGammonRatios(output, out float rG0, out float rBG0, out float rG1, out float rBG1);

        float mwcDead = Eq2Mwc(UtilityMatch(output, ci, met), ci, met);

        float[] arCP = new float[2];
        GetPoints(output, ci, met, arCP);

        float[] p0 = new float[MET_DTLBP1 + 1];
        float[] p1 = new float[MET_DTLBP1 + 1];
        GetMEMultiple(ci.Score[0], ci.Score[1], ci.MatchTo,
            ci.Cube, -1, -1, ci.Crawford, met, p0, p1);

        float[] res = ci.Move == 0 ? p0 : p1;
        float mwcCash = res[MET_NDW];
        float rTG = arCP[ci.Move];

        float mwcLive;
        if (output[Constants.OutputWin] <= rTG)
        {
            float mwcLose = (1.0f - rG1 - rBG1) * res[MET_NDL]
                + rG1 * res[MET_NDLG]
                + rBG1 * res[MET_NDLB];
            mwcLive = rTG > 0.0f
                ? mwcLose + (mwcCash - mwcLose) * output[Constants.OutputWin] / rTG
                : mwcLose;
        }
        else
        {
            float mwcWin = (1.0f - rG0 - rBG0) * res[MET_NDW]
                + rG0 * res[MET_NDWG]
                + rBG0 * res[MET_NDWB];
            mwcLive = rTG < 1.0f
                ? mwcCash + (mwcWin - mwcCash) * (output[Constants.OutputWin] - rTG) / (1.0f - rTG)
                : mwcWin;
        }

        return mwcDead * (1.0f - rCubeX) + mwcLive * rCubeX;
    }

    /// <summary>
    /// Unavailable cube match play cubeful equity.
    /// Port of Cl2CfMatchUnavailable() from eval.c.
    /// </summary>
    private static float Cl2CfMatchUnavailable(
        ReadOnlySpan<float> output, CubeInfo ci, MatchEquityTable met, float rCubeX)
    {
        ComputeGammonRatios(output, out float rG0, out float rBG0, out float rG1, out float rBG1);

        float mwcDead = Eq2Mwc(UtilityMatch(output, ci, met), ci, met);

        float[] arCP = new float[2];
        GetPoints(output, ci, met, arCP);

        float[] p0 = new float[MET_DTLBP1 + 1];
        float[] p1 = new float[MET_DTLBP1 + 1];
        GetMEMultiple(ci.Score[0], ci.Score[1], ci.MatchTo,
            ci.Cube, -1, -1, ci.Crawford, met, p0, p1);

        float[] res = ci.Move == 0 ? p0 : p1;
        float mwcOppCash = res[MET_NDL];
        float rOppTG = 1.0f - arCP[1 - ci.Move];

        float mwcLive;
        if (output[Constants.OutputWin] <= rOppTG)
        {
            float mwcLose = (1.0f - rG1 - rBG1) * res[MET_NDL]
                + rG1 * res[MET_NDLG]
                + rBG1 * res[MET_NDLB];
            mwcLive = rOppTG > 0.0f
                ? mwcLose + (mwcOppCash - mwcLose) * output[Constants.OutputWin] / rOppTG
                : mwcLose;
        }
        else
        {
            float mwcWin = (1.0f - rG0 - rBG0) * res[MET_NDW]
                + rG0 * res[MET_NDWG]
                + rBG0 * res[MET_NDWB];
            mwcLive = mwcOppCash + (mwcWin - mwcOppCash)
                * (output[Constants.OutputWin] - rOppTG) / (1.0f - rOppTG);
        }

        return mwcDead * (1.0f - rCubeX) + mwcLive * rCubeX;
    }

    // Indices into the getMEMultiple result arrays (met_indices enum from matchequity.h)
    // First cube level: win outcomes
    private const int MET_DP = 0;   // Double/pass = normal win
    private const int MET_NDW = 0;  // Normal win (1×cube)
    private const int MET_DTW = 1;  // Double-take win (2×cube) / normal win gammon
    private const int MET_NDWG = 1; // Normal win gammon (2×cube)
    private const int MET_NDWB = 2; // Normal win backgammon (3×cube)
    private const int MET_DTWG = 3; // Double-take win gammon (4×cube)
    private const int MET_DTWB = 4; // Double-take win backgammon (6×cube)
    // First cube level: lose outcomes
    private const int MET_NDL = 5;  // Normal loss (1×cube)
    private const int MET_DTL = 6;  // Double-take loss (2×cube) / normal loss gammon
    private const int MET_NDLG = 6; // Normal loss gammon (2×cube)
    private const int MET_NDLB = 7; // Normal loss backgammon (3×cube)
    private const int MET_DTLG = 8; // Double-take loss gammon (4×cube)
    private const int MET_DTLB = 9; // Double-take loss backgammon (6×cube)
    // Second cube level (cube prime 0): win outcomes
    private const int MET_DPP0 = 10;
    private const int MET_DTWP0 = 11;
    private const int MET_NDWBP0 = 12;
    private const int MET_DTWGP0 = 13;
    private const int MET_DTWBP0 = 14;
    // Second cube level: lose outcomes
    private const int MET_NDLP0 = 15;
    private const int MET_DTLP0 = 16;
    private const int MET_NDLBP0 = 17;
    private const int MET_DTLGP0 = 18;
    private const int MET_DTLBP0 = 19;
    // Third cube level (cube prime 1): win outcomes
    private const int MET_DPP1 = 20;
    private const int MET_DTWP1 = 21;
    private const int MET_NDWBP1 = 22;
    private const int MET_DTWGP1 = 23;
    private const int MET_DTWBP1 = 24;
    // Third cube level: lose outcomes
    private const int MET_NDLP1 = 25;
    private const int MET_DTLP1 = 26;
    private const int MET_NDLBP1 = 27;
    private const int MET_DTLGP1 = 28;
    private const int MET_DTLBP1 = 29;

    private const int NDL_COUNT = 5; // entries per direction per cube level
    private const int MAXCUBELEVEL = 7;

    /// <summary>
    /// Batch lookup of match equities for multiple outcomes.
    /// Port of getMEMultiple() from matchequity.c.
    /// Returns equities for both players across up to 3 cube levels.
    /// </summary>
    internal static void GetMEMultiple(int score0, int score1, int matchTo,
        int cube, int cubePrime0, int cubePrime1, bool crawford,
        MatchEquityTable met, float[] player0, float[] player1)
    {
        int[] mult = [1, 2, 3, 4, 6];
        int away0 = matchTo - score0 - 1;
        int away1 = matchTo - score1 - 1;
        bool crawf = crawford || (matchTo - score0 == 1) || (matchTo - score1 == 1);

        // Determine how many results to compute
        int maxRes = cubePrime0 < 0 ? MET_DTLB + 1
                   : cubePrime1 < 0 ? MET_DTLBP0 + 1
                   : MET_DTLBP1 + 1;

        // Build score arrays for all outcomes
        int[] s0 = new int[maxRes];
        int[] s1 = new int[maxRes];
        int idx = 0;

        // First cube level: player 0 wins
        for (int i = 0; i < NDL_COUNT; i++)
        {
            s0[idx] = away0 - mult[i] * cube;
            s1[idx] = away1;
            idx++;
        }
        // First cube level: player 1 wins
        for (int i = 0; i < NDL_COUNT; i++)
        {
            s0[idx] = away0;
            s1[idx] = away1 - mult[i] * cube;
            idx++;
        }

        if (maxRes > MET_DPP0)
        {
            // Second cube level (cubePrime0): player 0 wins
            for (int i = 0; i < NDL_COUNT; i++)
            {
                s0[idx] = away0 - mult[i] * cubePrime0;
                s1[idx] = away1;
                idx++;
            }
            // Second cube level: player 1 wins
            for (int i = 0; i < NDL_COUNT; i++)
            {
                s0[idx] = away0;
                s1[idx] = away1 - mult[i] * cubePrime0;
                idx++;
            }

            if (maxRes > MET_DPP1)
            {
                // Third cube level (cubePrime1): player 0 wins
                for (int i = 0; i < NDL_COUNT; i++)
                {
                    s0[idx] = away0 - mult[i] * cubePrime1;
                    s1[idx] = away1;
                    idx++;
                }
                // Third cube level: player 1 wins
                for (int i = 0; i < NDL_COUNT; i++)
                {
                    s0[idx] = away0;
                    s1[idx] = away1 - mult[i] * cubePrime1;
                    idx++;
                }
            }
        }

        // Look up equities
        for (int i = 0; i < maxRes; i++)
        {
            if (s0[i] < 0)
            {
                // Player 0 wins the match
                player0[i] = 1.0f;
                player1[i] = 0.0f;
            }
            else if (s1[i] < 0)
            {
                // Player 1 wins the match
                player0[i] = 0.0f;
                player1[i] = 1.0f;
            }
            else if (crawf)
            {
                int cs0 = Math.Min(s0[i], Constants.MaxScore - 1);
                int cs1 = Math.Min(s1[i], Constants.MaxScore - 1);
                if (s0[i] == 0)
                {
                    // Player 0 is leading (1-away)
                    player0[i] = 1.0f - met.PostCrawford[1, cs1];
                    player1[i] = met.PostCrawford[1, cs1];
                }
                else
                {
                    // Player 1 is leading (1-away)
                    player0[i] = met.PostCrawford[0, cs0];
                    player1[i] = 1.0f - met.PostCrawford[0, cs0];
                }
            }
            else
            {
                int cs0 = Math.Min(s0[i], Constants.MaxScore - 1);
                int cs1 = Math.Min(s1[i], Constants.MaxScore - 1);
                player0[i] = met.Met[cs0, cs1];
                player1[i] = 1.0f - met.Met[cs0, cs1];
            }
        }

        // Swap win/loss columns for player1 so both arrays have wins first
        // For each cube level block, swap the first NDL_COUNT entries with the next NDL_COUNT
        SwapBlock(player1, 0, NDL_COUNT, NDL_COUNT);
        if (maxRes > MET_DTLBP0)
            SwapBlock(player1, MET_DPP0, MET_DPP0 + NDL_COUNT, NDL_COUNT);
        if (maxRes > MET_DTLBP1)
            SwapBlock(player1, MET_DPP1, MET_DPP1 + NDL_COUNT, NDL_COUNT);
    }

    private static void SwapBlock(float[] arr, int offset0, int offset1, int count)
    {
        for (int i = 0; i < count; i++)
            (arr[offset0 + i], arr[offset1 + i]) = (arr[offset1 + i], arr[offset0 + i]);
    }

    /// <summary>
    /// Determine the cube prime value (automatic redouble value).
    /// Port of GetCubePrimeValue() from matchequity.c.
    /// </summary>
    private static int GetCubePrimeValue(int away, int oppAway, int cubeValue)
    {
        if (away < 2 * cubeValue && oppAway >= 2 * cubeValue)
            return 2 * cubeValue; // automatic double
        return cubeValue;
    }

    /// <summary>
    /// Calculate live cube cash points using iterative cube-prime analysis.
    /// Port of GetPoints() from matchequity.c.
    /// Returns arCP[0] = cash point for player 0, arCP[1] = cash point for player 1.
    /// </summary>
    internal static void GetPoints(ReadOnlySpan<float> output, CubeInfo ci,
        MatchEquityTable met, float[] arCP)
    {
        int away0 = ci.MatchTo - ci.Score[0] - 1;
        int away1 = ci.MatchTo - ci.Score[1] - 1;
        int cube = ci.Cube;

        // Compute gammon ratios — note that GetPoints computes based on fMove
        float[] arG = new float[2];
        float[] arBG = new float[2];

        if (ci.Move == 0)
        {
            // Output evaluated for player 0
            if (output[Constants.OutputWin] > 0.0f)
            {
                arG[0] = (output[Constants.OutputWinGammon] - output[Constants.OutputWinBackgammon]) / output[Constants.OutputWin];
                arBG[0] = output[Constants.OutputWinBackgammon] / output[Constants.OutputWin];
            }
            if (output[Constants.OutputWin] < 1.0f)
            {
                arG[1] = (output[Constants.OutputLoseGammon] - output[Constants.OutputLoseBackgammon]) / (1.0f - output[Constants.OutputWin]);
                arBG[1] = output[Constants.OutputLoseBackgammon] / (1.0f - output[Constants.OutputWin]);
            }
        }
        else
        {
            // Output evaluated for player 1
            if (output[Constants.OutputWin] > 0.0f)
            {
                arG[1] = (output[Constants.OutputWinGammon] - output[Constants.OutputWinBackgammon]) / output[Constants.OutputWin];
                arBG[1] = output[Constants.OutputWinBackgammon] / output[Constants.OutputWin];
            }
            if (output[Constants.OutputWin] < 1.0f)
            {
                arG[0] = (output[Constants.OutputLoseGammon] - output[Constants.OutputLoseBackgammon]) / (1.0f - output[Constants.OutputWin]);
                arBG[0] = output[Constants.OutputLoseBackgammon] / (1.0f - output[Constants.OutputWin]);
            }
        }

        // Find the dead cube level: double until one side can't absorb 2×cube
        int dead = cube;
        int nMax = 0;
        while (away0 >= 2 * dead && away1 >= 2 * dead)
        {
            nMax++;
            dead *= 2;
        }

        float[,] cpLive = new float[2, MAXCUBELEVEL];
        float[,] cpDead = new float[2, MAXCUBELEVEL];
        float[] p0 = new float[MET_DTLBP1 + 1];
        float[] p1 = new float[MET_DTLBP1 + 1];

        // Iterate from dead cube level down to current
        for (int cubeValue = dead, n = nMax; n >= 0; cubeValue >>= 1, n--)
        {
            int cp0 = GetCubePrimeValue(away0, away1, cubeValue);
            int cp1 = GetCubePrimeValue(away1, away0, cubeValue);

            GetMEMultiple(ci.Score[0], ci.Score[1], ci.MatchTo,
                cubeValue, cp0, cp1, ci.Crawford, met, p0, p1);

            for (int k = 0; k < 2; k++)
            {
                float[] res = k == 0 ? p0 : p1;

                if (away0 < 2 * cubeValue || away1 < 2 * cubeValue)
                {
                    // Doubled cube will be dead — use cube prime values
                    int oppK = 1 - k;
                    int dtlIdx = k != 0 ? MET_DTLP1 : MET_DTLP0;
                    int dtlgIdx = k != 0 ? MET_DTLGP1 : MET_DTLGP0;
                    int dtlbIdx = k != 0 ? MET_DTLBP1 : MET_DTLBP0;
                    int dtwIdx = k != 0 ? MET_DTWP1 : MET_DTWP0;
                    int dtwgIdx = k != 0 ? MET_DTWGP1 : MET_DTWGP0;
                    int dtwbIdx = k != 0 ? MET_DTWBP1 : MET_DTWBP0;

                    float rDTL = (1.0f - arG[oppK] - arBG[oppK]) * res[dtlIdx]
                        + arG[oppK] * res[dtlgIdx]
                        + arBG[oppK] * res[dtlbIdx];

                    float rDP = res[MET_DP];

                    float rDTW = (1.0f - arG[k] - arBG[k]) * res[dtwIdx]
                        + arG[k] * res[dtwgIdx]
                        + arBG[k] * res[dtwbIdx];

                    float denom = rDTL - rDTW;
                    cpDead[k, n] = Math.Abs(denom) < 1e-10f ? 0.5f : (rDTL - rDP) / denom;
                    cpLive[k, n] = cpDead[k, n]; // dead cube → live = dead
                }
                else
                {
                    // Doubled cube is alive — use recursive formula
                    float rRDP = res[MET_DTL]; // redouble, pass
                    float rDP = res[MET_DP];   // double, pass

                    float rDTW = (1.0f - arG[k] - arBG[k]) * res[MET_DTW]
                        + arG[k] * res[MET_DTWG]
                        + arBG[k] * res[MET_DTWB];

                    float denom = rRDP - rDTW;
                    if (Math.Abs(denom) < 1e-10f)
                        cpLive[k, n] = 0.5f;
                    else
                        cpLive[k, n] = 1.0f - cpLive[1 - k, n + 1] * (rDP - rDTW) / denom;
                }
            }
        }

        arCP[0] = cpLive[0, 0];
        arCP[1] = cpLive[1, 0];
    }

    /// <summary>
    /// Compute gammon and backgammon ratios from neural net output.
    /// </summary>
    private static void ComputeGammonRatios(ReadOnlySpan<float> output,
        out float rG0, out float rBG0, out float rG1, out float rBG1)
    {
        float win = output[Constants.OutputWin];
        if (win > 0.0f)
        {
            rG0 = (output[Constants.OutputWinGammon] - output[Constants.OutputWinBackgammon]) / win;
            rBG0 = output[Constants.OutputWinBackgammon] / win;
        }
        else
        {
            rG0 = 0.0f;
            rBG0 = 0.0f;
        }

        if (win < 1.0f)
        {
            rG1 = (output[Constants.OutputLoseGammon] - output[Constants.OutputLoseBackgammon]) / (1.0f - win);
            rBG1 = output[Constants.OutputLoseBackgammon] / (1.0f - win);
        }
        else
        {
            rG1 = 0.0f;
            rBG1 = 0.0f;
        }
    }

    /// <summary>
    /// Convert cubeless neural net output to cubeful equity for money game.
    /// Port of Cl2CfMoney() from eval.c using Janowski's formula.
    /// </summary>
    internal static float CubelessToCubefulMoney(
        ReadOnlySpan<float> output, int cubeOwner, bool jacoby, float cubeEfficiency, int move = 0)
    {
        const float epsilon = 0.0000001f;
        const float omepsilon = 0.9999999f;

        float winProb = output[Constants.OutputWin];

        float rW, rL;

        if (winProb > epsilon)
            rW = 1.0f + (output[Constants.OutputWinGammon] + output[Constants.OutputWinBackgammon]) / winProb;
        else
            return MatchEquityTable.MoneyEquity(output);

        if (winProb < omepsilon)
            rL = 1.0f + (output[Constants.OutputLoseGammon] + output[Constants.OutputLoseBackgammon]) / (1.0f - winProb);
        else
            return MatchEquityTable.MoneyEquity(output);

        float eqDead = MatchEquityTable.MoneyEquity(output);
        float eqLive = MoneyLive(rW, rL, winProb, cubeOwner, move, jacoby);

        return eqDead * (1.0f - cubeEfficiency) + eqLive * cubeEfficiency;
    }

    /// <summary>
    /// Live cube equity for money game using Janowski's model.
    /// Port of MoneyLive() from eval.c.
    /// cubeOwner: -1=centered, 0=player0, 1=player1.
    /// move: index of the player on roll (whose perspective the output is from).
    /// In C: pci->fCubeOwner == pci->fMove means "I own the cube".
    /// </summary>
    private static float MoneyLive(float rW, float rL, float p, int cubeOwner, int move, bool jacoby)
    {
        if (cubeOwner == -1)
        {
            // Centered cube
            float rTP = (rL - 0.5f) / (rW + rL + 0.5f);
            float rCP = (rL + 1.0f) / (rW + rL + 0.5f);

            if (p < rTP)
                return jacoby ? -1.0f : (-rL + (-1.0f + rL) * p / rTP);
            else if (p < rCP)
                return -1.0f + 2.0f * (p - rTP) / (rCP - rTP);
            else
                return jacoby ? 1.0f : (1.0f + (rW - 1.0f) * (p - rCP) / (1.0f - rCP));
        }
        else if (cubeOwner == move)
        {
            // Player owns cube
            float rCP = (rL + 1.0f) / (rW + rL + 0.5f);

            if (p < rCP)
                return -rL + (1.0f + rL) * p / rCP;
            else
                return 1.0f + (rW - 1.0f) * (p - rCP) / (1.0f - rCP);
        }
        else
        {
            // Opponent owns cube
            float rTP = (rL - 0.5f) / (rW + rL + 0.5f);

            if (p < rTP)
                return -rL + (-1.0f + rL) * p / rTP;
            else
                return -1.0f + (rW + 1.0f) * (p - rTP) / (1.0f - rTP);
        }
    }


    /// <summary>
    /// Convert standard error from MWC space to equity space.
    /// Port of se_mwc2eq() from eval.c.
    /// </summary>
    public static float SeMwc2Eq(float seMwc, CubeInfo ci, MatchEquityTable met)
    {
        float mwcWin = GetME(ci.Score[0], ci.Score[1], ci.MatchTo,
            ci.Move, ci.Cube, ci.Move, ci.Crawford, met);
        float mwcLose = GetME(ci.Score[0], ci.Score[1], ci.MatchTo,
            ci.Move, ci.Cube, 1 - ci.Move, ci.Crawford, met);

        float denom = mwcWin - mwcLose;
        if (Math.Abs(denom) < 1e-10f)
            return 0.0f;

        return 2.0f / denom * seMwc;
    }

    /// <summary>
    /// Convert standard error from equity space to MWC space.
    /// Port of se_eq2mwc() from eval.c.
    /// </summary>
    public static float SeEq2Mwc(float seEq, CubeInfo ci, MatchEquityTable met)
    {
        float mwcWin = GetME(ci.Score[0], ci.Score[1], ci.MatchTo,
            ci.Move, ci.Cube, ci.Move, ci.Crawford, met);
        float mwcLose = GetME(ci.Score[0], ci.Score[1], ci.MatchTo,
            ci.Move, ci.Cube, 1 - ci.Move, ci.Crawford, met);

        return (mwcWin - mwcLose) / 2.0f * seEq;
    }
}
