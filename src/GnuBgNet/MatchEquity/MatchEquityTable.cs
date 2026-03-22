// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2019 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Ported from matchequity.c (initMETZadeh, initPostCrawfordMET)

namespace GnuBgNet.MatchEquity;

/// <summary>
/// Match equity table: pre-Crawford and post-Crawford equities.
/// Port of aafMET/aafMETPostCrawford from matchequity.c.
/// </summary>
public sealed class MatchEquityTable : IMatchEquityTable
{
    private const int MaxScore = Constants.MaxScore;

    /// <summary>Pre-Crawford MET: Met[i][j] = equity of player needing i+1 points vs player needing j+1 points.</summary>
    public float[,] Met { get; } = new float[MaxScore, MaxScore];

    /// <summary>Post-Crawford MET: PostCrawford[side][i] = equity of player on Crawford side needing i+1 points.</summary>
    public float[,] PostCrawford { get; } = new float[2, MaxScore];

    /// <summary>
    /// Get match winning chance for player needing iAway points vs opponent needing jAway points.
    /// iAway and jAway are 1-based (1 = needs 1 point to win).
    /// </summary>
    public float GetEquity(int iAway, int jAway)
    {
        if (iAway <= 0) return 1.0f;
        if (jAway <= 0) return 0.0f;
        return Met[Math.Min(iAway - 1, MaxScore - 1), Math.Min(jAway - 1, MaxScore - 1)];
    }

    /// <summary>
    /// Compute the default match equity table using Zadeh's model.
    /// Default parameters match gnubg's Kazaross-XG2 configuration.
    /// </summary>
    private static MatchEquityTable? _defaultInstance;

    public static MatchEquityTable ComputeDefault()
    {
        if (_defaultInstance != null) return _defaultInstance;
        // Kazaross-XG2 default parameters (from gnubg met/Kazaross-XG2.xml)
        const float rG1 = 0.26f;   // gammon rate for leader
        const float rG2 = 0.26f;   // gammon rate for trailer
        const float rG3 = 0.26f;   // post-Crawford gammon rate
        const float rFD2 = 0.015f; // free drop at 1-away, 2-away
        const float rFD4 = 0.004f; // free drop at 1-away, 4-away
        const float rDelta = 0.12f;
        const float rDeltaBar = 0.6f;

        var met = new MatchEquityTable();

        // Initialize post-Crawford equities
        for (int side = 0; side < 2; side++)
        {
            met.PostCrawford[side, 0] = 0.5f; // 1-away post-Crawford = 50%
            InitPostCrawfordMET(met.PostCrawford, side, 1, rG3, rFD2, rFD4);
        }

        // Initialize pre-Crawford MET using Zadeh's formula
        InitMETZadeh(met.Met, met.PostCrawford, rG1, rG2, rDelta, rDeltaBar);

        _defaultInstance = met;
        return met;
    }

    private static void InitPostCrawfordMET(float[,] postCrawford, int side, int iStart,
        float rG, float rFD2, float rFD4)
    {
        for (int i = iStart; i < MaxScore; i++)
        {
            postCrawford[side, i] = rG * 0.5f * ((i - 4 >= 0) ? postCrawford[side, i - 4] : 1.0f)
                + (1.0f - rG) * 0.5f * ((i - 2 >= 0) ? postCrawford[side, i - 2] : 1.0f);

            if (i == 1) postCrawford[side, i] -= rFD2;
            if (i == 3) postCrawford[side, i] -= rFD4;

            postCrawford[side, i] = Math.Clamp(postCrawford[side, i], 0.0f, 1.0f);
        }
    }

    /// <summary>
    /// Full Zadeh MET computation with cube-level iteration.
    /// Port of initMETZadeh() from matchequity.c.
    /// </summary>
    private static void InitMETZadeh(float[,] met, float[,] postCrawford,
        float rG1, float rG2, float rDelta, float rDeltaBar)
    {
        const int MAXCUBELEVEL = 7;

        // D1bar/D2bar = dead-cube drop points, D1/D2 = semi-efficient recube drop points
        float[,,] D1 = new float[MaxScore, MaxScore, MAXCUBELEVEL];
        float[,,] D2 = new float[MaxScore, MaxScore, MAXCUBELEVEL];
        float[,,] D1bar = new float[MaxScore, MaxScore, MAXCUBELEVEL];
        float[,,] D2bar = new float[MaxScore, MaxScore, MAXCUBELEVEL];

        // 1-away, n-away
        for (int i = 0; i < MaxScore; i++)
        {
            met[i, 0] = rG1 * 0.5f * ((i - 2 >= 0) ? postCrawford[0, i - 2] : 1.0f)
                + (1.0f - rG1) * 0.5f * ((i - 1 >= 0) ? postCrawford[0, i - 1] : 1.0f);
            met[0, i] = 1.0f - met[i, 0];
        }

        for (int i = 0; i < MaxScore; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                for (int nCube = MAXCUBELEVEL - 1; nCube >= 0; nCube--)
                {
                    int nCubeValue = 1 << nCube;

                    // D1bar[i][j]
                    int cpv = GetCubePrimeValue(i, j, nCubeValue);
                    float num = GetMET(i - nCubeValue, j, met)
                        - rG2 * GetMET(i, j - 4 * cpv, met)
                        - (1.0f - rG2) * GetMET(i, j - 2 * cpv, met);
                    float den = rG1 * GetMET(i - 4 * cpv, j, met)
                        + (1.0f - rG1) * GetMET(i - 2 * cpv, j, met)
                        - rG2 * GetMET(i, j - 4 * cpv, met)
                        - (1.0f - rG2) * GetMET(i, j - 2 * cpv, met);
                    D1bar[i, j, nCube] = Math.Abs(den) > 1e-12f ? num / den : 0.5f;

                    if (i != j)
                    {
                        cpv = GetCubePrimeValue(j, i, nCubeValue);
                        num = GetMET(j - nCubeValue, i, met)
                            - rG2 * GetMET(j, i - 4 * cpv, met)
                            - (1.0f - rG2) * GetMET(j, i - 2 * cpv, met);
                        den = rG1 * GetMET(j - 4 * cpv, i, met)
                            + (1.0f - rG1) * GetMET(j - 2 * cpv, i, met)
                            - rG2 * GetMET(j, i - 4 * cpv, met)
                            - (1.0f - rG2) * GetMET(j, i - 2 * cpv, met);
                        D1bar[j, i, nCube] = Math.Abs(den) > 1e-12f ? num / den : 0.5f;
                    }

                    // D2bar[i][j]
                    cpv = GetCubePrimeValue(j, i, nCubeValue);
                    num = GetMET(j - nCubeValue, i, met)
                        - rG2 * GetMET(j, i - 4 * cpv, met)
                        - (1.0f - rG2) * GetMET(j, i - 2 * cpv, met);
                    den = rG1 * GetMET(j - 4 * cpv, i, met)
                        + (1.0f - rG1) * GetMET(j - 2 * cpv, i, met)
                        - rG2 * GetMET(j, i - 4 * cpv, met)
                        - (1.0f - rG2) * GetMET(j, i - 2 * cpv, met);
                    D2bar[i, j, nCube] = Math.Abs(den) > 1e-12f ? num / den : 0.5f;

                    if (i != j)
                    {
                        cpv = GetCubePrimeValue(i, j, nCubeValue);
                        num = GetMET(i - nCubeValue, j, met)
                            - rG2 * GetMET(i, j - 4 * cpv, met)
                            - (1.0f - rG2) * GetMET(i, j - 2 * cpv, met);
                        den = rG1 * GetMET(i - 4 * cpv, j, met)
                            + (1.0f - rG1) * GetMET(i - 2 * cpv, j, met)
                            - rG2 * GetMET(i, j - 4 * cpv, met)
                            - (1.0f - rG2) * GetMET(i, j - 2 * cpv, met);
                        D2bar[j, i, nCube] = Math.Abs(den) > 1e-12f ? num / den : 0.5f;
                    }

                    // D1
                    if (i < 2 * nCubeValue || j < 2 * nCubeValue)
                    {
                        D1[i, j, nCube] = D1bar[i, j, nCube];
                        if (i != j) D1[j, i, nCube] = D1bar[j, i, nCube];
                    }
                    else
                    {
                        D1[i, j, nCube] = 1.0f + (D2[i, j, nCube + 1] + rDelta)
                            * (D1bar[i, j, nCube] - 1.0f);
                        if (i != j)
                            D1[j, i, nCube] = 1.0f + (D2[j, i, nCube + 1] + rDelta)
                                * (D1bar[j, i, nCube] - 1.0f);
                    }

                    // D2
                    if (i < 2 * nCubeValue || j < 2 * nCubeValue)
                    {
                        D2[i, j, nCube] = D2bar[i, j, nCube];
                        if (i != j) D2[j, i, nCube] = D2bar[j, i, nCube];
                    }
                    else
                    {
                        D2[i, j, nCube] = 1.0f + (D1[i, j, nCube + 1] + rDelta)
                            * (D2bar[i, j, nCube] - 1.0f);
                        if (i != j)
                            D2[j, i, nCube] = 1.0f + (D1[j, i, nCube + 1] + rDelta)
                                * (D2bar[j, i, nCube] - 1.0f);
                    }

                    // Final MET entry at cube level 0
                    if (nCube == 0 && i > 0 && j > 0)
                    {
                        float d1 = D1[i, j, 0];
                        float d2 = D2[i, j, 0];
                        float denomMet = d1 + rDeltaBar + d2 + rDeltaBar - 1.0f;
                        if (Math.Abs(denomMet) > 1e-12f)
                        {
                            met[i, j] = ((d2 + rDeltaBar - 0.5f) * GetMET(i - 1, j, met)
                                + (d1 + rDeltaBar - 0.5f) * GetMET(i, j - 1, met))
                                / denomMet;
                        }
                        else
                        {
                            met[i, j] = 0.5f;
                        }

                        if (i != j)
                            met[j, i] = 1.0f - met[i, j];
                    }
                }
            }
        }
    }

    /// <summary>
    /// Port of GetCubePrimeValue() from matchequity.c.
    /// </summary>
    private static int GetCubePrimeValue(int i, int j, int nCubeValue)
    {
        if (i < 2 * nCubeValue && j >= 2 * nCubeValue)
            return 2 * nCubeValue;
        return nCubeValue;
    }

    /// <summary>
    /// Port of GET_MET macro from matchequity.h.
    /// </summary>
    private static float GetMET(int i, int j, float[,] met)
    {
        if (i < 0) return 1.0f;
        if (j < 0) return 0.0f;
        return met[Math.Min(i, MaxScore - 1), Math.Min(j, MaxScore - 1)];
    }

    /// <summary>
    /// Compute money game equity from output probabilities.
    /// Port of Utility() from eval.c for money play.
    /// </summary>
    public static float MoneyEquity(ReadOnlySpan<float> output)
    {
        return output[Constants.OutputWin] * 2.0f - 1.0f
            + output[Constants.OutputWinGammon]
            + output[Constants.OutputWinBackgammon]
            - output[Constants.OutputLoseGammon]
            - output[Constants.OutputLoseBackgammon];
    }
}
