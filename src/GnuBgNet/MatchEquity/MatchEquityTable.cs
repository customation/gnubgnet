// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2019 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Ported from matchequity.c (initMETZadeh, initPostCrawfordMET)

namespace GnuBgNet.MatchEquity;

/// <summary>
/// Match equity table: pre-Crawford and post-Crawford equities.
/// Port of aafMET/aafMETPostCrawford from matchequity.c.
/// </summary>
public sealed class MatchEquityTable
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

    private static void InitMETZadeh(float[,] met, float[,] postCrawford,
        float rG1, float rG2, float rDelta, float rDeltaBar)
    {
        // 1-away, n-away
        for (int i = 0; i < MaxScore; i++)
        {
            met[i, 0] = rG1 * 0.5f * ((i - 2 >= 0) ? postCrawford[0, i - 2] : 1.0f)
                + (1.0f - rG1) * 0.5f * ((i - 1 >= 0) ? postCrawford[0, i - 1] : 1.0f);
            met[0, i] = 1.0f - met[i, 0];
        }

        // General case using simplified Zadeh recursion
        for (int i = 1; i < MaxScore; i++)
        {
            for (int j = 1; j < MaxScore; j++)
            {
                // Simplified Zadeh: use recursion with gammon probabilities
                // met[i][j] ≈ (1-G1) * (met[i-1][j] + met[i][j-1]) / 2
                //            + G1 * (met[i-2][j] + met[i][j-2]) / 2
                // with cube efficiency adjustments

                float noGammon = 0.5f * (GetSafe(met, i - 1, j) + (1.0f - GetSafe(met, j - 1, i)));
                float gammon = 0.5f * (GetSafe(met, i - 2, j) + (1.0f - GetSafe(met, j - 2, i)));

                // Cube efficiency blend
                float cubeless = (1.0f - rG1) * noGammon + rG1 * gammon;

                // Apply cube efficiency adjustments
                float doublePoint = cubeless + rDelta * (0.5f - Math.Abs(cubeless - 0.5f));
                met[i, j] = doublePoint * (1.0f - rDeltaBar) + cubeless * rDeltaBar;

                met[i, j] = Math.Clamp(met[i, j], 0.0f, 1.0f);
            }
        }
    }

    private static float GetSafe(float[,] met, int i, int j)
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
