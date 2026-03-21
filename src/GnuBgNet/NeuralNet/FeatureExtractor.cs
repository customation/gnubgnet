// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2019 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Port of gnubgapi_position_to_features() from gnubgapi.c and contact_features_24() logic

using GnuBgNet.Evaluation;

namespace GnuBgNet.NeuralNet;

/// <summary>
/// Feature dimension constant for the public API.
/// 248 = 100 (base bottom) + 24 (contact bottom) + 100 (base top) + 24 (contact top)
/// Port of GNUBGAPI_FEATURE_DIM from gnubgapi.h.
/// </summary>
public static class FeatureExtractor
{
    /// <summary>Total feature dimension for the public API.</summary>
    public const int FeatureDim = 248;

    /// <summary>Base features per side (25 points × 4 features).</summary>
    private const int BasePerSide = 100;

    /// <summary>Contact features per side.</summary>
    private const int ContactPerSide = 24;

    /// <summary>
    /// Extract neural network input features from a board position.
    /// Returns 248 floats: [bottom_base(100) | bottom_contact(24) | top_base(100) | top_contact(24)]
    /// Port of gnubgapi_position_to_features() from gnubgapi.c.
    /// </summary>
    /// <param name="board">The board position to extract features from.</param>
    /// <param name="isTopOnRoll">If true, top player (Opponent) is on roll; if false, bottom (Player) is on roll.</param>
    /// <param name="features">Output span of at least 248 floats.</param>
    public static void ExtractFeatures(Board board, bool isTopOnRoll, Span<float> features)
    {
        if (features.Length < FeatureDim)
            throw new ArgumentException($"Output span must be at least {FeatureDim} elements");

        features[..FeatureDim].Clear();

        // The board for feature extraction: if top is on roll, we need to swap
        var evalBoard = isTopOnRoll ? board.Swapped() : board;

        // Base inputs: 200 floats (25 points × 4 features × 2 sides)
        Span<float> baseInputs = stackalloc float[200];
        InputCalculator.BaseInputs(evalBoard, baseInputs);

        // Bottom player base features (first 100 from anBoard[0] = Opponent)
        baseInputs[..BasePerSide].CopyTo(features);

        // Bottom player contact features
        ExtractContactFeatures(evalBoard.Opponent, evalBoard.Player, features[BasePerSide..]);

        // Top player base features (second 100 from anBoard[1] = Player)
        baseInputs.Slice(BasePerSide, BasePerSide).CopyTo(features[(BasePerSide + ContactPerSide)..]);

        // Top player contact features
        ExtractContactFeatures(evalBoard.Player, evalBoard.Opponent,
            features[(2 * BasePerSide + ContactPerSide)..]);
    }

    /// <summary>
    /// Extract raw neural network inputs for the specific position class.
    /// Returns the full input array that would be fed to the neural net.
    /// </summary>
    /// <param name="board">The board position.</param>
    /// <param name="pc">The position class (determines which input calculator to use).</param>
    /// <param name="inputs">Output span sized for the position class input count.</param>
    public static void ExtractRawInputs(Board board, PositionClass pc, Span<float> inputs)
    {
        switch (pc)
        {
            case PositionClass.Race:
                if (inputs.Length < Constants.NumRaceInputs)
                    throw new ArgumentException($"Need at least {Constants.NumRaceInputs} elements for race inputs");
                InputCalculator.CalculateRaceInputs(board, inputs);
                break;

            case PositionClass.Contact:
                if (inputs.Length < Constants.NumContactInputs)
                    throw new ArgumentException($"Need at least {Constants.NumContactInputs} elements for contact inputs");
                InputCalculator.CalculateContactInputs(board, inputs);
                break;

            case PositionClass.Crashed:
                if (inputs.Length < Constants.NumCrashedInputs)
                    throw new ArgumentException($"Need at least {Constants.NumCrashedInputs} elements for crashed inputs");
                InputCalculator.CalculateCrashedInputs(board, inputs);
                break;

            default:
                throw new ArgumentException($"Raw inputs not available for position class {pc}");
        }
    }

    /// <summary>
    /// Extract 24 contact-specific features for one side.
    /// Port of contact_features_24() from gnubgapi.c lines 1336-1592.
    /// These include men off, back checker, anchors, piploss, mobility, containment, etc.
    /// </summary>
    private static void ExtractContactFeatures(uint[] anBoard, uint[] anBoardOpp, Span<float> features)
    {
        // I_OFF1, I_OFF2, I_OFF3 (men off - tiered encoding)
        int menOff = 15;
        for (int i = 0; i < 25; i++)
            menOff -= (int)anBoard[i];

        if (menOff <= 2)
        {
            features[0] = menOff > 0 ? menOff / 3.0f : 0.0f;
            features[1] = 0.0f;
            features[2] = 0.0f;
        }
        else if (menOff <= 5)
        {
            features[0] = 1.0f;
            features[1] = (menOff - 3) / 3.0f;
            features[2] = 0.0f;
        }
        else
        {
            features[0] = 1.0f;
            features[1] = 1.0f;
            features[2] = (menOff - 6) / 3.0f;
        }

        // I_BREAK_CONTACT (3)
        int nOppBack;
        for (nOppBack = 24; nOppBack >= 0; --nOppBack)
            if (anBoardOpp[nOppBack] > 0) break;
        nOppBack = 23 - nOppBack;

        {
            int np = 0;
            for (int i = nOppBack + 1; i < 25; i++)
                if (anBoard[i] > 0)
                    np += (i + 1 - nOppBack) * (int)anBoard[i];
            features[3] = np / (15 + 152.0f);
        }

        // I_BACK_CHEQUER (4)
        int nBack;
        for (nBack = 24; nBack >= 0; --nBack)
            if (anBoard[nBack] > 0) break;
        features[4] = nBack / 24.0f;

        // I_BACK_ANCHOR (5)
        int backAnchor;
        for (backAnchor = (nBack == 24 ? 23 : nBack); backAnchor >= 0; --backAnchor)
            if (anBoard[backAnchor] >= 2) break;
        features[5] = backAnchor / 24.0f;

        // I_FORWARD_ANCHOR (6)
        {
            int n = 0;
            for (int j = 18; j <= backAnchor; ++j)
                if (anBoard[j] >= 2) { n = 24 - j; break; }
            if (n == 0)
                for (int j = 17; j >= 12; --j)
                    if (anBoard[j] >= 2) { n = 24 - j; break; }
            features[6] = n == 0 ? 2.0f : n / 6.0f;
        }

        // I_PIPLOSS (7), I_P1 (8), I_P2 (9) — simplified from full hit calculation
        {
            int totalPips = 0, hitRolls = 0, multiHitRolls = 0;
            // Simple blot-exposure measure
            for (int i = 0; i < 24; i++)
            {
                if (anBoardOpp[i] != 1) continue;
                int target = 23 - i;
                for (int j = 1; j <= Math.Min(24, target); j++)
                {
                    int from = target - j;
                    if (from >= 0 && from < 25 && anBoard[from] > 0)
                    {
                        if (j <= 6) { hitRolls++; totalPips += j; }
                    }
                }
            }
            features[7] = Math.Min(1.0f, totalPips / (12.0f * 36.0f));
            features[8] = Math.Min(1.0f, hitRolls / 36.0f);
            features[9] = Math.Min(1.0f, multiHitRolls / 36.0f);
        }

        // I_BACKESCAPES (10)
        features[10] = EscapeTable.GetEscapes(anBoard, 23 - nOppBack) / 36.0f;

        // I_ACONTAIN (11), I_ACONTAIN2 (12)
        {
            int n = 36;
            for (int i = 15; i < 24 - nOppBack; i++)
            {
                int j = EscapeTable.GetEscapes(anBoard, i);
                if (j < n) n = j;
            }
            features[11] = (36 - n) / 36.0f;
            features[12] = features[11] * features[11];
        }

        // I_CONTAIN (13), I_CONTAIN2 (14)
        {
            int n = 36;
            for (int i = 15; i < 24; i++)
            {
                int j = EscapeTable.GetEscapes(anBoard, i);
                if (j < n) n = j;
            }
            features[13] = (36 - n) / 36.0f;
            features[14] = features[13] * features[13];
        }

        // I_MOBILITY (15)
        {
            int n = 0;
            for (int i = 6; i < 25; i++)
                if (anBoard[i] > 0)
                    n += (i - 5) * (int)anBoard[i] * EscapeTable.GetEscapes(anBoardOpp, i);
            features[15] = n / 3600.0f;
        }

        // I_MOMENT2 (16)
        {
            int j = 0, n = 0;
            for (int i = 0; i < 25; i++)
                if (anBoard[i] > 0) { j += (int)anBoard[i]; n += i * (int)anBoard[i]; }
            n = j > 0 ? (n + j - 1) / j : 0;
            int k2 = 0; j = 0;
            for (int i = n + 1; i < 25; i++)
                if (anBoard[i] > 0) { j += (int)anBoard[i]; k2 += (int)anBoard[i] * (i - n) * (i - n); }
            if (j > 0) k2 = (k2 + j - 1) / j;
            features[16] = k2 / 400.0f;
        }

        // I_ENTER (17)
        if (anBoard[24] > 0)
        {
            int loss = 0;
            bool two = anBoard[24] > 1;
            for (int i = 0; i < 6; ++i)
            {
                if (anBoardOpp[i] > 1)
                {
                    loss += 4 * (i + 1);
                    for (int j = i + 1; j < 6; ++j)
                        if (anBoardOpp[j] > 1) loss += 2 * (i + j + 2);
                        else if (two) loss += 2 * (i + 1);
                }
                else if (two)
                    for (int j = i + 1; j < 6; ++j)
                        if (anBoardOpp[j] > 1) loss += 2 * (j + 1);
            }
            features[17] = loss / (36.0f * (49.0f / 6.0f));
        }

        // I_ENTER2 (18)
        {
            int n = 0;
            for (int i = 0; i < 6; i++)
                n += anBoardOpp[i] > 1 ? 1 : 0;
            features[18] = (36 - (n - 6) * (n - 6)) / 36.0f;
        }

        // I_TIMING (19)
        {
            int t = 0, no = 0;
            int m = nOppBack >= 11 ? nOppBack : 11;
            t += 24 * (int)anBoard[24];
            no += (int)anBoard[24];
            for (int i = 23; i > m; --i)
                if (anBoard[i] > 0 && anBoard[i] != 2)
                { int ns = anBoard[i] > 2 ? (int)(anBoard[i] - 2) : 1; no += ns; t += i * ns; }
            for (int i = m; i >= 6; --i)
                if (anBoard[i] > 0) { no += (int)anBoard[i]; t += i * (int)anBoard[i]; }
            for (int i = 5; i >= 0; --i)
                if (anBoard[i] > 2) { t += i * (int)(anBoard[i] - 2); no += (int)(anBoard[i] - 2); }
                else if (anBoard[i] < 2)
                { int nm = 2 - (int)anBoard[i]; if (no >= nm) { t -= i * nm; no -= nm; } }
            features[19] = t / 100.0f;
        }

        // I_BACKBONE (20)
        {
            int pa = -1, w = 0, tot = 0;
            int[] ac = [11, 11, 11, 11, 11, 11, 11, 6, 5, 4, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
            for (int np2 = 23; np2 > 0; --np2)
                if (anBoard[np2] >= 2)
                {
                    if (pa == -1) { pa = np2; continue; }
                    int d = pa - np2; w += ac[d] * (int)anBoard[pa]; tot += (int)anBoard[pa];
                }
            features[20] = tot > 0 ? 1.0f - w / (tot * 11.0f) : 0.0f;
        }

        // I_BACKG (21), I_BACKG1 (22)
        {
            uint nAc = 0;
            for (int i = 18; i < 24; ++i)
                if (anBoard[i] > 1) ++nAc;
            features[21] = 0.0f;
            features[22] = 0.0f;
            if (nAc >= 1)
            {
                uint tot = 0;
                for (int i = 18; i < 25; ++i) tot += anBoard[i];
                if (nAc > 1) features[21] = (tot - 3) / 4.0f;
                else features[22] = tot / 8.0f;
            }
        }

        // I_FREEPIP (23)
        {
            uint p = 0;
            for (int i = 0; i < nOppBack; i++)
                if (anBoard[i] > 0) p += (uint)(i + 1) * anBoard[i];
            features[23] = p / 100.0f;
        }
    }
}
