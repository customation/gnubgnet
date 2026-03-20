// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2019 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Ported from eval.c (CalculateRaceInputs, CalculateContactInputs, etc.) and lib/inputs.c

using GnuBgNet.Evaluation;

namespace GnuBgNet.NeuralNet;

/// <summary>
/// Calculates neural network input features from a board position.
/// Port of baseInputs, CalculateRaceInputs, CalculateContactInputs, CalculateCrashedInputs from eval.c/inputs.c.
/// </summary>
public static class InputCalculator
{
    // Lookup tables for base inputs: inpvec[nc] = {nc==1, nc==2, nc>=3, max(0,(nc-3)/2)}
    private static readonly float[][] InputVec =
    [
        [0f, 0f, 0f, 0f],  // 0
        [1f, 0f, 0f, 0f],  // 1
        [0f, 1f, 0f, 0f],  // 2
        [0f, 0f, 1f, 0f],  // 3
        [0f, 0f, 1f, 0.5f], // 4
        [0f, 0f, 1f, 1.0f], // 5
        [0f, 0f, 1f, 1.5f], // 6
        [0f, 0f, 1f, 2.0f], // 7
        [0f, 0f, 1f, 2.5f], // 8
        [0f, 0f, 1f, 3.0f], // 9
        [0f, 0f, 1f, 3.5f], // 10
        [0f, 0f, 1f, 4.0f], // 11
        [0f, 0f, 1f, 4.5f], // 12
        [0f, 0f, 1f, 5.0f], // 13
        [0f, 0f, 1f, 5.5f], // 14
        [0f, 0f, 1f, 6.0f], // 15
    ];

    // Bar uses cumulative encoding: inpvecb[nc] = {nc>=1, nc>=2, nc>=3, max(0,(nc-3)/2)}
    private static readonly float[][] InputVecBar =
    [
        [0f, 0f, 0f, 0f],  // 0
        [1f, 0f, 0f, 0f],  // 1
        [1f, 1f, 0f, 0f],  // 2
        [1f, 1f, 1f, 0f],  // 3
        [1f, 1f, 1f, 0.5f], // 4
        [1f, 1f, 1f, 1.0f], // 5
        [1f, 1f, 1f, 1.5f], // 6
        [1f, 1f, 1f, 2.0f], // 7
        [1f, 1f, 1f, 2.5f], // 8
        [1f, 1f, 1f, 3.0f], // 9
        [1f, 1f, 1f, 3.5f], // 10
        [1f, 1f, 1f, 4.0f], // 11
        [1f, 1f, 1f, 4.5f], // 12
        [1f, 1f, 1f, 5.0f], // 13
        [1f, 1f, 1f, 5.5f], // 14
        [1f, 1f, 1f, 6.0f], // 15
    ];

    private const int RI_OFF = 92;
    private const int RI_NCROSS = 106;
    private const int HalfRaceInputs = 107;

    /// <summary>
    /// Compute base inputs (200 floats: 25 points × 4 features × 2 sides).
    /// Port of baseInputs() from lib/inputs.c.
    /// </summary>
    public static void BaseInputs(Board board, Span<float> arInput)
    {
        // Side 0 = Opponent (anBoard[0]), Side 1 = Player (anBoard[1])
        for (int side = 0; side < 2; side++)
        {
            uint[] b = side == 0 ? board.Opponent : board.Player;
            int offset = side * 25 * 4;

            // 24 points
            for (int i = 0; i < 24; i++)
            {
                uint nc = Math.Min(b[i], 15);
                var vec = InputVec[nc];
                arInput[offset + i * 4 + 0] = vec[0];
                arInput[offset + i * 4 + 1] = vec[1];
                arInput[offset + i * 4 + 2] = vec[2];
                arInput[offset + i * 4 + 3] = vec[3];
            }

            // Bar (index 24)
            {
                uint nc = Math.Min(b[24], 15);
                var vec = InputVecBar[nc];
                arInput[offset + 24 * 4 + 0] = vec[0];
                arInput[offset + 24 * 4 + 1] = vec[1];
                arInput[offset + 24 * 4 + 2] = vec[2];
                arInput[offset + 24 * 4 + 3] = vec[3];
            }
        }
    }

    /// <summary>
    /// Compute race neural net inputs (214 floats).
    /// Port of CalculateRaceInputs() from eval.c.
    /// </summary>
    public static void CalculateRaceInputs(Board board, Span<float> inputs)
    {
        for (int side = 0; side < 2; side++)
        {
            uint[] b = side == 0 ? board.Opponent : board.Player;
            int offset = side * HalfRaceInputs;

            uint menOff = 15;

            // Points 0-22 (23 points, 4 features each = 92)
            for (int i = 0; i < 23; i++)
            {
                uint nc = b[i];
                menOff -= nc;
                int k = offset + i * 4;
                inputs[k + 0] = (nc == 1) ? 1.0f : 0.0f;
                inputs[k + 1] = (nc == 2) ? 1.0f : 0.0f;
                inputs[k + 2] = (nc >= 3) ? 1.0f : 0.0f;
                inputs[k + 3] = nc > 3 ? (nc - 3) / 2.0f : 0.0f;
            }

            // Men off: 14 one-hot indicators
            for (int k = 0; k < 14; k++)
            {
                inputs[offset + RI_OFF + k] = (menOff == (uint)(k + 1)) ? 1.0f : 0.0f;
            }

            // Cross-overs
            uint nCross = 0;
            for (int k = 1; k < 4; k++)
            {
                for (int i = 6 * k; i < 6 * k + 6; i++)
                {
                    uint nc = b[i];
                    if (nc > 0)
                        nCross += nc * (uint)k;
                }
            }
            inputs[offset + RI_NCROSS] = nCross / 10.0f;
        }
    }

    // Contact input indices (25 per side)
    private const int I_OFF1 = 0;
    private const int I_OFF2 = 1;
    private const int I_OFF3 = 2;
    private const int I_BREAK_CONTACT = 3;
    private const int I_BACK_CHEQUER = 4;
    private const int I_BACK_ANCHOR = 5;
    private const int I_FORWARD_ANCHOR = 6;
    private const int I_PIPLOSS = 7;
    private const int I_P1 = 8;
    private const int I_P2 = 9;
    private const int I_BACKESCAPES = 10;
    private const int I_ACONTAIN = 11;
    private const int I_ACONTAIN2 = 12;
    private const int I_CONTAIN = 13;
    private const int I_CONTAIN2 = 14;
    private const int I_MOBILITY = 15;
    private const int I_MOMENT2 = 16;
    private const int I_ENTER = 17;
    private const int I_ENTER2 = 18;
    private const int I_TIMING = 19;
    private const int I_BACKBONE = 20;
    private const int I_BACKG = 21;
    private const int I_BACKG1 = 22;
    private const int I_FREEPIP = 23;
    private const int I_BACKRESCAPES = 24;
    private const int MORE_INPUTS = 25;
    private const int MINPPERPOINT = 4;

    /// <summary>
    /// Compute contact neural net inputs (250 floats).
    /// Port of CalculateContactInputs() from eval.c.
    /// Note: in original C code, sides were accidentally switched for menOff.
    /// </summary>
    public static void CalculateContactInputs(Board board, Span<float> arInput)
    {
        BaseInputs(board, arInput);

        // Side 0 block (at offset 200): menOff uses Opponent (accidentally switched in training)
        {
            var b = arInput.Slice(MINPPERPOINT * 25 * 2);
            MenOffNonCrashed(board.Opponent, b.Slice(I_OFF1));
            CalculateHalfInputs(board.Player, board.Opponent, b);
        }

        // Side 1 block (at offset 225)
        {
            var b = arInput.Slice(MINPPERPOINT * 25 * 2 + MORE_INPUTS);
            MenOffNonCrashed(board.Player, b.Slice(I_OFF1));
            CalculateHalfInputs(board.Opponent, board.Player, b);
        }
    }

    /// <summary>
    /// Compute crashed neural net inputs (250 floats).
    /// Port of CalculateCrashedInputs() from eval.c.
    /// </summary>
    public static void CalculateCrashedInputs(Board board, Span<float> arInput)
    {
        BaseInputs(board, arInput);

        // Side 0 block: uses Player for menOff (swapped from contact)
        {
            var b = arInput.Slice(MINPPERPOINT * 25 * 2);
            MenOffAll(board.Player, b.Slice(I_OFF1));
            CalculateHalfInputs(board.Player, board.Opponent, b);
        }

        // Side 1 block
        {
            var b = arInput.Slice(MINPPERPOINT * 25 * 2 + MORE_INPUTS);
            MenOffAll(board.Opponent, b.Slice(I_OFF1));
            CalculateHalfInputs(board.Opponent, board.Player, b);
        }
    }

    /// <summary>
    /// Men off for crashed positions (wider buckets: 0-5, 5-10, 10-15).
    /// Port of menOffAll() from eval.c.
    /// </summary>
    private static void MenOffAll(uint[] anBoard, Span<float> afInput)
    {
        int menOff = 15;
        for (int i = 0; i < 25; i++)
            menOff -= (int)anBoard[i];

        if (menOff <= 5)
        {
            afInput[0] = menOff > 0 ? menOff / 5.0f : 0.0f;
            afInput[1] = 0.0f;
            afInput[2] = 0.0f;
        }
        else if (menOff <= 10)
        {
            afInput[0] = 1.0f;
            afInput[1] = (menOff - 5) / 5.0f;
            afInput[2] = 0.0f;
        }
        else
        {
            afInput[0] = 1.0f;
            afInput[1] = 1.0f;
            afInput[2] = (menOff - 10) / 5.0f;
        }
    }

    /// <summary>
    /// Men off for contact positions (tighter buckets: 0-2, 2-5, 5-8).
    /// Port of menOffNonCrashed() from eval.c.
    /// </summary>
    private static void MenOffNonCrashed(uint[] anBoard, Span<float> afInput)
    {
        int menOff = 15;
        for (int i = 0; i < 25; i++)
            menOff -= (int)anBoard[i];

        if (menOff <= 2)
        {
            afInput[0] = menOff > 0 ? menOff / 3.0f : 0.0f;
            afInput[1] = 0.0f;
            afInput[2] = 0.0f;
        }
        else if (menOff <= 5)
        {
            afInput[0] = 1.0f;
            afInput[1] = (menOff - 3) / 3.0f;
            afInput[2] = 0.0f;
        }
        else
        {
            afInput[0] = 1.0f;
            afInput[1] = 1.0f;
            afInput[2] = (menOff - 6) / 3.0f;
        }
    }

    /// <summary>
    /// Calculate 25 contact-specific inputs for one side.
    /// Port of CalculateHalfInputs() from eval.c (~677 lines).
    /// </summary>
    private static void CalculateHalfInputs(uint[] anBoard, uint[] anBoardOpp, Span<float> afInput)
    {
        // aanCombination[n] - ways to hit from distance n+1
        int[][] aanCombination =
        [
            [0, -1, -1, -1, -1],    /*  1 */
            [1, 2, -1, -1, -1],     /*  2 */
            [3, 4, 5, -1, -1],      /*  3 */
            [6, 7, 8, 9, -1],       /*  4 */
            [10, 11, 12, -1, -1],   /*  5 */
            [13, 14, 15, 16, 17],   /*  6 */
            [18, 19, 20, -1, -1],   /*  7 */
            [21, 22, 23, 24, -1],   /*  8 */
            [25, 26, 27, -1, -1],   /*  9 */
            [28, 29, -1, -1, -1],   /* 10 */
            [30, -1, -1, -1, -1],   /* 11 */
            [31, 32, 33, -1, -1],   /* 12 */
            [-1, -1, -1, -1, -1],   /* 13 */
            [-1, -1, -1, -1, -1],   /* 14 */
            [34, -1, -1, -1, -1],   /* 15 */
            [35, -1, -1, -1, -1],   /* 16 */
            [-1, -1, -1, -1, -1],   /* 17 */
            [36, -1, -1, -1, -1],   /* 18 */
            [-1, -1, -1, -1, -1],   /* 19 */
            [37, -1, -1, -1, -1],   /* 20 */
            [-1, -1, -1, -1, -1],   /* 21 */
            [-1, -1, -1, -1, -1],   /* 22 */
            [-1, -1, -1, -1, -1],   /* 23 */
            [38, -1, -1, -1, -1],   /* 24 */
        ];

        // aIntermediate: [fAll, inter0, inter1, inter2, nFaces, nPips]
        (bool fAll, int[] inter, int nFaces, int nPips)[] aIntermediate =
        [
            (true,  [0, 0, 0], 1, 1),   /*  0: 1x hits 1 */
            (true,  [0, 0, 0], 1, 2),   /*  1: 2x hits 2 */
            (true,  [1, 0, 0], 2, 2),   /*  2: 11 hits 2 */
            (true,  [0, 0, 0], 1, 3),   /*  3: 3x hits 3 */
            (false, [1, 2, 0], 2, 3),   /*  4: 21 hits 3 */
            (true,  [1, 2, 0], 3, 3),   /*  5: 11 hits 3 */
            (true,  [0, 0, 0], 1, 4),   /*  6: 4x hits 4 */
            (false, [1, 3, 0], 2, 4),   /*  7: 31 hits 4 */
            (true,  [2, 0, 0], 2, 4),   /*  8: 22 hits 4 */
            (true,  [1, 2, 3], 4, 4),   /*  9: 11 hits 4 */
            (true,  [0, 0, 0], 1, 5),   /* 10: 5x hits 5 */
            (false, [1, 4, 0], 2, 5),   /* 11: 41 hits 5 */
            (false, [2, 3, 0], 2, 5),   /* 12: 32 hits 5 */
            (true,  [0, 0, 0], 1, 6),   /* 13: 6x hits 6 */
            (false, [1, 5, 0], 2, 6),   /* 14: 51 hits 6 */
            (false, [2, 4, 0], 2, 6),   /* 15: 42 hits 6 */
            (true,  [3, 0, 0], 2, 6),   /* 16: 33 hits 6 */
            (true,  [2, 4, 0], 3, 6),   /* 17: 22 hits 6 */
            (false, [1, 6, 0], 2, 7),   /* 18: 61 hits 7 */
            (false, [2, 5, 0], 2, 7),   /* 19: 52 hits 7 */
            (false, [3, 4, 0], 2, 7),   /* 20: 43 hits 7 */
            (false, [2, 6, 0], 2, 8),   /* 21: 62 hits 8 */
            (false, [3, 5, 0], 2, 8),   /* 22: 53 hits 8 */
            (true,  [4, 0, 0], 2, 8),   /* 23: 44 hits 8 */
            (true,  [2, 4, 6], 4, 8),   /* 24: 22 hits 8 */
            (false, [3, 6, 0], 2, 9),   /* 25: 63 hits 9 */
            (false, [4, 5, 0], 2, 9),   /* 26: 54 hits 9 */
            (true,  [3, 6, 0], 3, 9),   /* 27: 33 hits 9 */
            (false, [4, 6, 0], 2, 10),  /* 28: 64 hits 10 */
            (true,  [5, 0, 0], 2, 10),  /* 29: 55 hits 10 */
            (false, [5, 6, 0], 2, 11),  /* 30: 65 hits 11 */
            (true,  [6, 0, 0], 2, 12),  /* 31: 66 hits 12 */
            (true,  [4, 8, 0], 3, 12),  /* 32: 44 hits 12 */
            (true,  [3, 6, 9], 4, 12),  /* 33: 33 hits 12 */
            (true,  [5, 10, 0], 3, 15), /* 34: 55 hits 15 */
            (true,  [4, 8, 12], 4, 16), /* 35: 44 hits 16 */
            (true,  [6, 12, 0], 3, 18), /* 36: 66 hits 18 */
            (true,  [5, 10, 15], 4, 20),/* 37: 55 hits 20 */
            (true,  [6, 12, 18], 4, 24),/* 38: 66 hits 24 */
        ];

        int[][] aaRoll =
        [
            [0, 2, 5, 9],           /* 11 */
            [1, 8, 17, 24],         /* 22 */
            [3, 16, 27, 33],        /* 33 */
            [6, 23, 32, 35],        /* 44 */
            [10, 29, 34, 37],       /* 55 */
            [13, 31, 36, 38],       /* 66 */
            [0, 1, 4, -1],          /* 21 */
            [0, 3, 7, -1],          /* 31 */
            [1, 3, 12, -1],         /* 32 */
            [0, 6, 11, -1],         /* 41 */
            [1, 6, 15, -1],         /* 42 */
            [3, 6, 20, -1],         /* 43 */
            [0, 10, 14, -1],        /* 51 */
            [1, 10, 19, -1],        /* 52 */
            [3, 10, 22, -1],        /* 53 */
            [6, 10, 26, -1],        /* 54 */
            [0, 13, 18, -1],        /* 61 */
            [1, 13, 21, -1],        /* 62 */
            [3, 13, 25, -1],        /* 63 */
            [6, 13, 28, -1],        /* 64 */
            [10, 13, 30, -1],       /* 65 */
        ];

        int nOppBack;
        int[] aHit = new int[39];
        int nBoard;

        // ---- I_BREAK_CONTACT ----
        {
            int np = 0;
            for (nOppBack = 24; nOppBack >= 0; --nOppBack)
                if (anBoardOpp[nOppBack] > 0) break;
            nOppBack = 23 - nOppBack;

            for (int i = nOppBack + 1; i < 25; i++)
                if (anBoard[i] > 0)
                    np += (i + 1 - nOppBack) * (int)anBoard[i];
            afInput[I_BREAK_CONTACT] = np / (15 + 152.0f);
        }

        // ---- I_FREEPIP ----
        {
            uint p = 0;
            for (int i = 0; i < nOppBack; i++)
                if (anBoard[i] > 0)
                    p += (uint)(i + 1) * anBoard[i];
            afInput[I_FREEPIP] = p / 100.0f;
        }

        // ---- I_TIMING ----
        {
            int t = 0, no = 0;
            int m = (nOppBack >= 11) ? nOppBack : 11;
            t += 24 * (int)anBoard[24];
            no += (int)anBoard[24];

            for (int i = 23; i > m; --i)
            {
                if (anBoard[i] > 0 && anBoard[i] != 2)
                {
                    int ns = (anBoard[i] > 2) ? (int)(anBoard[i] - 2) : 1;
                    no += ns;
                    t += i * ns;
                }
            }
            for (int i = m; i >= 6; --i)
            {
                if (anBoard[i] > 0)
                {
                    int nc = (int)anBoard[i];
                    no += nc;
                    t += i * nc;
                }
            }
            for (int i = 5; i >= 0; --i)
            {
                if (anBoard[i] > 2)
                {
                    t += i * (int)(anBoard[i] - 2);
                    no += (int)(anBoard[i] - 2);
                }
                else if (anBoard[i] < 2)
                {
                    int nm = 2 - (int)anBoard[i];
                    if (no >= nm)
                    {
                        t -= i * nm;
                        no -= nm;
                    }
                }
            }
            afInput[I_TIMING] = t / 100.0f;
        }

        // ---- I_BACK_CHEQUER, I_BACK_ANCHOR, I_FORWARD_ANCHOR ----
        {
            int nBack;
            for (nBack = 24; nBack >= 0; --nBack)
                if (anBoard[nBack] > 0) break;
            afInput[I_BACK_CHEQUER] = nBack / 24.0f;

            int i;
            for (i = ((nBack == 24) ? 23 : nBack); i >= 0; --i)
                if (anBoard[i] >= 2) break;
            afInput[I_BACK_ANCHOR] = i / 24.0f;

            int n = 0;
            for (int j = 18; j <= i; ++j)
            {
                if (anBoard[j] >= 2)
                {
                    n = 24 - j;
                    break;
                }
            }
            if (n == 0)
            {
                for (int j = 17; j >= 12; --j)
                {
                    if (anBoard[j] >= 2)
                    {
                        n = 24 - j;
                        break;
                    }
                }
            }
            afInput[I_FORWARD_ANCHOR] = n == 0 ? 2.0f : n / 6.0f;
        }

        // ---- Piploss (I_PIPLOSS, I_P1, I_P2) ----
        nBoard = 0;
        for (int i = 0; i < 6; i++)
            if (anBoard[i] >= 2) nBoard++;

        // For every blot we could hit
        for (int i = (nBoard > 2) ? 23 : 21; i >= 0; i--)
        {
            if (anBoardOpp[i] != 1) continue;

            for (int j = 24 - i; j < 25; j++)
            {
                if (anBoard[j] == 0 || (j < 6 && anBoard[j] == 2)) continue;

                for (int n = 0; n < 5; n++)
                {
                    int combIdx = aanCombination[j - 24 + i][n];
                    if (combIdx == -1) break;

                    var pi = aIntermediate[combIdx];

                    if (pi.fAll)
                    {
                        if (pi.nFaces > 1)
                        {
                            bool blocked = false;
                            for (int k = 0; k < 3 && pi.inter[k] > 0; k++)
                            {
                                if (anBoardOpp[i - pi.inter[k]] > 1)
                                {
                                    blocked = true;
                                    break;
                                }
                            }
                            if (blocked) continue;
                        }
                    }
                    else
                    {
                        if (anBoardOpp[i - pi.inter[0]] > 1 && anBoardOpp[i - pi.inter[1]] > 1)
                            continue;
                    }

                    aHit[combIdx] |= 1 << j;
                }
            }
        }

        int[] rollChequers = new int[21];
        int[] rollPips = new int[21];

        if (anBoard[24] == 0)
        {
            // Not on bar
            for (int i = 0; i < 21; i++)
            {
                int hitterUsed = -1;
                for (int j = 0; j < 4; j++)
                {
                    int r = aaRoll[i][j];
                    if (r < 0) break;
                    if (aHit[r] == 0) continue;

                    var pi = aIntermediate[r];
                    if (pi.nFaces == 1)
                    {
                        int k = Msb32(aHit[r]);
                        if (hitterUsed != k || anBoard[k] > 1)
                            rollChequers[i]++;
                        hitterUsed = k;
                        if (k - pi.nPips + 1 > rollPips[i])
                            rollPips[i] = k - pi.nPips + 1;
                        if (aaRoll[i][3] >= 0 && (aHit[r] & ~(1 << k)) != 0)
                            rollChequers[i]++;
                    }
                    else
                    {
                        if (rollChequers[i] == 0) rollChequers[i] = 1;
                        int k = Msb32(aHit[r]);
                        if (k - pi.nPips + 1 > rollPips[i])
                            rollPips[i] = k - pi.nPips + 1;
                        for (int l = 0; l < 3 && pi.inter[l] > 0; l++)
                        {
                            if (anBoardOpp[23 - k + pi.inter[l]] == 1)
                            {
                                rollChequers[i]++;
                                break;
                            }
                        }
                    }
                }
            }
        }
        else if (anBoard[24] == 1)
        {
            // One on bar
            for (int i = 0; i < 21; i++)
            {
                int entryUsed = 0;
                for (int j = 0; j < 4; j++)
                {
                    int r = aaRoll[i][j];
                    if (r < 0) break;
                    if (aHit[r] == 0) continue;

                    var pi = aIntermediate[r];
                    if (pi.nFaces == 1)
                    {
                        for (int k = 24; k > 0; k--)
                        {
                            if ((aHit[r] & (1 << k)) == 0) continue;
                            if (entryUsed != 0 && k != 24) break;
                            if (k != 24)
                            {
                                int npip = aIntermediate[aaRoll[i][1 - j]].nPips;
                                if (anBoardOpp[npip - 1] > 1) break;
                                entryUsed = 1;
                            }
                            rollChequers[i]++;
                            if (k - pi.nPips + 1 > rollPips[i])
                                rollPips[i] = k - pi.nPips + 1;
                        }
                    }
                    else
                    {
                        if ((aHit[r] & (1 << 24)) == 0) continue;
                        if (rollChequers[i] == 0) rollChequers[i] = 1;
                        if (25 - pi.nPips > rollPips[i])
                            rollPips[i] = 25 - pi.nPips;
                        for (int k = 0; k < 3 && pi.inter[k] > 0; k++)
                        {
                            if (anBoardOpp[pi.inter[k] + 1] == 1)
                            {
                                rollChequers[i]++;
                                break;
                            }
                        }
                    }
                }
            }
        }
        else
        {
            // Multiple on bar
            for (int i = 0; i < 21; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    int r = aaRoll[i][j];
                    if ((aHit[r] & (1 << 24)) == 0) continue;
                    var pi = aIntermediate[r];
                    if (pi.nFaces != 1) continue;
                    rollChequers[i]++;
                    if (25 - pi.nPips > rollPips[i])
                        rollPips[i] = 25 - pi.nPips;
                }
            }
        }

        // Aggregate piploss, p1, p2
        {
            int np = 0, n1 = 0, n2 = 0;
            for (int i = 0; i < 6; i++)
            {
                np += rollPips[i];
                if (rollChequers[i] > 0) { n1 += 1; if (rollChequers[i] > 1) n2 += 1; }
            }
            for (int i = 6; i < 21; i++)
            {
                np += rollPips[i] * 2;
                if (rollChequers[i] > 0) { n1 += 2; if (rollChequers[i] > 1) n2 += 2; }
            }
            afInput[I_PIPLOSS] = np / (12.0f * 36.0f);
            afInput[I_P1] = n1 / 36.0f;
            afInput[I_P2] = n2 / 36.0f;
        }

        // ---- Escapes ----
        afInput[I_BACKESCAPES] = EscapeTable.GetEscapes(anBoard, 23 - nOppBack) / 36.0f;
        afInput[I_BACKRESCAPES] = EscapeTable.GetEscapes1(anBoard, 23 - nOppBack) / 36.0f;

        // ---- Containment ----
        {
            int n = 36;
            int i;
            for (i = 15; i < 24 - nOppBack; i++)
            {
                int j = EscapeTable.GetEscapes(anBoard, i);
                if (j < n) n = j;
            }
            afInput[I_ACONTAIN] = (36 - n) / 36.0f;
            afInput[I_ACONTAIN2] = afInput[I_ACONTAIN] * afInput[I_ACONTAIN];

            if (nOppBack < 0) { i = 15; n = 36; }
            for (; i < 24; i++)
            {
                int j = EscapeTable.GetEscapes(anBoard, i);
                if (j < n) n = j;
            }
            afInput[I_CONTAIN] = (36 - n) / 36.0f;
            afInput[I_CONTAIN2] = afInput[I_CONTAIN] * afInput[I_CONTAIN];
        }

        // ---- Mobility ----
        {
            int n = 0;
            for (int i = 6; i < 25; i++)
                if (anBoard[i] > 0)
                    n += (i - 5) * (int)anBoard[i] * EscapeTable.GetEscapes(anBoardOpp, i);
            afInput[I_MOBILITY] = n / 3600.0f;
        }

        // ---- Moment2 ----
        {
            int j = 0, n = 0;
            for (int i = 0; i < 25; i++)
            {
                if (anBoard[i] > 0)
                {
                    j += (int)anBoard[i];
                    n += i * (int)anBoard[i];
                }
            }
            n = (n + j - 1) / j;
            int k2 = 0;
            j = 0;
            for (int i = n + 1; i < 25; i++)
            {
                if (anBoard[i] > 0)
                {
                    j += (int)anBoard[i];
                    k2 += (int)anBoard[i] * (i - n) * (i - n);
                }
            }
            if (j > 0) k2 = (k2 + j - 1) / j;
            afInput[I_MOMENT2] = k2 / 400.0f;
        }

        // ---- Enter (bar entry loss) ----
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
                    {
                        if (anBoardOpp[j] > 1)
                            loss += 2 * (i + j + 2);
                        else if (two)
                            loss += 2 * (i + 1);
                    }
                }
                else if (two)
                {
                    for (int j = i + 1; j < 6; ++j)
                        if (anBoardOpp[j] > 1)
                            loss += 2 * (j + 1);
                }
            }
            afInput[I_ENTER] = loss / (36.0f * (49.0f / 6.0f));
        }
        else
        {
            afInput[I_ENTER] = 0.0f;
        }

        // ---- Enter2 ----
        {
            int n = 0;
            for (int i = 0; i < 6; i++)
                n += anBoardOpp[i] > 1 ? 1 : 0;
            afInput[I_ENTER2] = (36 - (n - 6) * (n - 6)) / 36.0f;
        }

        // ---- Backbone ----
        {
            int pa = -1, w = 0, tot = 0;
            int[] ac = [11, 11, 11, 11, 11, 11, 11, 6, 5, 4, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

            for (int np2 = 23; np2 > 0; --np2)
            {
                if (anBoard[np2] >= 2)
                {
                    if (pa == -1) { pa = np2; continue; }
                    int d = pa - np2;
                    w += ac[d] * (int)anBoard[pa];
                    tot += (int)anBoard[pa];
                }
            }
            afInput[I_BACKBONE] = tot > 0 ? 1.0f - (w / (tot * 11.0f)) : 0.0f;
        }

        // ---- Back game ----
        {
            uint nAc = 0;
            for (int i = 18; i < 24; ++i)
                if (anBoard[i] > 1) ++nAc;

            afInput[I_BACKG] = 0.0f;
            afInput[I_BACKG1] = 0.0f;

            if (nAc >= 1)
            {
                uint tot = 0;
                for (int i = 18; i < 25; ++i)
                    tot += anBoard[i];

                if (nAc > 1)
                    afInput[I_BACKG] = (tot - 3) / 4.0f;
                else
                    afInput[I_BACKG1] = tot / 8.0f;
            }
        }
    }

    private static int Msb32(int n)
    {
        int b = 0;
        if (n >= 1 << 16) { b += 16; n >>= 16; }
        if (n >= 1 << 8) { b += 8; n >>= 8; }
        if (n >= 1 << 4) { b += 4; n >>= 4; }
        if (n >= 1 << 2) { b += 2; n >>= 2; }
        if (n >= 1 << 1) { b += 1; }
        return b;
    }
}
