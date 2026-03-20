// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2019 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Ported from eval.c (ComputeTable0, ComputeTable1, Escapes, Escapes1)

namespace GnuBgNet.Evaluation;

/// <summary>
/// Precomputed escape tables for contact input calculation.
/// Port of anEscapes[0x1000] and anEscapes1[0x1000] from eval.c.
/// </summary>
internal static class EscapeTable
{
    private static readonly int[] Escapes = new int[0x1000];
    private static readonly int[] Escapes1 = new int[0x1000];

    // anPoint[n] = n >= 2 ? 1 : 0
    private static readonly int[] AnPoint = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

    static EscapeTable()
    {
        ComputeTable0();
        ComputeTable1();
    }

    private static void ComputeTable0()
    {
        for (int i = 0; i < 0x1000; i++)
        {
            int c = 0;
            for (int n0 = 0; n0 <= 5; n0++)
                for (int n1 = 0; n1 <= n0; n1++)
                    if ((i & (1 << (n0 + n1 + 1))) == 0 && !((i & (1 << n0)) != 0 && (i & (1 << n1)) != 0))
                        c += (n0 == n1) ? 1 : 2;
            Escapes[i] = c;
        }
    }

    private static void ComputeTable1()
    {
        Escapes1[0] = 0;
        for (int i = 1; i < 0x1000; i++)
        {
            int c = 0;
            int low = 0;
            while ((i & (1 << low)) == 0)
                ++low;

            for (int n0 = 0; n0 <= 5; n0++)
                for (int n1 = 0; n1 <= n0; n1++)
                    if ((n0 + n1 + 1 > low) && (i & (1 << (n0 + n1 + 1))) == 0
                        && !((i & (1 << n0)) != 0 && (i & (1 << n1)) != 0))
                        c += (n0 == n1) ? 1 : 2;
            Escapes1[i] = c;
        }
    }

    /// <summary>
    /// Count dice rolls that allow escape past point n.
    /// Port of Escapes() from eval.c.
    /// </summary>
    public static int GetEscapes(ReadOnlySpan<uint> anBoard, int n)
    {
        int af = 0;
        int m = (n < 12) ? n : 12;
        for (int i = 0; i < m; i++)
            af |= (AnPoint[Math.Min(anBoard[24 + i - n], 15)] << i);
        return Escapes[af];
    }

    /// <summary>
    /// Escape count variant for rescue scenarios.
    /// Port of Escapes1() from eval.c.
    /// </summary>
    public static int GetEscapes1(ReadOnlySpan<uint> anBoard, int n)
    {
        int af = 0;
        int m = (n < 12) ? n : 12;
        for (int i = 0; i < m; i++)
            af |= (AnPoint[Math.Min(anBoard[24 + i - n], 15)] << i);
        return Escapes1[af];
    }
}
