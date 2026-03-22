// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2014 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later

using System.Runtime.CompilerServices;

namespace GnuBgNet;

/// <summary>
/// Gammon prices stored inline (no heap allocation).
/// </summary>
[InlineArray(4)]
public struct GammonPriceArray
{
    private float _element0;
}

/// <summary>
/// Cube and match state used for equity calculations.
/// Port of cubeinfo from eval.h — in C this is a stack struct with
/// int anScore[2] inline; we use Score0/Score1 to avoid heap arrays.
/// </summary>
public struct CubeInfo
{
    /// <summary>Current cube value (1, 2, 4, 8, ...).</summary>
    public int Cube { get; set; }

    /// <summary>Cube owner: -1 = centered, 0 = player 0, 1 = player 1.</summary>
    public int CubeOwner { get; set; }

    /// <summary>Player on roll (0 or 1).</summary>
    public int Move { get; set; }

    /// <summary>Match length. 0 = money game.</summary>
    public int MatchTo { get; set; }

    /// <summary>Score for player 0.</summary>
    public int Score0 { get; set; }

    /// <summary>Score for player 1.</summary>
    public int Score1 { get; set; }

    /// <summary>Get score by player index (0 or 1).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly int GetScore(int player) => player == 0 ? Score0 : Score1;

    /// <summary>Crawford rule in effect.</summary>
    public bool Crawford { get; set; }

    /// <summary>Jacoby rule (money game only).</summary>
    public bool Jacoby { get; set; }

    /// <summary>Beavers allowed (money game only).</summary>
    public bool Beavers { get; set; }

    /// <summary>Gammon prices [winGammon, loseGammon, winBackgammon, loseBackgammon].</summary>
    public GammonPriceArray GammonPrice;

    /// <summary>Game variation.</summary>
    public BackgammonVariation Variation { get; set; }

    /// <summary>Creates a default money-game cube info.</summary>
    public static CubeInfo Money()
    {
        var ci = new CubeInfo
        {
            Cube = 1,
            CubeOwner = -1,
            Move = 0,
            MatchTo = 0,
            Score0 = 0,
            Score1 = 0,
            Crawford = false,
            Jacoby = false,
            Beavers = true,
            Variation = BackgammonVariation.Standard,
        };
        ci.GammonPrice[0] = 2f;
        ci.GammonPrice[1] = 2f;
        ci.GammonPrice[2] = 3f;
        ci.GammonPrice[3] = 3f;
        return ci;
    }
}
