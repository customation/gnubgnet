// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2014 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later

namespace GnuBgNet;

/// <summary>
/// Cube and match state used for equity calculations.
/// </summary>
public sealed class CubeInfo
{
    /// <summary>Current cube value (1, 2, 4, 8, ...).</summary>
    public int Cube { get; set; } = 1;

    /// <summary>Cube owner: -1 = centered, 0 = player 0, 1 = player 1.</summary>
    public int CubeOwner { get; set; } = -1;

    /// <summary>Player on roll (0 or 1).</summary>
    public int Move { get; set; }

    /// <summary>Match length. 0 = money game.</summary>
    public int MatchTo { get; set; }

    /// <summary>Current scores [player0, player1].</summary>
    public int[] Score { get; set; } = [0, 0];

    /// <summary>Crawford rule in effect.</summary>
    public bool Crawford { get; set; }

    /// <summary>Jacoby rule (money game only).</summary>
    public bool Jacoby { get; set; }

    /// <summary>Beavers allowed (money game only).</summary>
    public bool Beavers { get; set; }

    /// <summary>Gammon prices [winGammon, loseGammon, winBackgammon, loseBackgammon].</summary>
    public float[] GammonPrice { get; set; } = new float[4];

    /// <summary>Game variation.</summary>
    public BackgammonVariation Variation { get; set; } = BackgammonVariation.Standard;

    /// <summary>Creates a default money-game cube info.</summary>
    public static CubeInfo Money() => new()
    {
        Cube = 1,
        CubeOwner = -1,
        Move = 0,
        MatchTo = 0,
        Score = [0, 0],
        Crawford = false,
        Jacoby = false,
        Beavers = true,
        GammonPrice = [2f, 2f, 3f, 3f],
        Variation = BackgammonVariation.Standard,
    };
}
