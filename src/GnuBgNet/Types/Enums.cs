// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2014 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Ported from GNU Backgammon to .NET

namespace GnuBgNet;

public enum BackgammonVariation
{
    Standard = 0,
    Nackgammon = 1,
    Hypergammon1 = 2,
    Hypergammon2 = 3,
    Hypergammon3 = 4,
}

public enum GameState
{
    None = 0,
    Playing = 1,
    Over = 2,
    Resigned = 3,
    Drop = 4,
}

public enum PositionClass
{
    Over = 0,
    Hypergammon1 = 1,
    Hypergammon2 = 2,
    Hypergammon3 = 3,
    Bearoff2 = 4,
    BearoffTwoSided = 5,
    Bearoff1 = 6,
    BearoffOneSided = 7,
    Race = 8,
    Crashed = 9,
    Contact = 10,
}

public enum CubeDecisionType
{
    Optimal = 0,
    NoDouble = 1,
    Take = 2,
    Drop = 3,
}

/// <summary>
/// Type of double being offered.
/// Port of doubletype enum from eval.h.
/// </summary>
public enum DoubleType
{
    Normal = 0,
    Beaver = 1,
    Raccoon = 2,
}

/// <summary>
/// How a double is being taken.
/// Port of taketype enum from eval.h.
/// </summary>
public enum TakeType
{
    NotAvailable = 0,
    Normal = 1,
    Beaver = 2,
}
