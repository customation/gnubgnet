// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2014 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later

namespace GnuBgNet;

public static class Constants
{
    public const int NumPoints = 25; // 0-23 board points + 24 = bar
    public const int NumCheckers = 15;
    public const int NumOutputs = 5;
    public const int NumRolloutOutputs = 7;
    public const int NumCubefulOutputs = 4;

    // Output indices
    public const int OutputWin = 0;
    public const int OutputWinGammon = 1;
    public const int OutputWinBackgammon = 2;
    public const int OutputLoseGammon = 3;
    public const int OutputLoseBackgammon = 4;
    public const int OutputEquity = 5;
    public const int OutputCubefulEquity = 6;

    // Neural net input sizes
    public const int NumRaceInputs = 214;
    public const int NumContactInputs = 250;
    public const int NumCrashedInputs = 250;
    public const int NumPruningInputs = 200;

    // Neural net activation
    public const float BetaHidden = 0.1f;
    public const float BetaOutput = 1.0f;

    // Weight file format
    public const float WeightsMagicBinary = 472.3782f;
    public const float WeightsVersionBinary = 1.01f;

    // Cache sizes (powers of 2)
    public const int CacheSizeMainLog2 = 19;  // 524,288 entries
    public const int CacheSizePruneLog2 = 16; // 65,536 entries

    // Move limits
    public const int MaxMoves = 3060;
    public const int MaxHalfMoves = 8; // up to 4 dice × 2 (src, dest)

    // Match equity
    public const int MaxScore = 64;
    public const int MaxCubeLevel = 7;

    // Position/Match ID lengths
    public const int PositionIdLength = 14;
    public const int MatchIdLength = 12;
}
