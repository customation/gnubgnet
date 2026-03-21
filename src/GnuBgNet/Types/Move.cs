// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2014 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later

namespace GnuBgNet;

/// <summary>
/// A single backgammon move: up to 4 sub-moves encoded as src/dest pairs in AnMove[0..7].
/// -1 terminated. Source 24 = bar, dest -1 = bear off.
/// </summary>
public sealed class Move
{
    /// <summary>Source-destination pairs, -1 terminated. Up to 4 sub-moves = 8 ints.</summary>
    public int[] AnMove { get; set; } = [-1, -1, -1, -1, -1, -1, -1, -1];

    /// <summary>Position key after move is applied.</summary>
    public PositionKey Key { get; set; }

    /// <summary>Number of sub-moves actually played (1-4).</summary>
    public uint SubMoveCount { get; set; }

    /// <summary>Total pips moved.</summary>
    public uint Pips { get; set; }

    /// <summary>Primary evaluation score (cubeful if fCubeful, else cubeless equity).</summary>
    public float Score { get; set; }

    /// <summary>Secondary evaluation score (always cubeless equity).</summary>
    public float Score2 { get; set; }

    /// <summary>All 7 evaluation outputs (5 probs + cubeless + cubeful equity).</summary>
    public float[] EvalOutputs { get; set; } = new float[Constants.NumRolloutOutputs];

    /// <summary>Standard deviations for rollout outputs.</summary>
    public float[] EvalStdDev { get; set; } = new float[Constants.NumRolloutOutputs];

    /// <summary>Evaluation setup used to produce this score.</summary>
    public EvalSetup MoveEvalSetup { get; set; } = new();
}

/// <summary>
/// List of legal moves for a given position and dice roll.
/// </summary>
public sealed class MoveList
{
    public List<Move> Moves { get; set; } = [];
    public int BestIndex { get; set; } = -1;
    public float BestScore { get; set; }
    public uint MaxPips { get; set; }
    public uint MaxMoves { get; set; }
}

/// <summary>
/// Public result type for move evaluation.
/// </summary>
public readonly record struct MoveResult(
    int[] AnMove,
    string ResultPositionId,
    int SubMoveCount,
    int Pips);

/// <summary>
/// Public result type for scored moves.
/// </summary>
public readonly record struct ScoredMove(
    int[] AnMove,
    string ResultPositionId,
    int SubMoveCount,
    int Pips,
    double Equity,
    double WinProbability,
    double WinGammonProbability,
    double WinBackgammonProbability,
    double LoseGammonProbability,
    double LoseBackgammonProbability);

/// <summary>
/// A single turn in a game for analysis.
/// </summary>
public sealed class GameTurn
{
    public required string PositionId { get; init; }
    public int Die1 { get; init; }
    public int Die2 { get; init; }
    public int[] PlayedMove { get; init; } = [-1, -1, -1, -1, -1, -1, -1, -1];
    public bool IsCubeAction { get; init; }
    public bool Doubled { get; init; }
    public bool Took { get; init; }

    /// <summary>Player index (0 or 1). Used by MAT analysis to track which side is on roll.</summary>
    public int Player { get; init; }
}

/// <summary>
/// Analysis result for a single turn.
/// </summary>
public sealed class TurnAnalysis
{
    public required string PositionId { get; init; }
    public double EquityBefore { get; init; }
    public double EquityAfterBestMove { get; init; }
    public double EquityAfterPlayedMove { get; init; }
    public double EquityLoss { get; init; }
    public IReadOnlyList<ScoredMove> RankedMoves { get; init; } = [];
    public int PlayedMoveRank { get; init; }
    public MoveClassification Classification { get; init; }
}

/// <summary>
/// Move quality classification.
/// </summary>
public enum MoveClassification
{
    Best,       // equity loss = 0
    Good,       // equity loss < 0.02
    Inaccuracy, // equity loss < 0.04
    Doubtful,   // equity loss < 0.08
    Bad,        // equity loss < 0.16
    Blunder,    // equity loss >= 0.16
}

/// <summary>
/// Result of a full game analysis.
/// </summary>
public sealed class GameAnalysis
{
    public IReadOnlyList<TurnAnalysis> Turns { get; init; } = [];
    public double TotalEquityLost { get; init; }
    public double AverageEquityLoss { get; init; }
    public int ErrorCount { get; init; }
    public int BlunderCount { get; init; }
}

/// <summary>
/// GnuBG skill classification levels.
/// Port of skilltype from gnubgapi.h.
/// </summary>
public enum SkillLevel
{
    VeryBad = 0,  // blunder (equity loss > 0.12)
    Bad = 1,      // equity loss > 0.06
    Doubtful = 2, // equity loss > 0.03
    None = 3,     // good move
}

/// <summary>
/// Result of MAT file analysis with per-player statistics.
/// Port of gnubgapi_analysis_result from gnubgapi.h.
/// </summary>
public sealed class MatAnalysisResult
{
    /// <summary>Per-turn analysis details.</summary>
    public IReadOnlyList<TurnAnalysis> Turns { get; init; } = [];

    /// <summary>Total moves per player [0] and [1].</summary>
    public int[] TotalMoves { get; init; } = new int[2];

    /// <summary>Unforced moves (decisions) per player.</summary>
    public int[] UnforcedMoves { get; init; } = new int[2];

    /// <summary>Skill counts [player, skill level]. Dimensions: [2, 4].</summary>
    public int[,] SkillCounts { get; init; } = new int[2, 4];

    /// <summary>Total equity error per player.</summary>
    public float[] TotalError { get; init; } = new float[2];

    /// <summary>Error per unforced move per player.</summary>
    public float[] ErrorPerMove { get; init; } = new float[2];

    /// <summary>Millipoints per move (error_per_move × 1000).</summary>
    public float[] MillipointsPerMove { get; init; } = new float[2];

    /// <summary>Rating string per player ("Beginner" .. "Super Grandmaster").</summary>
    public string[] Rating { get; init; } = ["Beginner", "Beginner"];
}
