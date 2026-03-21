// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2014 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later

namespace GnuBgNet;

/// <summary>
/// Configuration for Monte Carlo rollout simulations.
/// Port of rolloutcontext from eval.h.
/// </summary>
public sealed class RolloutSettings
{
    public uint Trials { get; init; } = 1296;
    public bool Cubeful { get; init; } = true;
    public bool VarianceReduction { get; init; } = true;
    public int ChequerPlies { get; init; } = 0;
    public int CubePlies { get; init; } = 2;
    public uint Seed { get; init; } = 0;
    public bool Truncate { get; init; } = true;
    public int TruncatePlies { get; init; } = 10;
    public bool TruncateBearoff2 { get; init; } = true;
    public bool TruncateBearoffOS { get; init; } = true;
    public bool Rotate { get; init; } = true;
    public bool StopOnStdDev { get; init; }
    public float StdDevLimit { get; init; } = 0.01f;
    public bool StopOnJsd { get; init; }
    public float JsdLimit { get; init; } = 0.0f;
    public uint MinimumGames { get; init; } = 144;
    public uint MinimumJsdGames { get; init; } = 144;

    /// <summary>
    /// Enable late evaluations: after LateMoveThreshold moves, switch to
    /// ChequerPliesLate/CubePliesLate for chequer and cube decisions.
    /// Port of fLateEvals from rolloutcontext in eval.h.
    /// </summary>
    public bool LateEvals { get; init; }

    /// <summary>
    /// Move number at which to switch to late evaluation contexts.
    /// Port of nLate from rolloutcontext in eval.h.
    /// </summary>
    public int LateMoveThreshold { get; init; } = 5;

    /// <summary>
    /// Chequer play plies to use after LateMoveThreshold moves.
    /// Port of aecChequerLate from rolloutcontext.
    /// </summary>
    public int ChequerPliesLate { get; init; } = 0;

    /// <summary>
    /// Cube decision plies to use after LateMoveThreshold moves.
    /// Port of aecCubeLate from rolloutcontext.
    /// </summary>
    public int CubePliesLate { get; init; } = 0;

    /// <summary>
    /// Initial position rollout: the position is the opening position before any moves.
    /// When true: no doubles on first roll, no cube decisions on turn 0,
    /// variance reduction divides by 30 instead of 36.
    /// Port of fInitial from rolloutcontext in eval.h.
    /// </summary>
    public bool Initial { get; init; }

    public static RolloutSettings Default { get; } = new();
}

/// <summary>Result of a position evaluation.</summary>
public readonly record struct EvaluationResult(
    double Win,
    double WinGammon,
    double WinBackgammon,
    double LoseGammon,
    double LoseBackgammon,
    double Equity,
    double CubefulEquity);

/// <summary>Full evaluation result with all 7 outputs.</summary>
public readonly record struct FullEvaluationResult(
    double WinProbability,
    double WinGammonProbability,
    double WinBackgammonProbability,
    double LoseGammonProbability,
    double LoseBackgammonProbability,
    double CubelessEquity,
    double CubefulEquity);

/// <summary>Full rollout result with means and standard deviations.</summary>
public readonly record struct RolloutResult(
    double WinProbability,
    double WinGammonProbability,
    double WinBackgammonProbability,
    double LoseGammonProbability,
    double LoseBackgammonProbability,
    double CubelessEquity,
    double CubefulEquity,
    double WinProbabilityStdDev,
    double WinGammonProbabilityStdDev,
    double WinBackgammonProbabilityStdDev,
    double LoseGammonProbabilityStdDev,
    double LoseBackgammonProbabilityStdDev,
    double CubelessEquityStdDev,
    double CubefulEquityStdDev);
