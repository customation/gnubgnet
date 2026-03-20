// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2019 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Ported from eval.h (evalcontext, evalsetup, evaltype)

namespace GnuBgNet;

/// <summary>
/// Evaluation context controlling search depth and noise.
/// Port of evalcontext from eval.h.
/// </summary>
public sealed class EvalContext
{
    /// <summary>Whether to compute cubeful equity.</summary>
    public bool Cubeful { get; set; }

    /// <summary>Search depth in plies (0-4).</summary>
    public int Plies { get; set; }

    /// <summary>Use pruning nets to filter moves during search.</summary>
    public bool UsePrune { get; set; }

    /// <summary>Deterministic evaluation (MD5-based noise, reproducible).</summary>
    public bool Deterministic { get; set; } = true;

    /// <summary>Standard deviation of noise added to evaluations (0 = no noise).</summary>
    public float Noise { get; set; }

    public static EvalContext ZeroPly() => new() { Plies = 0, Cubeful = true, Deterministic = true };

    public static EvalContext WorldClass() => new() { Plies = 2, Cubeful = true, UsePrune = true, Deterministic = true };

    /// <summary>Clone this context.</summary>
    public EvalContext Clone() => new()
    {
        Cubeful = Cubeful,
        Plies = Plies,
        UsePrune = UsePrune,
        Deterministic = Deterministic,
        Noise = Noise,
    };
}

/// <summary>
/// Type of evaluation performed.
/// Port of evaltype from eval.h.
/// </summary>
public enum EvalType
{
    None = 0,
    Eval = 1,
    Rollout = 2,
}

/// <summary>
/// Evaluation setup: combines evaluation type with context.
/// Port of evalsetup from eval.h.
/// </summary>
public sealed class EvalSetup
{
    public EvalType Type { get; set; } = EvalType.Eval;
    public EvalContext Context { get; set; } = new();
}
