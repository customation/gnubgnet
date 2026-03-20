// Copyright (C) 1998-2003 Gary Wong <gtw@gnu.org>
// Copyright (C) 1999-2013 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Ported from eval.h (movefilter) and movefilters.inc

namespace GnuBgNet;

/// <summary>
/// Move filter for progressive pruning during n-ply search.
/// Port of movefilter struct from eval.h.
/// </summary>
public readonly struct MoveFilter
{
    /// <summary>Always allow this many moves. -1 means don't use this level.</summary>
    public int Accept { get; init; }

    /// <summary>Add up to this many more moves...</summary>
    public int Extra { get; init; }

    /// <summary>...if they are within this equity of the best.</summary>
    public float Threshold { get; init; }

    public MoveFilter(int accept, int extra = 0, float threshold = 0f)
    {
        Accept = accept;
        Extra = extra;
        Threshold = threshold;
    }

    /// <summary>A filter that skips this ply level (Accept = -1).</summary>
    public static MoveFilter Skip => new(-1);

    /// <summary>A null filter (Accept = -1) used when ply exceeds MAX_FILTER_PLIES.</summary>
    public static MoveFilter Null => new(-1);
}

/// <summary>
/// Predefined move filter presets.
/// Port of MOVEFILTER_TINY/NARROW/NORMAL/LARGE/HUGE from movefilters.inc.
///
/// Indexed as [nPlies-1][iPly] where nPlies is the total search depth (1-4)
/// and iPly is the intermediate ply being filtered (0 to nPlies-1).
/// </summary>
public static class MoveFilterPresets
{
    public const int MaxFilterPlies = 4;

    /// <summary>
    /// Tiny: Accept=0, Extra=5, Threshold=0.08 at ply 0; Skip intermediate; Extra=2/0.02 at deep ply.
    /// </summary>
    public static readonly MoveFilter[,] Tiny = BuildPreset(5, 0.08f, 2, 0.02f);

    /// <summary>
    /// Narrow: Accept=0, Extra=8, Threshold=0.12 at ply 0; Skip intermediate; Extra=2/0.03 at deep ply.
    /// </summary>
    public static readonly MoveFilter[,] Narrow = BuildPreset(8, 0.12f, 2, 0.03f);

    /// <summary>
    /// Normal: Accept=0, Extra=8, Threshold=0.16 at ply 0; Skip intermediate; Extra=2/0.04 at deep ply.
    /// </summary>
    public static readonly MoveFilter[,] Normal = BuildPreset(8, 0.16f, 2, 0.04f);

    /// <summary>
    /// Large: Accept=0, Extra=16, Threshold=0.32 at ply 0; Skip intermediate; Extra=4/0.08 at deep ply.
    /// </summary>
    public static readonly MoveFilter[,] Large = BuildPreset(16, 0.32f, 4, 0.08f);

    /// <summary>
    /// Huge: Accept=0, Extra=20, Threshold=0.44 at ply 0; Skip intermediate; Extra=6/0.11 at deep ply.
    /// </summary>
    public static readonly MoveFilter[,] Huge = BuildPreset(20, 0.44f, 6, 0.11f);

    /// <summary>Default filters (Normal).</summary>
    public static MoveFilter[,] Default => Normal;

    /// <summary>All 5 presets indexed by MoveFilterSetting enum.</summary>
    public static readonly MoveFilter[][,] All = [Tiny, Narrow, Normal, Large, Huge];

    /// <summary>
    /// Build a 4×4 filter preset array matching the C macro layout.
    /// The pattern from movefilters.inc is:
    ///   Row 0 (1-ply): [ply0Filter, zero, zero, zero]
    ///   Row 1 (2-ply): [ply0Filter, skip, zero, zero]
    ///   Row 2 (3-ply): [ply0Filter, skip, deepFilter, zero]
    ///   Row 3 (4-ply): [ply0Filter, skip, deepFilter, skip]
    /// </summary>
    private static MoveFilter[,] BuildPreset(int extra0, float thresh0, int extraDeep, float threshDeep)
    {
        var f = new MoveFilter[MaxFilterPlies, MaxFilterPlies];
        var ply0 = new MoveFilter(0, extra0, thresh0);
        var deep = new MoveFilter(0, extraDeep, threshDeep);
        var zero = new MoveFilter(0, 0, 0f);
        var skip = MoveFilter.Skip;

        // Row 0: 1-ply search
        f[0, 0] = ply0; f[0, 1] = zero; f[0, 2] = zero; f[0, 3] = zero;
        // Row 1: 2-ply search
        f[1, 0] = ply0; f[1, 1] = skip; f[1, 2] = zero; f[1, 3] = zero;
        // Row 2: 3-ply search
        f[2, 0] = ply0; f[2, 1] = skip; f[2, 2] = deep; f[2, 3] = zero;
        // Row 3: 4-ply search
        f[3, 0] = ply0; f[3, 1] = skip; f[3, 2] = deep; f[3, 3] = skip;

        return f;
    }

    /// <summary>
    /// Get the filter row for a given total search depth.
    /// Returns the filter array for intermediate plies [0..nPlies-1].
    /// </summary>
    public static MoveFilter GetFilter(MoveFilter[,] preset, int nPlies, int iPly)
    {
        int row = Math.Min(nPlies - 1, MaxFilterPlies - 1);
        if (iPly >= MaxFilterPlies)
            return MoveFilter.Null;
        return preset[row, iPly];
    }
}

/// <summary>Move filter preset names.</summary>
public enum MoveFilterSetting
{
    Tiny = 0,
    Narrow = 1,
    Normal = 2,
    Large = 3,
    Huge = 4,
}
