// SPDX-License-Identifier: GPL-3.0-or-later

namespace GnuBgNet.MatchEquity;

/// <summary>
/// Interface for match equity table lookup.
/// Allows plugging in alternative METs (Woolsey, Kazaross, custom computed, etc.).
/// </summary>
public interface IMatchEquityTable
{
    /// <summary>Pre-Crawford match equities indexed by [iAway, jAway].</summary>
    float[,] Met { get; }

    /// <summary>Post-Crawford equities indexed by [side, scoreAway].</summary>
    float[,] PostCrawford { get; }

    /// <summary>Get match equity for the given scores-away.</summary>
    float GetEquity(int iAway, int jAway);
}
