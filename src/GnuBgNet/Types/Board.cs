// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2014 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later

namespace GnuBgNet;

/// <summary>
/// Backgammon board representation. Two arrays of 25 unsigned ints each:
/// points 0-23 are board points, index 24 is the bar.
/// Player = anBoard[1] in C (on-roll), Opponent = anBoard[0] in C.
/// </summary>
public sealed class Board
{
    /// <summary>Checker counts for the player on roll (anBoard[1] in C). Index 24 = bar.</summary>
    public readonly uint[] Player = new uint[Constants.NumPoints];

    /// <summary>Checker counts for the opponent (anBoard[0] in C). Index 24 = bar.</summary>
    public readonly uint[] Opponent = new uint[Constants.NumPoints];

    public Board()
    {
    }

    public Board(uint[] player, uint[] opponent)
    {
        ArgumentNullException.ThrowIfNull(player);
        ArgumentNullException.ThrowIfNull(opponent);
        if (player.Length != Constants.NumPoints)
            throw new ArgumentException("Player array must have 25 elements.", nameof(player));
        if (opponent.Length != Constants.NumPoints)
            throw new ArgumentException("Opponent array must have 25 elements.", nameof(opponent));

        Array.Copy(player, Player, Constants.NumPoints);
        Array.Copy(opponent, Opponent, Constants.NumPoints);
    }

    /// <summary>Creates a deep copy of this board.</summary>
    public Board Clone()
    {
        var b = new Board();
        Array.Copy(Player, b.Player, Constants.NumPoints);
        Array.Copy(Opponent, b.Opponent, Constants.NumPoints);
        return b;
    }

    /// <summary>Swaps player and opponent (equivalent to SwapSides in C).</summary>
    public void SwapSides()
    {
        for (int i = 0; i < Constants.NumPoints; i++)
        {
            (Player[i], Opponent[i]) = (Opponent[i], Player[i]);
        }
    }

    /// <summary>Returns a new board with sides swapped.</summary>
    public Board Swapped()
    {
        var b = new Board();
        Array.Copy(Opponent, b.Player, Constants.NumPoints);
        Array.Copy(Player, b.Opponent, Constants.NumPoints);
        return b;
    }

    /// <summary>Sets up the standard backgammon opening position.</summary>
    public static Board Opening()
    {
        var b = new Board();
        // Standard starting position. Index = point - 1 (0-based).
        // Each player: 2 on 24-pt (idx 23), 5 on 13-pt (idx 12), 3 on 8-pt (idx 7), 5 on 6-pt (idx 5)
        b.Player[5] = 5;
        b.Player[7] = 3;
        b.Player[12] = 5;
        b.Player[23] = 2;

        b.Opponent[5] = 5;
        b.Opponent[7] = 3;
        b.Opponent[12] = 5;
        b.Opponent[23] = 2;
        return b;
    }

    public bool Equals(Board other)
    {
        for (int i = 0; i < Constants.NumPoints; i++)
        {
            if (Player[i] != other.Player[i] || Opponent[i] != other.Opponent[i])
                return false;
        }
        return true;
    }
}

/// <summary>
/// Fast position key: 7 × uint32, 4 bits per point.
/// Used for evaluation caching and move generation.
/// </summary>
public struct PositionKey : IEquatable<PositionKey>
{
    public uint D0, D1, D2, D3, D4, D5, D6;

    public bool Equals(PositionKey other) =>
        D0 == other.D0 && D1 == other.D1 && D2 == other.D2 &&
        D3 == other.D3 && D4 == other.D4 && D5 == other.D5 &&
        D6 == other.D6;

    public override bool Equals(object? obj) => obj is PositionKey pk && Equals(pk);

    public override int GetHashCode() => HashCode.Combine(D0, D1, D2, D3, D4, D5, D6);

    public static bool operator ==(PositionKey left, PositionKey right) => left.Equals(right);
    public static bool operator !=(PositionKey left, PositionKey right) => !left.Equals(right);
}
