// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2014 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace GnuBgNet;

/// <summary>
/// 25 checker counts stored inline (no heap array).
/// Matches C's unsigned int anBoard[25].
/// </summary>
[InlineArray(25)]
public struct BoardSide : IEquatable<BoardSide>
{
    private uint _element0;

    /// <summary>Get a Span over all 25 elements.</summary>
    public Span<uint> AsSpan() =>
        MemoryMarshal.CreateSpan(ref _element0, 25);

    /// <summary>Get a ReadOnlySpan over all 25 elements.</summary>
    public readonly ReadOnlySpan<uint> AsReadOnlySpan() =>
        MemoryMarshal.CreateReadOnlySpan(
            ref Unsafe.AsRef(in _element0), 25);

    public readonly bool Equals(BoardSide other) =>
        AsReadOnlySpan().SequenceEqual(other.AsReadOnlySpan());

    public override readonly bool Equals(object? obj) =>
        obj is BoardSide other && Equals(other);

    public override readonly int GetHashCode()
    {
        var hc = new HashCode();
        foreach (uint v in AsReadOnlySpan())
            hc.Add(v);
        return hc.ToHashCode();
    }
}

/// <summary>
/// Backgammon board representation as a value type.
/// Two inline arrays of 25 unsigned ints each:
/// points 0-23 are board points, index 24 is the bar.
/// Player = anBoard[1] in C (on-roll), Opponent = anBoard[0] in C.
/// Matches C's TanBoard (unsigned int[2][25]) — lives on the stack.
/// </summary>
public struct Board : IEquatable<Board>
{
    /// <summary>Checker counts for the player on roll (anBoard[1] in C). Index 24 = bar.</summary>
    public BoardSide Player;

    /// <summary>Checker counts for the opponent (anBoard[0] in C). Index 24 = bar.</summary>
    public BoardSide Opponent;

    public Board(ReadOnlySpan<uint> player, ReadOnlySpan<uint> opponent)
    {
        player.CopyTo(Player.AsSpan());
        opponent.CopyTo(Opponent.AsSpan());
    }

    /// <summary>Creates a deep copy of this board. With struct, this is just assignment.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Board Clone() => this;

    /// <summary>Swaps player and opponent (equivalent to SwapSides in C).</summary>
    public void SwapSides()
    {
        for (int i = 0; i < Constants.NumPoints; i++)
        {
            (Player[i], Opponent[i]) = (Opponent[i], Player[i]);
        }
    }

    /// <summary>Returns a new board with sides swapped.</summary>
    public readonly Board Swapped()
    {
        var b = new Board();
        for (int i = 0; i < Constants.NumPoints; i++)
        {
            b.Player[i] = Opponent[i];
            b.Opponent[i] = Player[i];
        }
        return b;
    }

    /// <summary>Sets up the standard backgammon opening position.</summary>
    public static Board Opening()
    {
        var b = new Board();
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

    public readonly bool Equals(in Board other) =>
        Player.Equals(other.Player) && Opponent.Equals(other.Opponent);

    public readonly bool Equals(Board other) =>
        Player.Equals(other.Player) && Opponent.Equals(other.Opponent);

    public override readonly bool Equals(object? obj) =>
        obj is Board other && Equals(other);

    public override readonly int GetHashCode() =>
        HashCode.Combine(Player.GetHashCode(), Opponent.GetHashCode());

    public static bool operator ==(Board left, Board right) => left.Equals(right);
    public static bool operator !=(Board left, Board right) => !left.Equals(right);
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
