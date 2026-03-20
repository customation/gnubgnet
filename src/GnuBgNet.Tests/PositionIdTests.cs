// SPDX-License-Identifier: GPL-3.0-or-later

using Xunit;
using GnuBgNet;
using GnuBgNet.Encoding;

namespace GnuBgNet.Tests;

public class PositionIdTests
{
    [Fact]
    public void Encode_OpeningPosition_Returns4HPwATDgc()
    {
        var board = Board.Opening();
        var id = PositionId.Encode(board);
        Assert.Equal("4HPwATDgc/ABMA", id);
    }

    [Fact]
    public void Decode_OpeningPositionId_ReturnsCorrectBoard()
    {
        var board = PositionId.Decode("4HPwATDgc/ABMA");
        Assert.NotNull(board);

        var expected = Board.Opening();
        Assert.True(expected.Equals(board));
    }

    [Fact]
    public void Encode_Decode_RoundTrip_OpeningPosition()
    {
        var original = Board.Opening();
        var id = PositionId.Encode(original);
        var decoded = PositionId.Decode(id);

        Assert.NotNull(decoded);
        Assert.True(original.Equals(decoded));
    }

    [Fact]
    public void Encode_Decode_RoundTrip_EmptyBoard()
    {
        var board = new Board();
        var id = PositionId.Encode(board);
        var decoded = PositionId.Decode(id);

        Assert.NotNull(decoded);
        Assert.True(board.Equals(decoded));
    }

    [Fact]
    public void Encode_Decode_RoundTrip_AllOnBar()
    {
        var board = new Board();
        board.Player[24] = 15;
        board.Opponent[24] = 15;

        var id = PositionId.Encode(board);
        var decoded = PositionId.Decode(id);

        Assert.NotNull(decoded);
        Assert.True(board.Equals(decoded));
    }

    [Fact]
    public void Encode_Decode_RoundTrip_BearoffPosition()
    {
        // Player has all checkers on the 1-point, opponent borne off
        var board = new Board();
        board.Player[0] = 15;

        var id = PositionId.Encode(board);
        var decoded = PositionId.Decode(id);

        Assert.NotNull(decoded);
        Assert.True(board.Equals(decoded));
    }

    [Fact]
    public void Encode_Decode_RoundTrip_SingleCheckerPositions()
    {
        // Player has checkers on various points; opponent on non-overlapping points.
        // Overlap rule: Player[i] and Opponent[23-i] cannot both be nonzero.
        var board = new Board();
        board.Player[0] = 1;   // Opponent[23] must be 0
        board.Player[5] = 2;   // Opponent[18] must be 0
        board.Player[7] = 1;   // Opponent[16] must be 0
        board.Player[12] = 3;  // Opponent[11] must be 0
        // Opponent on points that don't conflict
        board.Opponent[0] = 2;  // Player[23] must be 0 ✓
        board.Opponent[5] = 3;  // Player[18] must be 0 ✓
        board.Opponent[7] = 2;  // Player[16] must be 0 ✓

        var id = PositionId.Encode(board);
        var decoded = PositionId.Decode(id);

        Assert.NotNull(decoded);
        Assert.True(board.Equals(decoded));
    }

    [Fact]
    public void PositionId_Length_Is14()
    {
        var board = Board.Opening();
        var id = PositionId.Encode(board);
        Assert.Equal(14, id.Length);
    }

    [Fact]
    public void ToKey_FromKey_RoundTrip()
    {
        var original = Board.Opening();
        var key = PositionId.ToKey(original);
        var decoded = PositionId.FromKey(key);

        Assert.True(original.Equals(decoded));
    }

    [Fact]
    public void FromKeySwapped_ProducesSwappedBoard()
    {
        var original = Board.Opening();
        var key = PositionId.ToKey(original);

        var swapped = PositionId.FromKeySwapped(key);
        var manualSwap = PositionId.FromKey(key);
        manualSwap.SwapSides();

        Assert.True(swapped.Equals(manualSwap));
    }

    [Fact]
    public void EncodeFromKey_MatchesEncode()
    {
        var board = Board.Opening();
        var key = PositionId.ToKey(board);

        var fromBoard = PositionId.Encode(board);
        var fromKey = PositionId.EncodeFromKey(key);

        Assert.Equal(fromBoard, fromKey);
    }

    [Fact]
    public void CheckPosition_OpeningPosition_IsValid()
    {
        var board = Board.Opening();
        Assert.True(PositionId.CheckPosition(board));
    }

    [Fact]
    public void CheckPosition_TooManyCheckers_IsInvalid()
    {
        var board = new Board();
        board.Player[0] = 16; // Over 15
        Assert.False(PositionId.CheckPosition(board));
    }

    [Fact]
    public void CheckPosition_OverlappingCheckers_IsInvalid()
    {
        // Opponent[i] and Player[23-i] on same physical point
        var board = new Board();
        board.Opponent[0] = 2;     // Opponent on their 1-point
        board.Player[23] = 2;     // Player on their 24-point = same physical point
        Assert.False(PositionId.CheckPosition(board));
    }

    [Fact]
    public void Decode_InvalidString_ReturnsNull()
    {
        Assert.Null(PositionId.Decode(""));
        Assert.Null(PositionId.Decode(null!));
    }

    [Fact]
    public void Combination_BasicValues()
    {
        // C(1,1) = 1
        Assert.Equal(1u, PositionId.Combination(1, 1));
        // C(5,2) = 10
        Assert.Equal(10u, PositionId.Combination(5, 2));
        // C(10,3) = 120
        Assert.Equal(120u, PositionId.Combination(10, 3));
    }

    [Fact]
    public void PositionBearoff_RoundTrip()
    {
        // 5 checkers on point 0 (6 points, 15 chequers capacity)
        uint[] board = [5, 3, 2, 2, 1, 2];
        var id = PositionId.PositionBearoff(board, 6, 15);

        Span<uint> decoded = stackalloc uint[6];
        PositionId.PositionFromBearoff(decoded, id, 6, 15);

        for (int i = 0; i < 6; i++)
            Assert.Equal(board[i], decoded[i]);
    }
}
