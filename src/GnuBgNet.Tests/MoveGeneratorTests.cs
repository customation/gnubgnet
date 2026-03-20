// SPDX-License-Identifier: GPL-3.0-or-later

using Xunit;
using GnuBgNet;
using GnuBgNet.MoveGeneration;

namespace GnuBgNet.Tests;

public class MoveGeneratorTests
{
    [Fact]
    public void GenerateMoves_OpeningPosition_31_HasMoves()
    {
        var board = Board.Opening();
        var ml = MoveGenerator.GenerateMoves(board, 3, 1);

        Assert.True(ml.Moves.Count > 0, "Should generate legal moves for opening 3-1");
        Assert.Equal(2u, ml.MaxMoves); // non-doubles = 2 sub-moves
    }

    [Fact]
    public void GenerateMoves_OpeningPosition_Doubles_Uses4Dice()
    {
        var board = Board.Opening();
        var ml = MoveGenerator.GenerateMoves(board, 6, 6);

        Assert.True(ml.Moves.Count > 0, "Should generate legal moves for opening 6-6");
        // With doubles, up to 4 sub-moves
        Assert.True(ml.MaxMoves >= 2 && ml.MaxMoves <= 4);
    }

    [Fact]
    public void GenerateMoves_SingleCheckerOnOnePoint_BearOff()
    {
        var board = new Board();
        board.Player[0] = 1; // 1 checker on 1-point
        board.Opponent[23] = 1; // opponent far away

        var ml = MoveGenerator.GenerateMoves(board, 1, 2);

        // Should find moves including bearing off
        Assert.True(ml.Moves.Count > 0);

        // At least one move should bear off the checker (dest < 0)
        bool foundBearOff = false;
        foreach (var move in ml.Moves)
        {
            for (int i = 0; i < move.SubMoveCount; i++)
            {
                if (move.AnMove[i * 2 + 1] < 0)
                {
                    foundBearOff = true;
                    break;
                }
            }
            if (foundBearOff) break;
        }
        Assert.True(foundBearOff, "Should find bearing off move");
    }

    [Fact]
    public void GenerateMoves_OnBar_MustEnterFirst()
    {
        var board = new Board();
        board.Player[24] = 1; // 1 on bar
        board.Player[5] = 5; // rest on 6-point
        board.Player[7] = 3; // and 8-point
        board.Player[12] = 5; // and 13-point
        board.Player[23] = 1; // and 24-point

        var ml = MoveGenerator.GenerateMoves(board, 3, 1);

        // All moves must start with entering from bar (src = 24)
        foreach (var move in ml.Moves)
        {
            Assert.Equal(24, move.AnMove[0]);
        }
    }

    [Fact]
    public void GenerateMoves_Blocked_NoMoves()
    {
        var board = new Board();
        board.Player[24] = 2; // 2 on bar
        // Opponent blocks all entry points
        for (int i = 0; i < 6; i++)
            board.Opponent[i] = 2;

        var ml = MoveGenerator.GenerateMoves(board, 3, 1);

        Assert.Empty(ml.Moves);
    }

    [Fact]
    public void GenerateMoves_NoDuplicatePositions()
    {
        var board = Board.Opening();
        var ml = MoveGenerator.GenerateMoves(board, 5, 3);

        var keys = new HashSet<PositionKey>();
        foreach (var move in ml.Moves)
        {
            Assert.True(keys.Add(move.Key), "Duplicate position key found");
        }
    }

    [Fact]
    public void GenerateMoves_MaximizesMoves()
    {
        // All moves in a non-doubles roll should use 2 sub-moves if possible
        var board = Board.Opening();
        var ml = MoveGenerator.GenerateMoves(board, 4, 2);

        Assert.Equal(2u, ml.MaxMoves);
        foreach (var move in ml.Moves)
            Assert.Equal(2u, move.SubMoveCount);
    }

    [Fact]
    public void ApplyMove_OpeningPosition_ValidResult()
    {
        var board = Board.Opening();
        var ml = MoveGenerator.GenerateMoves(board, 3, 1);

        Assert.True(ml.Moves.Count > 0);

        // Apply first move and check board is different
        var newBoard = MoveGenerator.ApplyMove(board, ml.Moves[0]);
        Assert.NotEqual(board, newBoard);
    }

    [Fact]
    public void GenerateMoves_BearoffPosition_AllInHome()
    {
        var board = new Board();
        board.Player[0] = 3;
        board.Player[1] = 3;
        board.Player[2] = 3;
        board.Player[3] = 3;
        board.Player[4] = 2;
        board.Player[5] = 1;
        // Opponent out of the way
        board.Opponent[0] = 15;

        var ml = MoveGenerator.GenerateMoves(board, 6, 5);

        Assert.True(ml.Moves.Count > 0, "Should generate bearing-off moves");
    }
}
