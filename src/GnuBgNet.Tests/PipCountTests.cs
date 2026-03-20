// SPDX-License-Identifier: GPL-3.0-or-later

using Xunit;
using GnuBgNet;
using GnuBgNet.Evaluation;

namespace GnuBgNet.Tests;

public class PipCountTests
{
    [Fact]
    public void PipCount_Opening_Is167Each()
    {
        var board = Board.Opening();
        var (player, opponent) = PipCount.Count(board);
        Assert.Equal(167, player);
        Assert.Equal(167, opponent);
    }

    [Fact]
    public void PipCount_EmptyBoard_IsZero()
    {
        var board = new Board();
        var (player, opponent) = PipCount.Count(board);
        Assert.Equal(0, player);
        Assert.Equal(0, opponent);
    }

    [Fact]
    public void PipCount_BearingOff_CorrectPips()
    {
        var board = new Board();
        board.Player[0] = 5;  // 5 checkers on 1-point = 5 pips
        board.Player[1] = 5;  // 5 checkers on 2-point = 10 pips
        board.Player[2] = 5;  // 5 checkers on 3-point = 15 pips
        board.Opponent[0] = 15; // all on 1-point = 15 pips

        var (player, opponent) = PipCount.Count(board);
        Assert.Equal(30, player);
        Assert.Equal(15, opponent);
    }

    [Fact]
    public void KleinmanCount_EqualPips_IsNearHalf()
    {
        float prob = PipCount.KleinmanCount(100, 100);
        Assert.InRange(prob, 0.45f, 0.65f);
    }

    [Fact]
    public void KleinmanCount_LargeLeadOnRoll_IsNearOne()
    {
        float prob = PipCount.KleinmanCount(50, 120);
        Assert.InRange(prob, 0.90f, 1.0f);
    }

    [Fact]
    public void KleinmanCount_LargeDeficit_IsNearZero()
    {
        float prob = PipCount.KleinmanCount(120, 50);
        Assert.InRange(prob, 0.0f, 0.10f);
    }

    [Fact]
    public void KeithCount_Opening_ReturnsAdjustedCounts()
    {
        var board = Board.Opening();
        var (player, opponent) = PipCount.KeithCount(board);
        // Keith count adds wastage penalties: gaps in 4-6 points, stacking on 1-2 points
        Assert.True(player > 167, $"Keith count should exceed raw pips, got {player}");
        Assert.True(opponent > 167, $"Keith count should exceed raw pips, got {opponent}");
        Assert.Equal(player, opponent); // symmetric position
    }

    [Fact]
    public void IsightCount_Opening_ReturnsAdjustedCounts()
    {
        var board = Board.Opening();
        var (player, opponent) = PipCount.IsightCount(board);
        Assert.True(player >= 167, $"Isight count should >= raw pips, got {player}");
        Assert.Equal(player, opponent); // symmetric position
    }

    [Fact]
    public void ThorpCount_Opening_ReturnsValidCounts()
    {
        var board = Board.Opening();
        var (leader, adjusted, trailer) = PipCount.ThorpCount(board);
        Assert.True(leader > 0);
        Assert.True(adjusted > 0);
        Assert.True(trailer > 0);
        // Symmetric: leader == trailer
        Assert.Equal(leader, trailer);
    }
}

public class GameStatusTests
{
    private static string DataDir => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        @"source\gnubg");

    private static Evaluator? CreateEvaluator()
    {
        string wdPath = Path.Combine(DataDir, "gnubg.wd");
        if (!File.Exists(wdPath)) return null;
        var nets = GnuBgNet.NeuralNet.NetworkSet.LoadBinary(wdPath);
        return new Evaluator(nets);
    }

    [Fact]
    public void GameStatus_Opening_IsNotOver()
    {
        var eval = CreateEvaluator();
        if (eval == null) return;

        var board = Board.Opening();
        Assert.Equal(0, eval.GameStatus(board));
    }

    [Fact]
    public void GameStatus_AllBorneOff_IsNormalWin()
    {
        var eval = CreateEvaluator();
        if (eval == null) return;

        var board = new Board();
        // Player has no pieces (all borne off), opponent has some
        board.Opponent[0] = 5;
        board.Opponent[1] = 5;
        board.Opponent[2] = 4;
        // Player array is all zeros = all borne off

        int status = eval.GameStatus(board);
        Assert.Equal(1, status); // normal win
    }

    [Fact]
    public void GameStatus_Gammon_Returns2()
    {
        var eval = CreateEvaluator();
        if (eval == null) return;

        var board = new Board();
        // Player has no pieces, opponent has all 15
        board.Opponent[5] = 5;
        board.Opponent[6] = 5;
        board.Opponent[7] = 5;

        int status = eval.GameStatus(board);
        Assert.Equal(2, status); // gammon
    }

    [Fact]
    public void GameStatus_Backgammon_Returns3()
    {
        var eval = CreateEvaluator();
        if (eval == null) return;

        var board = new Board();
        // Player has no pieces, opponent has all 15 including in player's home board
        board.Opponent[20] = 5;
        board.Opponent[21] = 5;
        board.Opponent[22] = 5;

        int status = eval.GameStatus(board);
        Assert.Equal(3, status); // backgammon
    }
}

public class InvertEvaluationRTests
{
    [Fact]
    public void InvertEvaluationR_MoneyGame_NegatesEquities()
    {
        float[] ar = [0.6f, 0.1f, 0.01f, 0.15f, 0.02f, 0.3f, 0.4f];
        Evaluator.InvertEvaluationR(ar, isMatchPlay: false);

        Assert.InRange(ar[Constants.OutputWin], 0.39f, 0.41f);
        Assert.Equal(-0.3f, ar[Constants.OutputEquity], 0.001f);
        Assert.Equal(-0.4f, ar[Constants.OutputCubefulEquity], 0.001f);
    }

    [Fact]
    public void InvertEvaluationR_MatchPlay_ComplementsCubeful()
    {
        float[] ar = [0.6f, 0.1f, 0.01f, 0.15f, 0.02f, 0.3f, 0.7f];
        Evaluator.InvertEvaluationR(ar, isMatchPlay: true);

        Assert.Equal(-0.3f, ar[Constants.OutputEquity], 0.001f);
        Assert.Equal(0.3f, ar[Constants.OutputCubefulEquity], 0.001f); // 1.0 - 0.7
    }
}

public class EvalKeyTests
{
    [Fact]
    public void ComputeEvalKey_SimpleMoney_MatchesOld()
    {
        int key1 = EvalCache.ComputeEvalKey(2, false, true);
        int key2 = EvalCache.ComputeEvalKey(2, false, true, null, false);
        Assert.Equal(key1, key2);
    }

    [Fact]
    public void ComputeEvalKey_MatchPlay_EncodesScore()
    {
        var ci1 = new CubeInfo { MatchTo = 7, Score = [2, 3], Move = 0, Cube = 1, CubeOwner = -1 };
        var ci2 = new CubeInfo { MatchTo = 7, Score = [3, 2], Move = 0, Cube = 1, CubeOwner = -1 };

        int key1 = EvalCache.ComputeEvalKey(2, true, true, ci1, false);
        int key2 = EvalCache.ComputeEvalKey(2, true, true, ci2, false);

        Assert.NotEqual(key1, key2);
    }

    [Fact]
    public void ComputeEvalKey_CubefulEquity_DiffersFromNot()
    {
        var ci = new CubeInfo { MatchTo = 7, Score = [2, 3], Move = 0, Cube = 2, CubeOwner = 0 };

        int key1 = EvalCache.ComputeEvalKey(0, true, false, ci, false);
        int key2 = EvalCache.ComputeEvalKey(0, true, false, ci, true);

        Assert.NotEqual(key1, key2);
    }
}
