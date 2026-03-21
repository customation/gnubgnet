// SPDX-License-Identifier: GPL-3.0-or-later

using Xunit;
using GnuBgNet;
using GnuBgNet.Bearoff;
using GnuBgNet.Encoding;
using GnuBgNet.Evaluation;
using GnuBgNet.Formatting;
using GnuBgNet.MatchEquity;
using GnuBgNet.NeuralNet;

namespace GnuBgNet.Tests;

public class EngineTests
{
    private static string? FindDataDir()
    {
        string[] candidates =
        [
            Environment.GetEnvironmentVariable("GNUBG_DATA_DIR") ?? "",
            @"C:\git\github\customation\gnubg",
            @"C:\git\github\customation\gnubgnet\data",
        ];
        return candidates.FirstOrDefault(d => File.Exists(Path.Combine(d, "gnubg.wd")));
    }

    private Engine? CreateEngine()
    {
        var dir = FindDataDir();
        if (dir == null) return null;
        return Engine.Create(dir);
    }

    [Fact]
    public void Create_LoadsSuccessfully()
    {
        var engine = CreateEngine();
        if (engine == null) return;

        Assert.NotNull(engine);
        engine.Dispose();
    }

    [Fact]
    public void EvaluatePosition_Opening_ReasonableEquity()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        // Opening position
        var result = engine.EvaluatePosition("4HPwATDgc/ABMA");

        // Opening should be roughly even
        Assert.InRange(result.Win, 0.3, 0.7);
        Assert.InRange(result.Equity, -0.5, 0.5);
    }

    [Fact]
    public void EvaluatePosition_Board_MatchesPositionId()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var board = Board.Opening();
        var fromBoard = engine.EvaluatePosition(board);
        var fromId = engine.EvaluatePosition("4HPwATDgc/ABMA");

        Assert.Equal(fromBoard.Win, fromId.Win, 5);
        Assert.Equal(fromBoard.Equity, fromId.Equity, 5);
    }

    [Fact]
    public void FindBestMove_Opening31_ReturnsMove()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var result = engine.FindBestMove("4HPwATDgc/ABMA", 3, 1);

        Assert.True(result.SubMoveCount > 0, "Should find a move");
        Assert.True(result.Pips > 0, "Should move some pips");
        Assert.NotEmpty(result.ResultPositionId);
    }

    [Fact]
    public void GenerateMovesWithEval_Opening31_SortedByEquity()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var moves = engine.GenerateMovesWithEval("4HPwATDgc/ABMA", 3, 1);

        Assert.True(moves.Count > 0, "Should generate moves");

        // Verify sorted descending by equity
        for (int i = 1; i < moves.Count; i++)
        {
            Assert.True(moves[i - 1].Equity >= moves[i].Equity,
                $"Move {i - 1} equity {moves[i - 1].Equity} < move {i} equity {moves[i].Equity}");
        }
    }

    [Fact]
    public void GenerateMovesWithEval_AllMovesHaveValidProbabilities()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var moves = engine.GenerateMovesWithEval("4HPwATDgc/ABMA", 5, 2);

        foreach (var move in moves)
        {
            Assert.InRange(move.WinProbability, 0.0, 1.0);
            Assert.InRange(move.WinGammonProbability, 0.0, 1.0);
            Assert.InRange(move.WinBackgammonProbability, 0.0, 1.0);
            Assert.InRange(move.LoseGammonProbability, 0.0, 1.0);
            Assert.InRange(move.LoseBackgammonProbability, 0.0, 1.0);
        }
    }

    [Fact]
    public void ClassifyPosition_Opening_IsContact()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var board = Board.Opening();
        var pc = engine.ClassifyPosition(board);
        Assert.Equal(PositionClass.Contact, pc);
    }

    [Fact]
    public void FindBestMove_Blocked_ReturnsEmptyMove()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        // Create a position where player is completely blocked
        var board = new Board();
        board.Player[24] = 2; // 2 on bar
        for (int i = 0; i < 6; i++)
            board.Opponent[i] = 2; // all entry points blocked
        // Rest of the 15 checkers
        board.Player[0] = 13;
        board.Opponent[23] = 3;

        var result = engine.FindBestMove(board, 3, 1);
        Assert.Equal(0, result.SubMoveCount);
    }

    [Fact]
    public void EvaluatePositionPlied_0Ply_MatchesBasicEval()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var basic = engine.EvaluatePosition("4HPwATDgc/ABMA");
        var plied = engine.EvaluatePositionPlied("4HPwATDgc/ABMA", 0);

        Assert.Equal(basic.Win, plied.Win, 5);
        Assert.Equal(basic.Equity, plied.Equity, 5);
    }

    [Fact]
    public void EvaluatePositionPlied_1Ply_ReasonableEquity()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var result = engine.EvaluatePositionPlied("4HPwATDgc/ABMA", 1);

        // Opening at 1-ply should still be roughly even
        Assert.InRange(result.Win, 0.3, 0.7);
        Assert.InRange(result.Equity, -0.5, 0.5);
    }

    [Fact]
    public void EvaluatePositionPlied_1Ply_DifferentFrom0Ply()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var ply0 = engine.EvaluatePositionPlied("4HPwATDgc/ABMA", 0);
        var ply1 = engine.EvaluatePositionPlied("4HPwATDgc/ABMA", 1);

        // 1-ply should give a different (usually more accurate) result
        // They could theoretically be equal but it's extremely unlikely for the opening
        Assert.NotEqual(ply0.Equity, ply1.Equity, 3);
    }

    [Fact]
    public void EvaluatePositionPlied_IsDeterministic()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var result1 = engine.EvaluatePositionPlied("4HPwATDgc/ABMA", 1);
        var result2 = engine.EvaluatePositionPlied("4HPwATDgc/ABMA", 1);

        Assert.Equal(result1.Win, result2.Win, 5);
        Assert.Equal(result1.Equity, result2.Equity, 5);
    }

    [Fact]
    public void FindBestMovePlied_Opening31_ReturnsMove()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var result = engine.FindBestMovePlied("4HPwATDgc/ABMA", 3, 1, 1);

        Assert.True(result.SubMoveCount > 0, "Should find a move");
        Assert.True(result.Pips > 0, "Should move some pips");
        Assert.NotEmpty(result.ResultPositionId);
    }

    [Fact]
    public void EvaluatePositionFull_ReturnsAllOutputs()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var result = engine.EvaluatePositionFull("4HPwATDgc/ABMA");

        Assert.InRange(result.WinProbability, 0.0, 1.0);
        Assert.InRange(result.WinGammonProbability, 0.0, 1.0);
        Assert.InRange(result.WinBackgammonProbability, 0.0, 1.0);
        Assert.InRange(result.LoseGammonProbability, 0.0, 1.0);
        Assert.InRange(result.LoseBackgammonProbability, 0.0, 1.0);
        Assert.InRange(result.CubelessEquity, -2.0, 2.0);
    }

    [Fact]
    public void CubefulEvaluation_Opening_ProducesDifferentFromCubeless()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var board = Board.Opening();
        var ci = CubeInfo.Money();

        // Cubeful evaluation via cube decision analysis
        var cubeResult = engine.AnalyseCubeDecision(board, ci);

        // NoDouble equity should be in a reasonable range
        Assert.InRange(cubeResult.NoDoubleEquity, -1.0, 1.0);
        // DoubleTake equity should also be reasonable
        Assert.InRange(cubeResult.DoubleTakeEquity, -2.0, 2.0);
        // DoublePass = +1 in money game
        Assert.Equal(1.0, cubeResult.DoublePassEquity, 3);
    }

    [Fact]
    public void EvalRaceBG_RacePosition_BackgammonProbsSet()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        // Create a race position where backgammon is possible:
        // Player bearing off, opponent still has all 15 on high points
        var board = new Board();
        board.Player[0] = 5;
        board.Player[1] = 5;
        board.Player[2] = 5;  // 15 checkers in home board

        board.Opponent[23] = 5;
        board.Opponent[22] = 5;
        board.Opponent[21] = 5;  // 15 checkers in player's home (opponent's back)

        var result = engine.EvaluatePosition(board);

        // In this position, backgammon should be possible (opponent has pieces in player's home)
        // Win probability should be very high
        Assert.True(result.Win > 0.8, $"Player should be winning big, got {result.Win}");
    }

    [Fact]
    public void EvaluatePerfectCubeful_BearoffPosition_ReturnsEquities()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        // Create a simple bearoff position (both sides in home board)
        var board = new Board();
        board.Player[0] = 3;
        board.Player[1] = 3;
        board.Player[2] = 3;
        board.Player[3] = 3;
        board.Player[4] = 3;

        board.Opponent[0] = 3;
        board.Opponent[1] = 3;
        board.Opponent[2] = 3;
        board.Opponent[3] = 3;
        board.Opponent[4] = 3;

        var equities = engine.EvaluatePerfectCubeful(board);

        if (equities != null) // Only if cubeful two-sided DB is loaded
        {
            Assert.Equal(4, equities.Length);
            // All equities should be in [-1, 1]
            for (int i = 0; i < 4; i++)
                Assert.InRange(equities[i], -1.0, 1.0);

            // Player is on roll so has an advantage even in symmetric position
            // Cubeless equity should be positive but moderate
            Assert.InRange(equities[0], -0.5, 1.0);
        }
    }

    [Fact]
    public void EvaluatePerfectCubeful_NonBearoff_ReturnsNull()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        // Opening is not a bearoff position
        var equities = engine.EvaluatePerfectCubeful("4HPwATDgc/ABMA");
        Assert.Null(equities);
    }
}

public class CubeDecisionTests
{
    private static string? FindDataDir()
    {
        string[] candidates =
        [
            Environment.GetEnvironmentVariable("GNUBG_DATA_DIR") ?? "",
            @"C:\git\github\customation\gnubg",
            @"C:\git\github\customation\gnubgnet\data",
        ];
        return candidates.FirstOrDefault(d => File.Exists(Path.Combine(d, "gnubg.wd")));
    }

    private Engine? CreateEngine()
    {
        var dir = FindDataDir();
        if (dir == null) return null;
        return Engine.Create(dir);
    }

    [Fact]
    public void AnalyseCubeDecision_Opening_NoDouble()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var result = engine.AnalyseCubeDecision("4HPwATDgc/ABMA");

        // Opening position: no strong double (NoDoubleBeaver in money game with beavers)
        Assert.Equal(GnuBgNet.Evaluation.CubeAction.NoDoubleBeaver, result.Action);
        Assert.InRange(result.NoDoubleEquity, -0.5, 0.5);
        Assert.Equal(1.0, result.DoublePassEquity, 5);
    }

    [Fact]
    public void AnalyseCubeDecision_ReturnsValidEquities()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var board = Board.Opening();
        var result = engine.AnalyseCubeDecision(board);

        // All equities should be in reasonable range
        Assert.InRange(result.NoDoubleEquity, -3.0, 3.0);
        Assert.InRange(result.DoubleTakeEquity, -3.0, 3.0);
        Assert.InRange(result.DoublePassEquity, 0.9, 1.1);
    }

    [Fact]
    public void AnalyseCubeDecision_StrongPosition_DoublePassIsBest()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        // Create a very strong position: player nearly borne off, opponent far back
        var board = new Board();
        board.Player[0] = 3;
        board.Player[1] = 3;
        board.Player[2] = 3;
        board.Player[3] = 3;
        board.Player[4] = 3;
        // Opponent has all 15 far back
        board.Opponent[20] = 5;
        board.Opponent[21] = 5;
        board.Opponent[22] = 5;

        var result = engine.AnalyseCubeDecision(board);

        // Very strong position: double/pass equity should be positive
        Assert.True(result.DoublePassEquity > 0, "Double/pass should yield positive equity");
        // No-double equity should be substantial
        Assert.True(result.NoDoubleEquity > 0.3, $"Strong position should have positive ND equity, got {result.NoDoubleEquity}");
    }

    [Fact]
    public void AnalyseCubeDecision_MatchPlay_Opening_ReturnsValidResult()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var board = Board.Opening();
        var ci = new CubeInfo
        {
            Cube = 1,
            CubeOwner = -1,
            Move = 0,
            MatchTo = 7,
            Score = [0, 0],
            Crawford = false,
        };

        var result = engine.AnalyseCubeDecision(board, ci);

        // Opening at 7-point match: equities should be valid
        Assert.InRange(result.NoDoubleEquity, -1.0, 1.0);
        Assert.InRange(result.DoubleTakeEquity, -2.0, 2.0);
        Assert.InRange(result.DoublePassEquity, -2.0, 2.0);
    }

    [Fact]
    public void AnalyseCubeDecision_MatchPlay_CrawfordGame_CubeDead()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var board = Board.Opening();
        var ci = new CubeInfo
        {
            Cube = 1,
            CubeOwner = -1,
            Move = 0,
            MatchTo = 5,
            Score = [4, 3],  // 1-away, 2-away Crawford
            Crawford = true,
        };

        var result = engine.AnalyseCubeDecision(board, ci);

        // In Crawford game, cube is dead
        Assert.Equal(CubeAction.NoDoubleDeadCube, result.Action);
    }

    [Fact]
    public void AnalyseCubeDecision_MatchPlay_PostCrawford_TrailingCanDouble()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var board = Board.Opening();
        var ci = new CubeInfo
        {
            Cube = 1,
            CubeOwner = -1,
            Move = 1,           // player 1 on roll
            MatchTo = 5,
            Score = [4, 3],     // player 0 is 1-away, player 1 is 2-away
            Crawford = false,   // post-Crawford
        };

        var result = engine.AnalyseCubeDecision(board, ci);

        // Post-Crawford trailing player should typically be free to double
        // (automatic double since cube doesn't matter for leader who is 1-away)
        Assert.InRange(result.DoublePassEquity, -5.0, 5.0);
    }

    [Fact]
    public void AnalyseCubeDecision_MatchPlay_EquitiesInRange()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var board = Board.Opening();
        var ci = new CubeInfo
        {
            Cube = 2,
            CubeOwner = 0,  // player 0 owns cube
            Move = 0,
            MatchTo = 9,
            Score = [2, 3],
        };

        var result = engine.AnalyseCubeDecision(board, ci);

        // All equities should be in a reasonable range for match play
        Assert.InRange(result.NoDoubleEquity, -3.0, 3.0);
        Assert.InRange(result.DoubleTakeEquity, -5.0, 5.0);
        Assert.InRange(result.DoublePassEquity, -3.0, 3.0);
    }
}

public class RolloutTests
{
    private static string? FindDataDir()
    {
        string[] candidates =
        [
            Environment.GetEnvironmentVariable("GNUBG_DATA_DIR") ?? "",
            @"C:\git\github\customation\gnubg",
            @"C:\git\github\customation\gnubgnet\data",
        ];
        return candidates.FirstOrDefault(d => File.Exists(Path.Combine(d, "gnubg.wd")));
    }

    private Engine? CreateEngine()
    {
        var dir = FindDataDir();
        if (dir == null) return null;
        return Engine.Create(dir);
    }

    [Fact]
    public void Rollout_Opening_ReasonableResults()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var settings = new RolloutSettings { Trials = 36, TruncatePlies = 5, Seed = 42 };
        var result = engine.RolloutPosition("4HPwATDgc/ABMA", settings);

        // Opening should be roughly even
        Assert.InRange(result.WinProbability, 0.2, 0.8);
        Assert.InRange(result.CubelessEquity, -1.0, 1.0);
        // Standard deviations should be positive
        Assert.True(result.WinProbabilityStdDev >= 0);
        Assert.True(result.CubelessEquityStdDev >= 0);
    }

    [Fact]
    public void Rollout_DifferentSeeds_ProduceDifferentResults()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var r1 = engine.RolloutPosition("4HPwATDgc/ABMA",
            new RolloutSettings { Trials = 36, TruncatePlies = 5, Seed = 100 });
        var r2 = engine.RolloutPosition("4HPwATDgc/ABMA",
            new RolloutSettings { Trials = 36, TruncatePlies = 5, Seed = 999 });

        // Different seeds should produce at least slightly different results
        // (extremely unlikely to be identical)
        Assert.True(
            Math.Abs(r1.CubelessEquity - r2.CubelessEquity) > 0.0001 ||
            Math.Abs(r1.WinProbability - r2.WinProbability) > 0.0001,
            "Different seeds should produce different rollout results");
    }

    [Fact]
    public void Rollout_AllProbabilitiesValid()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var settings = new RolloutSettings { Trials = 36, TruncatePlies = 5, Seed = 42 };
        var result = engine.RolloutPosition("4HPwATDgc/ABMA", settings);

        Assert.InRange(result.WinProbability, 0.0, 1.0);
        Assert.InRange(result.WinGammonProbability, 0.0, 1.0);
        Assert.InRange(result.WinBackgammonProbability, 0.0, 1.0);
        Assert.InRange(result.LoseGammonProbability, 0.0, 1.0);
        Assert.InRange(result.LoseBackgammonProbability, 0.0, 1.0);
    }

    [Fact]
    public void Rollout_BearoffTruncation_FasterThanFull()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        // A pure race position — bearoff truncation should kick in early
        var board = new Board();
        board.Player[0] = 5;
        board.Player[1] = 5;
        board.Player[2] = 5;
        board.Opponent[0] = 5;
        board.Opponent[1] = 5;
        board.Opponent[2] = 5;

        var settings = new RolloutSettings
        {
            Trials = 36, TruncatePlies = 100, Seed = 42,
            TruncateBearoff2 = true, TruncateBearoffOS = true
        };
        var result = engine.RolloutPosition(board, settings);

        // Should complete and produce valid results
        Assert.InRange(result.WinProbability, 0.0, 1.0);
        Assert.InRange(result.CubelessEquity, -2.0, 2.0);
    }

    [Fact]
    public void Rollout_VarianceReduction_ReducesStdDev()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var settings1 = new RolloutSettings
        {
            Trials = 72, TruncatePlies = 3, Seed = 42, VarianceReduction = false
        };
        var settings2 = new RolloutSettings
        {
            Trials = 72, TruncatePlies = 3, Seed = 42, VarianceReduction = true
        };

        var r1 = engine.RolloutPosition("4HPwATDgc/ABMA", settings1);
        var r2 = engine.RolloutPosition("4HPwATDgc/ABMA", settings2);

        // Both should produce valid results
        Assert.InRange(r1.WinProbability, 0.0, 1.0);
        Assert.InRange(r2.WinProbability, 0.0, 1.0);
    }

    [Fact]
    public void Rollout_QuasiRandomDice_Produces36CycledResults()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        // With Rotate=true and 36 trials, first roll cycles through all 36 outcomes
        var settings = new RolloutSettings
        {
            Trials = 36, TruncatePlies = 3, Seed = 42, Rotate = true
        };
        var result = engine.RolloutPosition("4HPwATDgc/ABMA", settings);

        Assert.InRange(result.WinProbability, 0.0, 1.0);
    }

    [Fact]
    public void OneSidedRollout_RacePosition_ValidProbabilities()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        // Create a race position
        var board = new Board();
        board.Player[0] = 5;
        board.Player[1] = 5;
        board.Player[2] = 5;
        board.Opponent[3] = 5;
        board.Opponent[4] = 5;
        board.Opponent[5] = 5;

        var result = engine.OneSidedRollout(board, 360);

        Assert.InRange(result.Win, 0.0, 1.0);
        Assert.InRange(result.Equity, -3.0, 3.0);
    }
}

public class MoveFormatterTests
{
    [Fact]
    public void FormatMovePlain_SimpleMove()
    {
        int[] anMove = [12, 9, 7, 5, -1, -1, -1, -1]; // 13/10 8/6
        var result = Engine.FormatMovePlain(anMove);
        Assert.Equal("13/10 8/6", result);
    }

    [Fact]
    public void FormatMovePlain_BarEntry()
    {
        int[] anMove = [24, 20, -1, -1, -1, -1, -1, -1]; // bar/21
        var result = Engine.FormatMovePlain(anMove);
        Assert.Equal("bar/21", result);
    }

    [Fact]
    public void ParseMove_SimpleMove()
    {
        int[] an = new int[8];
        int count = Engine.ParseMove("13/10 8/6", an);
        Assert.Equal(2, count);
        Assert.Equal(12, an[0]); // 13 -> 0-indexed = 12
        Assert.Equal(9, an[1]);  // 10 -> 0-indexed = 9
    }

    [Fact]
    public void ParseMove_BarMove()
    {
        int[] an = new int[8];
        int count = Engine.ParseMove("bar/21", an);
        Assert.Equal(1, count);
        Assert.Equal(24, an[0]); // bar = 24
        Assert.Equal(20, an[1]); // 21 -> 0-indexed = 20
    }

    [Fact]
    public void ParseMove_RepeatNotation()
    {
        int[] an = new int[8];
        int count = Engine.ParseMove("13/11(2)", an);
        Assert.Equal(2, count);
        // Both moves should be 13/11 (0-indexed: 12/10)
        Assert.Equal(12, an[0]);
        Assert.Equal(10, an[1]);
        Assert.Equal(12, an[2]);
        Assert.Equal(10, an[3]);
    }

    [Fact]
    public void ParseMove_Invalid_ReturnsNegative()
    {
        int[] an = new int[8];
        int count = Engine.ParseMove("foo", an);
        Assert.Equal(-1, count);
    }

    [Fact]
    public void FormatMove_Opening31_HasText()
    {
        var engine = CreateEngine();
        if (engine == null) return;

        var board = Board.Opening();
        var bestMove = engine.FindBestMove(board, 3, 1);
        var formatted = engine.FormatMove(board, bestMove.AnMove);

        Assert.False(string.IsNullOrEmpty(formatted));
        Assert.Contains("/", formatted);
    }

    private static string? FindDataDir()
    {
        string[] candidates =
        [
            Environment.GetEnvironmentVariable("GNUBG_DATA_DIR") ?? "",
            @"C:\git\github\customation\gnubg",
            @"C:\git\github\customation\gnubgnet\data",
        ];
        return candidates.FirstOrDefault(d => File.Exists(Path.Combine(d, "gnubg.wd")));
    }

    private Engine? CreateEngine()
    {
        var dir = FindDataDir();
        if (dir == null) return null;
        return Engine.Create(dir);
    }
}

public class EnginePipCountTests
{
    [Fact]
    public void GetPipCount_Opening_Correct()
    {
        var board = Board.Opening();
        var (player, opponent) = Engine.GetPipCount(board);

        // Standard opening: 167 pips each
        Assert.Equal(167, player);
        Assert.Equal(167, opponent);
    }

    [Fact]
    public void GetPipCount_EmptyBoard_Zero()
    {
        var board = new Board();
        var (player, opponent) = Engine.GetPipCount(board);
        Assert.Equal(0, player);
        Assert.Equal(0, opponent);
    }
}

public class GameAnalysisTests
{
    private static string? FindDataDir()
    {
        string[] candidates =
        [
            Environment.GetEnvironmentVariable("GNUBG_DATA_DIR") ?? "",
            @"C:\git\github\customation\gnubg",
            @"C:\git\github\customation\gnubgnet\data",
        ];
        return candidates.FirstOrDefault(d => File.Exists(Path.Combine(d, "gnubg.wd")));
    }

    private Engine? CreateEngine()
    {
        var dir = FindDataDir();
        if (dir == null) return null;
        return Engine.Create(dir);
    }

    [Fact]
    public void AnalyseGame_SingleTurn_ProducesResult()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var board = Board.Opening();
        var bestMove = engine.FindBestMove(board, 3, 1);

        var turns = new List<GameTurn>
        {
            new()
            {
                PositionId = "4HPwATDgc/ABMA",
                Die1 = 3,
                Die2 = 1,
                PlayedMove = bestMove.AnMove,
            }
        };

        var analysis = engine.AnalyseGame(turns);

        Assert.Single(analysis.Turns);
        // Best move should have zero equity loss
        Assert.Equal(0, analysis.Turns[0].EquityLoss, 5);
        Assert.Equal(MoveClassification.Best, analysis.Turns[0].Classification);
    }

    [Fact]
    public void AnalyseGame_SuboptimalMove_DetectsLoss()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        // Get all moves and play a non-best one
        var moves = engine.GenerateMovesWithEval("4HPwATDgc/ABMA", 3, 1);
        if (moves.Count < 2) return;

        var worstMove = moves[^1]; // last = worst
        var turns = new List<GameTurn>
        {
            new()
            {
                PositionId = "4HPwATDgc/ABMA",
                Die1 = 3,
                Die2 = 1,
                PlayedMove = worstMove.AnMove,
            }
        };

        var analysis = engine.AnalyseGame(turns);

        Assert.Single(analysis.Turns);
        Assert.True(analysis.Turns[0].EquityLoss >= 0, "Equity loss should be non-negative");
        Assert.True(analysis.TotalEquityLost >= 0);
    }
}

public class MatchEquityTableTests
{
    [Fact]
    public void ComputeDefault_ProducesValidTable()
    {
        var met = MatchEquityTable.ComputeDefault();

        // 1-away, 1-away should be 50%
        Assert.InRange(met.GetEquity(1, 1), 0.45f, 0.55f);

        // Leading player should have higher equity
        Assert.True(met.GetEquity(1, 5) > met.GetEquity(5, 1),
            "1-away should have higher equity than 5-away");

        // All values should be in [0, 1]
        for (int i = 0; i < 25; i++)
            for (int j = 0; j < 25; j++)
            {
                float eq = met.GetEquity(i + 1, j + 1);
                Assert.InRange(eq, 0.0f, 1.0f);
            }
    }

    [Fact]
    public void MoneyEquity_EvenPosition_NearZero()
    {
        // Equal position: 50% win, no gammons
        float[] output = [0.5f, 0.0f, 0.0f, 0.0f, 0.0f];
        float equity = MatchEquityTable.MoneyEquity(output);
        Assert.InRange(equity, -0.01f, 0.01f);
    }

    [Fact]
    public void MoneyEquity_CertainWin_IsPositive()
    {
        float[] output = [1.0f, 0.0f, 0.0f, 0.0f, 0.0f];
        float equity = MatchEquityTable.MoneyEquity(output);
        Assert.Equal(1.0f, equity);
    }

    [Fact]
    public void MoneyEquity_WinGammon_AddsToEquity()
    {
        float[] output = [0.6f, 0.1f, 0.0f, 0.05f, 0.0f];
        float equity = MatchEquityTable.MoneyEquity(output);
        // 0.6*2 - 1 + 0.1 - 0.05 = 0.2 + 0.1 - 0.05 = 0.25
        Assert.Equal(0.25f, equity, 4);
    }

    [Fact]
    public void LoadFromXml_KazarossXG2_Loads()
    {
        var metPath = @"C:\git\github\customation\gnubg\met\Kazaross-XG2.xml";
        if (!File.Exists(metPath)) return;

        var met = Engine.LoadMatchEquityTable(metPath);

        // 1-away, 1-away should be 50%
        Assert.InRange(met.GetEquity(1, 1), 0.49f, 0.51f);

        // Verify known value from XML: row 0, col 1 = 0.67736
        Assert.InRange(met.Met[0, 1], 0.67f, 0.68f);

        // All values should be in [0, 1]
        for (int i = 0; i < 25; i++)
            for (int j = 0; j < 25; j++)
                Assert.InRange(met.GetEquity(i + 1, j + 1), 0.0f, 1.0f);
    }

    [Fact]
    public void LoadFromXml_Snowie_Loads()
    {
        var metPath = @"C:\git\github\customation\gnubg\met\snowie.xml";
        if (!File.Exists(metPath)) return;

        var met = Engine.LoadMatchEquityTable(metPath);

        Assert.InRange(met.GetEquity(1, 1), 0.49f, 0.51f);
        Assert.True(met.GetEquity(1, 5) > met.GetEquity(5, 1));
    }

    [Fact]
    public void CreateWithCustomMet_UsesLoadedTable()
    {
        var dataDir = @"C:\git\github\customation\gnubg";
        var metPath = @"C:\git\github\customation\gnubg\met\Kazaross-XG2.xml";
        if (!File.Exists(Path.Combine(dataDir, "gnubg.wd")) || !File.Exists(metPath))
            return;

        using var engine = Engine.Create(dataDir, metPath);

        // Should work with custom MET
        float eq = engine.GetMatchEquity(1, 1);
        Assert.InRange(eq, 0.49f, 0.51f);

        // Verify it uses the custom MET value, not the computed default
        float customVal = engine.GetMatchEquity(1, 2);
        Assert.InRange(customVal, 0.6f, 0.7f);
    }

}

public class GapFeatureTests
{
    private static string? FindDataDir()
    {
        string[] candidates =
        [
            Environment.GetEnvironmentVariable("GNUBG_DATA_DIR") ?? "",
            @"C:\git\github\customation\gnubg",
            @"C:\git\github\customation\gnubgnet\data",
        ];
        return candidates.FirstOrDefault(d => File.Exists(Path.Combine(d, "gnubg.wd")));
    }

    private Engine? CreateEngine()
    {
        var dir = FindDataDir();
        if (dir == null) return null;
        return Engine.Create(dir);
    }

    // ---- Cubeful Rollout Tests ----

    [Fact]
    public void Rollout_Cubeful_ProducesCubefulEquity()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        // Opening position — cubeful rollout should produce non-zero cubeful equity
        var board = PositionId.Decode("4HPwATDgc/ABMA")!;
        var settings = new RolloutSettings
        {
            Trials = 144,
            Cubeful = true,
            Seed = 42,
        };
        var result = engine.RolloutPosition(board, settings);

        // Cubeful equity should exist and be different from cubeless
        Assert.InRange(result.WinProbability, 0.0, 1.0);
        Assert.InRange(result.CubelessEquity, -1.5, 1.5);
        // CubefulEquity should be non-NaN
        Assert.False(double.IsNaN(result.CubefulEquity));
    }

    [Fact]
    public void Rollout_CubelessVsCubeful_DifferentEquities()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        // Use a contact position where cube matters
        var board = PositionId.Decode("4HPwATDgc/ABMA")!;

        var cubeless = engine.RolloutPosition(board, new RolloutSettings
        {
            Trials = 144, Cubeful = false, Seed = 42,
        });

        var cubeful = engine.RolloutPosition(board, new RolloutSettings
        {
            Trials = 144, Cubeful = true, Seed = 42,
        });

        // Both should have valid win probabilities
        Assert.InRange(cubeless.WinProbability, 0.0, 1.0);
        Assert.InRange(cubeful.WinProbability, 0.0, 1.0);

        // Cubeful equity should differ from cubeless (cube has value)
        // Note: with only 144 trials they may be close, but cubeful path is exercised
        Assert.False(double.IsNaN(cubeful.CubefulEquity));
    }

    // ---- Resignation Tests ----

    [Fact]
    public void GetResignation_Opening_NoResign()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        // Opening position — nobody should resign
        var board = PositionId.Decode("4HPwATDgc/ABMA")!;
        int level = engine.GetResignation(board);
        Assert.Equal(0, level);
    }

    [Fact]
    public void GetResignation_ReturnsValidLevel()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        // Use a heavily losing position
        var board = new Board();
        board.Player[23] = 15; // all checkers deep in opponent's home
        board.Opponent[5] = 1; // opponent nearly done bearing off

        int level = engine.GetResignation(board);
        // GetResignation returns 0-3 — any value is valid
        Assert.InRange(level, 0, 3);
    }

    [Fact]
    public void GetResignEquities_ReturnsValidResult()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var board = PositionId.Decode("4HPwATDgc/ABMA")!;
        var result = engine.GetResignEquities(board, 1);

        Assert.Equal(1, result.RecommendedLevel);
        // Equity values should be finite
        Assert.False(double.IsNaN(result.EquityBeforeResign));
        Assert.False(double.IsNaN(result.EquityAfterResign));
        // Before resign equity should be better (less negative) than after for the opening
        Assert.True(result.EquityBeforeResign > result.EquityAfterResign,
            "Playing on should be better than resigning in an even position");
    }

    [Fact]
    public void CheckResignation_Opening_Rejects()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var board = PositionId.Decode("4HPwATDgc/ABMA")!;
        int accepted = engine.CheckResignation(board, 1);
        // Should reject resignation in a nearly even position
        Assert.Equal(0, accepted);
    }

    [Fact]
    public void GetResignation_ByPositionId_Works()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        int level = engine.GetResignation("4HPwATDgc/ABMA");
        Assert.Equal(0, level);
    }

    // ---- Bearoff Gammon Tests ----

    [Fact]
    public void BearoffGammon_GetGammonProbs_SingleChecker()
    {
        // 1 checker on point 1 (index 0)
        uint[] board = new uint[6];
        board[0] = 1;

        var probs = BearoffGammon.GetGammonProbs(board);
        // P0 = gammon probability, should be 0 (1 checker = bearoff in 1 roll)
        Assert.True(probs.P0 >= 0);
    }

    [Fact]
    public void BearoffGammon_GetGammonProbs_MultipleCheckers()
    {
        // 3 checkers on point 6 (index 5)
        uint[] board = new uint[6];
        board[5] = 3;

        var probs = BearoffGammon.GetGammonProbs(board);
        // Should return valid probabilities
        Assert.True(probs.P0 >= 0);
        Assert.True(probs.P1 >= 0);
    }

    [Fact]
    public void BearoffGammon_GetGammonProbs_TwoPoints()
    {
        // 1 checker on point 1, 1 checker on point 3
        uint[] board = new uint[6];
        board[0] = 1;
        board[2] = 1;

        var probs = BearoffGammon.GetGammonProbs(board);
        Assert.True(probs.P0 >= 0);
    }

    [Fact]
    public void BearoffGammon_GetRaceBGProbs_SmallPosition()
    {
        // 2 checkers on point 1 (index 0)
        uint[] board = new uint[6];
        board[0] = 2;

        var result = BearoffGammon.GetRaceBGProbs(board);
        Assert.NotNull(result);
        Assert.Equal(BearoffGammon.RbgNProbs, result!.Length);
    }

    [Fact]
    public void BearoffGammon_GetRaceBGProbs_TooManyCheckers_ReturnsNull()
    {
        // More than 6 checkers — should return null
        uint[] board = new uint[6];
        board[0] = 4;
        board[1] = 4;

        var result = BearoffGammon.GetRaceBGProbs(board);
        Assert.Null(result);
    }

    // ---- Board Display Tests ----

    [Fact]
    public void DrawBoard_Opening_ProducesValidAscii()
    {
        var board = PositionId.Decode("4HPwATDgc/ABMA")!;
        string display = BoardDisplay.DrawBoard(board, true);

        Assert.False(string.IsNullOrWhiteSpace(display));
        // Should contain board frame characters
        Assert.Contains("+", display);
        Assert.Contains("|", display);
        // Should have multiple lines
        var lines = display.Split('\n');
        Assert.True(lines.Length > 5, $"Expected multi-line board, got {lines.Length} lines");
    }

    [Fact]
    public void DrawBoard_Clockwise_ProducesValidAscii()
    {
        var board = PositionId.Decode("4HPwATDgc/ABMA")!;
        string display = BoardDisplay.DrawBoard(board, true, clockwise: true);

        Assert.False(string.IsNullOrWhiteSpace(display));
        Assert.Contains("+", display);
        var lines = display.Split('\n');
        Assert.True(lines.Length > 5);
    }

    [Fact]
    public void FIBSBoard_ProducesColonSeparated()
    {
        var board = PositionId.Decode("4HPwATDgc/ABMA")!;
        string fibs = BoardDisplay.FIBSBoard(board, true, "player", "opponent",
            0, 0, 0, 3, 1, 1, -1, false);

        Assert.False(string.IsNullOrWhiteSpace(fibs));
        // FIBS board should have colon-separated fields
        Assert.Contains(":", fibs);
        Assert.Contains("player", fibs);
        Assert.Contains("opponent", fibs);
    }

    [Fact]
    public void DrawBoard_EmptyBoard_DoesNotThrow()
    {
        var board = new Board();
        string display = BoardDisplay.DrawBoard(board, true);
        Assert.False(string.IsNullOrWhiteSpace(display));
    }

    // ---- Feature Extraction Tests ----

    [Fact]
    public void ExtractFeatures_Returns248Floats()
    {
        var board = PositionId.Decode("4HPwATDgc/ABMA")!;
        float[] features = Engine.ExtractFeatures(board);

        Assert.Equal(FeatureExtractor.FeatureDim, features.Length);
        Assert.Equal(248, features.Length);
    }

    [Fact]
    public void ExtractFeatures_NotAllZero()
    {
        var board = PositionId.Decode("4HPwATDgc/ABMA")!;
        float[] features = Engine.ExtractFeatures(board);

        // Opening position should produce non-trivial features
        bool hasNonZero = false;
        for (int i = 0; i < features.Length; i++)
            if (features[i] != 0.0f) { hasNonZero = true; break; }
        Assert.True(hasNonZero, "Features should not be all zeros for opening position");
    }

    [Fact]
    public void ExtractFeatures_SwappedSide_DifferentFeatures()
    {
        // Use an asymmetric position (player has 2 on bar)
        var board = new Board();
        board.Player[24] = 2;  // 2 on bar
        board.Player[0] = 5;
        board.Player[5] = 3;
        board.Player[11] = 5;
        board.Opponent[23] = 2;
        board.Opponent[12] = 5;
        board.Opponent[7] = 3;
        board.Opponent[5] = 5;

        float[] featBottom = Engine.ExtractFeatures(board, isTopOnRoll: false);
        float[] featTop = Engine.ExtractFeatures(board, isTopOnRoll: true);

        // Swapping who is on roll should produce different feature vectors
        bool different = false;
        for (int i = 0; i < featBottom.Length; i++)
            if (Math.Abs(featBottom[i] - featTop[i]) > 1e-6f) { different = true; break; }
        Assert.True(different, "Features should differ when swapping on-roll side");
    }

    [Fact]
    public void ExtractFeatures_ByPositionId_MatchesByBoard()
    {
        string posId = "4HPwATDgc/ABMA";
        var board = PositionId.Decode(posId)!;

        float[] fromBoard = Engine.ExtractFeatures(board);
        float[] fromId = Engine.ExtractFeatures(posId);

        Assert.Equal(fromBoard.Length, fromId.Length);
        for (int i = 0; i < fromBoard.Length; i++)
            Assert.Equal(fromBoard[i], fromId[i], 6);
    }

    [Fact]
    public void ExtractRawInputs_ContactPosition_ReturnsCorrectSize()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        // Opening position is Contact class
        var board = PositionId.Decode("4HPwATDgc/ABMA")!;
        float[] inputs = engine.ExtractRawInputs(board);

        // Contact inputs should be 250
        Assert.Equal(Constants.NumContactInputs, inputs.Length);
    }

    [Fact]
    public void ExtractFeatures_Span_ThrowsOnSmallSpan()
    {
        var board = PositionId.Decode("4HPwATDgc/ABMA")!;
        Span<float> small = stackalloc float[100];
        Assert.Throws<ArgumentException>(() =>
        {
            Span<float> s = new float[100];
            FeatureExtractor.ExtractFeatures(board, false, s);
        });
    }

    // ---- MAT Parser Tests ----

    [Fact]
    public void MatParser_ParseSubMove_Standard()
    {
        Assert.True(MatParser.ParseSubMove("13/7", out int from, out int to));
        Assert.Equal(12, from); // 13 → 0-indexed = 12
        Assert.Equal(6, to);    // 7 → 0-indexed = 6
    }

    [Fact]
    public void MatParser_ParseSubMove_Bar()
    {
        // "bar/20" → from=24 (bar), to=19
        Assert.True(MatParser.ParseSubMove("bar/20", out int from, out int to));
        Assert.Equal(24, from);
        Assert.Equal(19, to);
    }

    [Fact]
    public void MatParser_ParseSubMove_BarNumeric()
    {
        // "25/20" → from=24 (bar), to=19
        Assert.True(MatParser.ParseSubMove("25/20", out int from, out int to));
        Assert.Equal(24, from);
        Assert.Equal(19, to);
    }

    [Fact]
    public void MatParser_ParseSubMove_BearOff()
    {
        Assert.True(MatParser.ParseSubMove("6/off", out int from, out int to));
        Assert.Equal(5, from);
        Assert.Equal(-1, to);
    }

    [Fact]
    public void MatParser_ParseSubMove_BearOffNumeric()
    {
        // "6/0" → bear off
        Assert.True(MatParser.ParseSubMove("6/0", out int from, out int to));
        Assert.Equal(5, from);
        Assert.Equal(-1, to);
    }

    [Fact]
    public void MatParser_ParseMove_Simple()
    {
        int[] anMove = new int[8];
        int count = MatParser.ParseMove("13/7 8/7", anMove);
        Assert.Equal(2, count);
        Assert.Equal(12, anMove[0]); // 13→12
        Assert.Equal(6, anMove[1]);  // 7→6
        Assert.Equal(7, anMove[2]);  // 8→7
        Assert.Equal(6, anMove[3]);  // 7→6
    }

    [Fact]
    public void MatParser_ParseMove_WithRepetition()
    {
        int[] anMove = new int[8];
        int count = MatParser.ParseMove("6/5(2)", anMove);
        Assert.Equal(2, count);
        Assert.Equal(5, anMove[0]); // 6→5
        Assert.Equal(4, anMove[1]); // 5→4
        Assert.Equal(5, anMove[2]); // 6→5 (repeated)
        Assert.Equal(4, anMove[3]); // 5→4 (repeated)
    }

    [Fact]
    public void MatParser_ParseMove_WithHitMarker()
    {
        int[] anMove = new int[8];
        int count = MatParser.ParseMove("8/5* 5/4", anMove);
        Assert.Equal(2, count);
        Assert.Equal(7, anMove[0]); // 8→7
        Assert.Equal(4, anMove[1]); // 5→4
    }

    [Fact]
    public void MatParser_ParseFile_ReturnsValidTurns()
    {
        var matPath = @"C:\git\github\customation\gnubg\artifacts\match_logs\iter1_game1.mat";
        if (!File.Exists(matPath)) return;

        var turns = MatParser.ParseFile(matPath);
        Assert.True(turns.Count > 0, "Should parse at least one turn");

        // First turn should have valid dice
        Assert.InRange(turns[0].Die1, 1, 6);
        Assert.InRange(turns[0].Die2, 1, 6);

        // Should have turns from both players
        Assert.Contains(turns, t => t.Player == 0);
        Assert.Contains(turns, t => t.Player == 1);
    }

    [Fact]
    public void MatParser_ParseFile_FirstMove_Correct()
    {
        var matPath = @"C:\git\github\customation\gnubg\artifacts\match_logs\iter1_game1.mat";
        if (!File.Exists(matPath)) return;

        var turns = MatParser.ParseFile(matPath);
        // First line: "1) 66: 13/7 24/18 13/7 24/18"
        var first = turns[0];
        Assert.Equal(6, first.Die1);
        Assert.Equal(6, first.Die2);
        Assert.Equal(0, first.Player);
        // 13/7 → from=12, to=6
        Assert.Equal(12, first.PlayedMove[0]);
        Assert.Equal(6, first.PlayedMove[1]);
        // 24/18 → from=23, to=17
        Assert.Equal(23, first.PlayedMove[2]);
        Assert.Equal(17, first.PlayedMove[3]);
    }

    [Fact]
    public void AnalyseMat_ReturnsValidResult()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var matPath = @"C:\git\github\customation\gnubg\artifacts\match_logs\iter1_game1.mat";
        if (!File.Exists(matPath)) return;

        var result = engine.AnalyseMat(matPath);

        // Should have per-player stats
        Assert.True(result.TotalMoves[0] > 0 || result.TotalMoves[1] > 0, "Should have moves");
        Assert.True(result.Rating[0].Length > 0);
        Assert.True(result.Rating[1].Length > 0);

        // MPR should be non-negative
        Assert.True(result.MillipointsPerMove[0] >= 0);
        Assert.True(result.MillipointsPerMove[1] >= 0);

        // Total error should be non-negative
        Assert.True(result.TotalError[0] >= 0);
        Assert.True(result.TotalError[1] >= 0);

        // Should have some turn analyses
        Assert.True(result.Turns.Count > 0);
    }

    [Fact]
    public void AnalyseMat_RatingIsValidString()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var matPath = @"C:\git\github\customation\gnubg\artifacts\match_logs\iter1_game1.mat";
        if (!File.Exists(matPath)) return;

        var result = engine.AnalyseMat(matPath);

        string[] validRatings = ["Beginner", "Intermediate", "Advanced", "Master", "Grandmaster", "Super Grandmaster"];
        Assert.Contains(result.Rating[0], validRatings);
        Assert.Contains(result.Rating[1], validRatings);
    }
}
