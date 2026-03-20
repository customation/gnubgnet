// SPDX-License-Identifier: GPL-3.0-or-later

using Xunit;
using GnuBgNet;
using GnuBgNet.Bearoff;
using GnuBgNet.Encoding;

namespace GnuBgNet.Tests;

public class BearoffTests
{
    private static string? FindFile(string name)
    {
        string[] candidates =
        [
            Path.Combine(Environment.GetEnvironmentVariable("GNUBG_DATA_DIR") ?? "", name),
            $@"C:\git\github\customation\gnubg\{name}",
            $@"C:\git\github\customation\gnubgnet\data\{name}",
        ];
        return candidates.FirstOrDefault(File.Exists);
    }

    // ---- One-sided database tests ----

    [Fact]
    public void LoadOneSided_ParsesHeaderCorrectly()
    {
        var path = FindFile("gnubg_os0.bd");
        if (path == null) return;

        var db = BearoffDatabase.Load(path);

        Assert.Equal(BearoffType.OneSided, db.Type);
        Assert.Equal(6, db.Points);
        Assert.Equal(15, db.Chequers);
        Assert.True(db.Gammon);
        Assert.True(db.Compressed);
        Assert.False(db.NormalDist);
        // C(21, 6) = 54264
        Assert.Equal(54264u, db.NumPositions);
    }

    [Fact]
    public void LoadTwoSided_ParsesHeaderCorrectly()
    {
        var path = FindFile("gnubg_ts0.bd");
        if (path == null) return;

        var db = BearoffDatabase.Load(path);

        Assert.Equal(BearoffType.TwoSided, db.Type);
        Assert.Equal(6, db.Points);
        Assert.Equal(6, db.Chequers);
        Assert.True(db.Cubeful);
        // C(12, 6) = 924
        Assert.Equal(924u, db.NumPositions);
    }

    [Fact]
    public void OneSided_Distribution_AllBorneOff_IsImmediate()
    {
        var path = FindFile("gnubg_os0.bd");
        if (path == null) return;

        var db = BearoffDatabase.Load(path);

        // Position where all 15 checkers are already off: empty board
        // PositionBearoff with all zeros should give position 0
        uint[] emptyBoard = new uint[6];
        uint posId = PositionId.PositionBearoff(emptyBoard, 6, 15);

        Span<float> probs = stackalloc float[32];
        db.GetDistribution(posId, probs);

        // Already off: all probability at roll 0
        Assert.True(probs[0] > 0.99f, $"P(off at 0) = {probs[0]}, expected ~1.0");
    }

    [Fact]
    public void OneSided_Distribution_SingleChecker_NonZero()
    {
        var path = FindFile("gnubg_os0.bd");
        if (path == null) return;

        var db = BearoffDatabase.Load(path);

        // 1 checker on 1-point: should bear off in 1 roll with certainty
        uint[] board = [1, 0, 0, 0, 0, 0];
        uint posId = PositionId.PositionBearoff(board, 6, 15);

        Span<float> probs = stackalloc float[32];
        db.GetDistribution(posId, probs);

        // Should bear off in 1 roll with very high probability
        Assert.True(probs[1] > 0.9f, $"P(off in 1 roll) = {probs[1]}, expected > 0.9");
    }

    [Fact]
    public void OneSided_Distribution_SumsToOne()
    {
        var path = FindFile("gnubg_os0.bd");
        if (path == null) return;

        var db = BearoffDatabase.Load(path);

        // Test a few positions
        uint[][] boards =
        [
            [3, 2, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 5],
            [2, 3, 3, 2, 3, 2],
        ];

        Span<float> probs = stackalloc float[32];
        foreach (var board in boards)
        {
            uint posId = PositionId.PositionBearoff(board, 6, 15);
            db.GetDistribution(posId, probs);

            float sum = 0f;
            for (int i = 0; i < 32; i++)
                sum += probs[i];

            Assert.InRange(sum, 0.98f, 1.02f);
        }
    }

    // ---- Two-sided evaluation tests ----

    [Fact]
    public void TwoSided_Evaluate_SymmetricPosition_NearHalf()
    {
        var path = FindFile("gnubg_ts0.bd");
        if (path == null) return;

        var db = BearoffDatabase.Load(path);

        // Symmetric position: both sides have same layout
        // 6 chequers on 6 points for TS database
        var board = new Board();
        board.Player[0] = 1;
        board.Player[1] = 1;
        board.Player[2] = 1;
        board.Player[3] = 1;
        board.Player[4] = 1;
        board.Player[5] = 1;
        board.Opponent[0] = 1;
        board.Opponent[1] = 1;
        board.Opponent[2] = 1;
        board.Opponent[3] = 1;
        board.Opponent[4] = 1;
        board.Opponent[5] = 1;

        float[] output = new float[Constants.NumOutputs];
        db.Evaluate(board, output);

        // Symmetric position where player moves first: significant first-mover advantage
        // with checkers spread across all 6 points
        Assert.InRange(output[Constants.OutputWin], 0.50f, 0.80f);
    }

    [Fact]
    public void TwoSided_Evaluate_PlayerAdvantage_WinAboveHalf()
    {
        var path = FindFile("gnubg_ts0.bd");
        if (path == null) return;

        var db = BearoffDatabase.Load(path);

        // Player has 1 checker close, opponent has 6 far away
        var board = new Board();
        board.Player[0] = 6; // all on 1-point
        board.Opponent[5] = 6; // all on 6-point

        float[] output = new float[Constants.NumOutputs];
        db.Evaluate(board, output);

        // Player should win most of the time
        Assert.True(output[Constants.OutputWin] > 0.8f,
            $"Win prob = {output[Constants.OutputWin]}, expected > 0.8");
    }

    [Fact]
    public void TwoSided_Evaluate_OutputsInValidRange()
    {
        var path = FindFile("gnubg_ts0.bd");
        if (path == null) return;

        var db = BearoffDatabase.Load(path);

        var board = new Board();
        board.Player[0] = 2;
        board.Player[2] = 2;
        board.Player[4] = 2;
        board.Opponent[1] = 2;
        board.Opponent[3] = 2;
        board.Opponent[5] = 2;

        float[] output = new float[Constants.NumOutputs];
        db.Evaluate(board, output);

        Assert.InRange(output[Constants.OutputWin], 0.0f, 1.0f);
        Assert.Equal(0.0f, output[Constants.OutputWinGammon]);
        Assert.Equal(0.0f, output[Constants.OutputWinBackgammon]);
        Assert.Equal(0.0f, output[Constants.OutputLoseGammon]);
        Assert.Equal(0.0f, output[Constants.OutputLoseBackgammon]);
    }

    [Fact]
    public void TwoSided_CubefulEquities_FourValues()
    {
        var path = FindFile("gnubg_ts0.bd");
        if (path == null) return;

        var db = BearoffDatabase.Load(path);

        var board = new Board();
        board.Player[0] = 3;
        board.Player[1] = 3;
        board.Opponent[0] = 3;
        board.Opponent[1] = 3;

        float[] equities = new float[4];
        db.GetCubefulEquities(board, equities);

        // All equities should be in [-1, 1]
        for (int i = 0; i < 4; i++)
        {
            Assert.InRange(equities[i], -1.0f, 1.0f);
        }
    }

    [Fact]
    public void TwoSided_Evaluate_AllOff_WinsAlways()
    {
        var path = FindFile("gnubg_ts0.bd");
        if (path == null) return;

        var db = BearoffDatabase.Load(path);

        // Player has no checkers left (all off), opponent still has some
        var board = new Board();
        // Player: empty board (all 6 borne off)
        board.Opponent[5] = 6; // opponent still on 6-point

        float[] output = new float[Constants.NumOutputs];
        db.Evaluate(board, output);

        // Player already bore off everything: win prob ≈ 1.0
        Assert.True(output[Constants.OutputWin] > 0.99f,
            $"Win prob = {output[Constants.OutputWin]}, expected ≈ 1.0");
    }
}
