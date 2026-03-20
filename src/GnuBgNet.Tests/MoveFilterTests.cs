// SPDX-License-Identifier: GPL-3.0-or-later

using Xunit;
using GnuBgNet;
using GnuBgNet.Evaluation;

namespace GnuBgNet.Tests;

public class MoveFilterTests
{
    [Fact]
    public void MoveFilterPresets_Normal_CorrectLayout()
    {
        var f = MoveFilterPresets.Normal;

        // Row 0 (1-ply): ply0 filter active
        Assert.Equal(0, f[0, 0].Accept);
        Assert.Equal(8, f[0, 0].Extra);
        Assert.Equal(0.16f, f[0, 0].Threshold, 0.001f);

        // Row 1 (2-ply): ply0 active, ply1 skip
        Assert.Equal(0, f[1, 0].Accept);
        Assert.Equal(-1, f[1, 1].Accept);

        // Row 2 (3-ply): ply0 active, ply1 skip, ply2 deep filter
        Assert.Equal(-1, f[2, 1].Accept);
        Assert.Equal(0, f[2, 2].Accept);
        Assert.Equal(2, f[2, 2].Extra);
        Assert.Equal(0.04f, f[2, 2].Threshold, 0.001f);

        // Row 3 (4-ply): ply0 active, ply1 skip, ply2 deep, ply3 skip
        Assert.Equal(-1, f[3, 3].Accept);
    }

    [Fact]
    public void MoveFilterPresets_AllPresetsExist()
    {
        Assert.Equal(5, MoveFilterPresets.All.Length);
        Assert.NotNull(MoveFilterPresets.Tiny);
        Assert.NotNull(MoveFilterPresets.Narrow);
        Assert.NotNull(MoveFilterPresets.Normal);
        Assert.NotNull(MoveFilterPresets.Large);
        Assert.NotNull(MoveFilterPresets.Huge);
    }

    [Fact]
    public void MoveFilterPresets_Large_MorePermissiveThanNormal()
    {
        Assert.True(MoveFilterPresets.Large[0, 0].Extra > MoveFilterPresets.Normal[0, 0].Extra);
        Assert.True(MoveFilterPresets.Large[0, 0].Threshold > MoveFilterPresets.Normal[0, 0].Threshold);
    }
}

public class FindnSaveBestMovesTests
{
    private static string DataDir => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        @"source\gnubg");

    private static Engine? CreateEngine()
    {
        string wdPath = Path.Combine(DataDir, "gnubg.wd");
        if (!File.Exists(wdPath)) return null;
        return Engine.Create(DataDir);
    }

    [Fact]
    public void GenerateMovesFiltered_Opening_ReturnsFilteredMoves()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var board = Board.Opening();
        var ec = new EvalContext { Plies = 2, Cubeful = false, UsePrune = true };
        var moves = engine.GenerateMovesFiltered(board, 3, 1, ec, MoveFilterPresets.Normal);

        // Should have moves (31 opening has several options)
        Assert.True(moves.Count > 0, "Should have at least one move");
        // Should be filtered down from the full set
        Assert.True(moves.Count <= 10, $"Normal filter should reduce moves, got {moves.Count}");
        // Should be sorted by equity
        for (int i = 1; i < moves.Count; i++)
            Assert.True(moves[i - 1].Equity >= moves[i].Equity,
                $"Moves not sorted: {moves[i - 1].Equity} < {moves[i].Equity}");
    }

    [Fact]
    public void GenerateMovesFiltered_TinyFilter_FewerMovesThanHuge()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var board = Board.Opening();
        var ec = new EvalContext { Plies = 1, Cubeful = false, UsePrune = true };

        var tinyMoves = engine.GenerateMovesFiltered(board, 6, 1, ec, MoveFilterPresets.Tiny);
        var hugeMoves = engine.GenerateMovesFiltered(board, 6, 1, ec, MoveFilterPresets.Huge);

        Assert.True(tinyMoves.Count <= hugeMoves.Count,
            $"Tiny ({tinyMoves.Count}) should have <= moves than Huge ({hugeMoves.Count})");
    }

    [Fact]
    public void GenerateMovesFiltered_ZeroPly_ReturnsAllMoves()
    {
        using var engine = CreateEngine();
        if (engine == null) return;

        var board = Board.Opening();
        var ec = new EvalContext { Plies = 0, Cubeful = false };
        var filteredMoves = engine.GenerateMovesFiltered(board, 3, 1, ec);

        // At 0-ply, no filtering should occur — all moves returned
        var allMoves = engine.GenerateMovesWithEval(board, 3, 1);
        Assert.Equal(allMoves.Count, filteredMoves.Count);
    }
}
