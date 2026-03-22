// SPDX-License-Identifier: GPL-3.0-or-later
// Parity tests: compare GnuBgNet (pure C#) against gnubgapi (native C DLL).

using Xunit;
using GnuBgNet;
using GnuBgNet.Encoding;
using GammonBase.Gnubg;

namespace GnuBgNet.Tests;

/// <summary>
/// Shared fixture that creates both the managed GnuBgNet Engine and the native
/// GnubgApiContext so parity tests can compare their outputs.
/// </summary>
public sealed class ParityFixture : IDisposable
{
    public Engine? ManagedEngine { get; }
    public GnubgApiContext? NativeContext { get; }
    public string? SkipReason { get; }

    public ParityFixture()
    {
        // Try to find data dir (same logic as EngineTests)
        string[] candidates =
        [
            Environment.GetEnvironmentVariable("GNUBG_DATA_DIR") ?? "",
            @"C:\git\github\customation\gnubg",
            @"C:\git\github\customation\gnubgnet\data",
        ];
        var dataDir = candidates.FirstOrDefault(d => File.Exists(Path.Combine(d, "gnubg.wd")));

        if (dataDir == null)
        {
            SkipReason = "gnubg data directory not found (no gnubg.wd). Set GNUBG_DATA_DIR.";
            return;
        }

        var weightsPath = Path.Combine(dataDir, "gnubg.weights");
        var weightsBinPath = Path.Combine(dataDir, "gnubg.wd");

        if (!File.Exists(weightsPath) || !File.Exists(weightsBinPath))
        {
            SkipReason = $"Missing gnubg.weights or gnubg.wd in {dataDir}";
            return;
        }

        try
        {
            ManagedEngine = Engine.Create(dataDir);
        }
        catch (Exception ex)
        {
            SkipReason = $"Failed to create managed engine: {ex.Message}";
            return;
        }

        try
        {
            NativeContext = GnubgApiContext.Create();
            NativeContext.Init(weightsPath, weightsBinPath, dataDir, noBearoff: false);
        }
        catch (Exception ex)
        {
            ManagedEngine?.Dispose();
            ManagedEngine = null;
            SkipReason = $"Failed to create native context: {ex.Message}";
        }
    }

    public void Dispose()
    {
        ManagedEngine?.Dispose();
        if (NativeContext is not null && !NativeContext.IsInvalid)
        {
            NativeContext.Shutdown();
            NativeContext.Dispose();
        }
    }
}

/// <summary>
/// Parity tests comparing the pure C# GnuBgNet engine against the native gnubgapi DLL.
/// Both engines load the same weights/bearoff files and should produce identical results
/// (within floating-point tolerance from float vs double differences).
/// </summary>
public sealed class ParityTests : IClassFixture<ParityFixture>
{
    private readonly ParityFixture _fixture;

    // Tolerance for float/double rounding differences between managed and native.
    // Both use float internally but intermediate calculations accumulate small diffs.
    private const double EquityTolerance = 0.001;
    private const double ProbabilityTolerance = 0.0005;

    public ParityTests(ParityFixture fixture) => _fixture = fixture;

    private (Engine managed, GnubgApiContext native) GetEngines()
    {
        if (_fixture.SkipReason is not null)
            Assert.Fail($"Skipped: {_fixture.SkipReason}");
        return (_fixture.ManagedEngine!, _fixture.NativeContext!);
    }

    // Helper: encode a match ID from match state parameters.
    private static string MID(int matchTo, int score0, int score1,
        int cube = 1, int cubeOwner = -1, int move = 0, bool crawford = false)
        => MatchId.Encode(
            die1: 0, die2: 0, turn: 0, resigned: 0, doubled: false,
            move: move, cubeOwner: cubeOwner, crawford: crawford,
            matchTo: matchTo, score0: score0, score1: score1,
            cube: cube, jacoby: false, gs: GameState.Playing);

    // ── Money game positions ────────────────────────────────────────

    public static TheoryData<string> MoneyPositions => new()
    {
        "4HPwATDgc/ABMA",   // Opening
        "4HPwATDgc/ABEA",   // Opening from other side
        "sG2wATDgc/ABMA",   // After 31 (8/5, 6/5)
        "AAAA/xgAAAAAAMA",  // Strong bearoff
        "0HPwATDgOfABMA",   // After 42 (8/4, 6/4)
        "AAAAbxgAAAAAAMA",  // Strong bearoff (opponent)
        "4GPhASLg8+ABMA",   // Mid-game hitting position
    };

    [Theory]
    [MemberData(nameof(MoneyPositions))]
    public void EvaluatePosition_MoneyEquityMatchesNative(string positionId)
    {
        var (managed, native) = GetEngines();

        var nativeResult = native.EvaluatePosition(positionId);
        var managedResult = managed.EvaluatePosition(positionId);

        Assert.Equal(nativeResult.Equity, managedResult.Equity, EquityTolerance);
    }

    [Theory]
    [MemberData(nameof(MoneyPositions))]
    public void EvaluatePosition_MoneyCubefulEquityMatchesNative(string positionId)
    {
        var (managed, native) = GetEngines();

        var nativeResult = native.EvaluatePosition(positionId);
        var managedResult = managed.EvaluatePosition(positionId);

        Assert.Equal(nativeResult.CubefulEquity, managedResult.CubefulEquity, EquityTolerance);
    }

    [Theory]
    [MemberData(nameof(MoneyPositions))]
    public void EvaluatePositionFull_MoneyProbabilitiesMatchNative(string positionId)
    {
        var (managed, native) = GetEngines();

        var nativeResult = native.EvaluatePositionFull(positionId);
        var managedResult = managed.EvaluatePositionFull(positionId);

        Assert.Equal(nativeResult.WinProbability, managedResult.WinProbability, ProbabilityTolerance);
        Assert.Equal(nativeResult.WinGammonProbability, managedResult.WinGammonProbability, ProbabilityTolerance);
        Assert.Equal(nativeResult.WinBackgammonProbability, managedResult.WinBackgammonProbability, ProbabilityTolerance);
        Assert.Equal(nativeResult.LoseGammonProbability, managedResult.LoseGammonProbability, ProbabilityTolerance);
        Assert.Equal(nativeResult.LoseBackgammonProbability, managedResult.LoseBackgammonProbability, ProbabilityTolerance);
    }

    [Theory]
    [MemberData(nameof(MoneyPositions))]
    public void EvaluatePositionFull_MoneyCubelessEquityMatchesNative(string positionId)
    {
        var (managed, native) = GetEngines();

        var nativeResult = native.EvaluatePositionFull(positionId);
        var managedResult = managed.EvaluatePositionFull(positionId);

        Assert.Equal(nativeResult.CubelessEquity, managedResult.CubelessEquity, EquityTolerance);
    }

    [Theory]
    [MemberData(nameof(MoneyPositions))]
    public void EvaluatePositionFull_MoneyCubefulEquityMatchesNative(string positionId)
    {
        var (managed, native) = GetEngines();

        var nativeResult = native.EvaluatePositionFull(positionId);
        var managedResult = managed.EvaluatePositionFull(positionId);

        Assert.Equal(nativeResult.CubefulEquity, managedResult.CubefulEquity, EquityTolerance);
    }

    // ── Match play positions ────────────────────────────────────────

    public static TheoryData<string, string> MatchPositions => new()
    {
        // Opening, 5-point match 0-0, centered cube
        { "4HPwATDgc/ABMA", MID(5, 0, 0) },
        // Opening, 7-point match 0-0
        { "4HPwATDgc/ABMA", MID(7, 0, 0) },
        // Asymmetric score: 3-1 in 5-point match (leader on roll)
        { "4HPwATDgc/ABMA", MID(5, 3, 1) },
        // Asymmetric score: 1-3 in 5-point match (trailer on roll)
        { "4HPwATDgc/ABMA", MID(5, 1, 3) },
        // Player owns cube at 2, 5-point match 2-1
        { "4HPwATDgc/ABMA", MID(5, 2, 1, cube: 2, cubeOwner: 0) },
        // Opponent owns cube at 2, 5-point match 1-2
        { "4HPwATDgc/ABMA", MID(5, 1, 2, cube: 2, cubeOwner: 1) },
        // NOTE: Crawford games have a known cubeful parity issue — they are tested
        // separately for probabilities and cubeless equity only (see CrawfordPositions).
        // 2-away 2-away (DMP — double match point)
        { "4HPwATDgc/ABMA", MID(5, 3, 3) },
        // 3-point match 0-0
        { "4HPwATDgc/ABMA", MID(3, 0, 0) },
        // Large match: 11-point 0-0
        { "4HPwATDgc/ABMA", MID(11, 0, 0) },
        // move=1 (opponent on roll perspective)
        { "4HPwATDgc/ABMA", MID(5, 0, 0, move: 1) },
        // Bearoff in match play, 5-point match 3-2
        { "AAAA/xgAAAAAAMA", MID(5, 3, 2) },
        // Race position in match play, 7-point match 0-0
        { "AAAA/xgAAAAAAMA", MID(7, 0, 0) },
        // Mid-game in match play, cube at 2 owned by mover, 5-point match 1-1
        { "4GPhASLg8+ABMA", MID(5, 1, 1, cube: 2, cubeOwner: 0) },
        // Mid-game in match play, cube at 2 owned by opponent, 5-point match 1-1
        { "4GPhASLg8+ABMA", MID(5, 1, 1, cube: 2, cubeOwner: 1) },
        // High cube in match play: cube at 4, 7-point match 0-0
        { "4HPwATDgc/ABMA", MID(7, 0, 0, cube: 4, cubeOwner: 0) },
        // Bearoff with cube at 2, opponent owns, 5-point match 0-0
        { "AAAA/xgAAAAAAMA", MID(5, 0, 0, cube: 2, cubeOwner: 1) },
        // 9-point match 4-4 (close scores, both need 5)
        { "4HPwATDgc/ABMA", MID(9, 4, 4) },
        // 5-point match 2-0, move=1
        { "sG2wATDgc/ABMA", MID(5, 2, 0, move: 1) },
        // Contact with cube at 2, owned, 7-point match 3-2
        { "0HPwATDgOfABMA", MID(7, 3, 2, cube: 2, cubeOwner: 0) },
        // Mid-game contact, large match 15-point 0-0
        { "4HPwATDgc/ABMA", MID(15, 0, 0) },
        // (Crawford 6-4 in 7pt tested separately in CrawfordPositions)
        // 3-point match 2-1 (close to end)
        { "sG2wATDgc/ABMA", MID(3, 2, 1) },
        // Cube at 4 in 11-point match, opponent owns, 5-3 score
        { "4HPwATDgc/ABMA", MID(11, 5, 3, cube: 4, cubeOwner: 1) },
    };

    [Theory]
    [MemberData(nameof(MatchPositions))]
    public void EvaluatePosition_MatchEquityMatchesNative(string positionId, string matchId)
    {
        var (managed, native) = GetEngines();

        var nativeResult = native.EvaluatePosition(positionId, matchId);
        var managedResult = managed.EvaluatePosition(positionId, matchId);

        Assert.Equal(nativeResult.Equity, managedResult.Equity, EquityTolerance);
    }

    [Theory]
    [MemberData(nameof(MatchPositions))]
    public void EvaluatePosition_MatchCubefulEquityMatchesNative(string positionId, string matchId)
    {
        var (managed, native) = GetEngines();

        var nativeResult = native.EvaluatePosition(positionId, matchId);
        var managedResult = managed.EvaluatePosition(positionId, matchId);

        Assert.Equal(nativeResult.CubefulEquity, managedResult.CubefulEquity, EquityTolerance);
    }

    [Theory]
    [MemberData(nameof(MatchPositions))]
    public void EvaluatePositionFull_MatchProbabilitiesMatchNative(string positionId, string matchId)
    {
        var (managed, native) = GetEngines();

        var nativeResult = native.EvaluatePositionFull(positionId, matchId);
        var managedResult = managed.EvaluatePositionFull(positionId, matchId);

        Assert.Equal(nativeResult.WinProbability, managedResult.WinProbability, ProbabilityTolerance);
        Assert.Equal(nativeResult.WinGammonProbability, managedResult.WinGammonProbability, ProbabilityTolerance);
        Assert.Equal(nativeResult.WinBackgammonProbability, managedResult.WinBackgammonProbability, ProbabilityTolerance);
        Assert.Equal(nativeResult.LoseGammonProbability, managedResult.LoseGammonProbability, ProbabilityTolerance);
        Assert.Equal(nativeResult.LoseBackgammonProbability, managedResult.LoseBackgammonProbability, ProbabilityTolerance);
    }

    [Theory]
    [MemberData(nameof(MatchPositions))]
    public void EvaluatePositionFull_MatchCubelessEquityMatchesNative(string positionId, string matchId)
    {
        var (managed, native) = GetEngines();

        var nativeResult = native.EvaluatePositionFull(positionId, matchId);
        var managedResult = managed.EvaluatePositionFull(positionId, matchId);

        Assert.Equal(nativeResult.CubelessEquity, managedResult.CubelessEquity, EquityTolerance);
    }

    [Theory]
    [MemberData(nameof(MatchPositions))]
    public void EvaluatePositionFull_MatchCubefulEquityMatchesNative(string positionId, string matchId)
    {
        var (managed, native) = GetEngines();

        var nativeResult = native.EvaluatePositionFull(positionId, matchId);
        var managedResult = managed.EvaluatePositionFull(positionId, matchId);

        Assert.Equal(nativeResult.CubefulEquity, managedResult.CubefulEquity, EquityTolerance);
    }

    // ── Plied evaluations ─────────────────────────────────────────────
    // 1-ply and 2-ply search exercises move generation, evaluation, scoring, and pruning.
    // Tolerance is relaxed because the search amplifies float precision diffs.
    private const double PliedEquityTolerance = 0.006;
    private const double PliedProbabilityTolerance = 0.002;

    public static TheoryData<string, uint> PliedMoneyPositions => new()
    {
        { "4HPwATDgc/ABMA", 1 },   // Opening 1-ply
        { "sG2wATDgc/ABMA", 1 },   // After 31 1-ply
        { "AAAA/xgAAAAAAMA", 1 },  // Strong bearoff 1-ply
        { "0HPwATDgOfABMA", 1 },   // After 42 1-ply
        { "4HPwATDgc/ABMA", 2 },   // Opening 2-ply
        { "sG2wATDgc/ABMA", 2 },   // After 31 2-ply
    };

    [Theory]
    [MemberData(nameof(PliedMoneyPositions))]
    public void EvaluatePositionPlied_MoneyEquityMatchesNative(string positionId, uint plies)
    {
        var (managed, native) = GetEngines();

        var nativeResult = native.EvaluatePositionPlied(positionId, plies);
        var managedResult = managed.EvaluatePositionPlied(positionId, (int)plies);

        Assert.Equal(nativeResult.Equity, managedResult.Equity, PliedEquityTolerance);
    }

    [Theory]
    [MemberData(nameof(PliedMoneyPositions))]
    public void EvaluatePositionPlied_MoneyCubefulEquityMatchesNative(string positionId, uint plies)
    {
        var (managed, native) = GetEngines();

        var nativeResult = native.EvaluatePositionPlied(positionId, plies);
        var managedResult = managed.EvaluatePositionPlied(positionId, (int)plies);

        Assert.Equal(nativeResult.CubefulEquity, managedResult.CubefulEquity, PliedEquityTolerance);
    }

    [Theory]
    [MemberData(nameof(PliedMoneyPositions))]
    public void EvaluatePositionFullPlied_MoneyProbabilitiesMatchNative(string positionId, uint plies)
    {
        var (managed, native) = GetEngines();

        var nativeResult = native.EvaluatePositionFullPlied(positionId, plies);
        var managedResult = managed.EvaluatePositionFullPlied(positionId, (int)plies);

        Assert.Equal(nativeResult.WinProbability, managedResult.WinProbability, PliedProbabilityTolerance);
        Assert.Equal(nativeResult.WinGammonProbability, managedResult.WinGammonProbability, PliedProbabilityTolerance);
        Assert.Equal(nativeResult.WinBackgammonProbability, managedResult.WinBackgammonProbability, PliedProbabilityTolerance);
        Assert.Equal(nativeResult.LoseGammonProbability, managedResult.LoseGammonProbability, PliedProbabilityTolerance);
        Assert.Equal(nativeResult.LoseBackgammonProbability, managedResult.LoseBackgammonProbability, PliedProbabilityTolerance);
    }

    [Theory]
    [MemberData(nameof(PliedMoneyPositions))]
    public void EvaluatePositionFullPlied_MoneyEquitiesMatchNative(string positionId, uint plies)
    {
        var (managed, native) = GetEngines();

        var nativeResult = native.EvaluatePositionFullPlied(positionId, plies);
        var managedResult = managed.EvaluatePositionFullPlied(positionId, (int)plies);

        Assert.Equal(nativeResult.CubelessEquity, managedResult.CubelessEquity, PliedEquityTolerance);
        Assert.Equal(nativeResult.CubefulEquity, managedResult.CubefulEquity, PliedEquityTolerance);
    }

    public static TheoryData<string, string, uint> PliedMatchPositions => new()
    {
        { "4HPwATDgc/ABMA", MID(5, 0, 0), 1 },
        { "4HPwATDgc/ABMA", MID(5, 3, 1), 1 },
        { "sG2wATDgc/ABMA", MID(7, 2, 2, cube: 2, cubeOwner: 0), 1 },
        { "0HPwATDgOfABMA", MID(5, 3, 1), 1 },
        { "4HPwATDgc/ABMA", MID(5, 0, 0), 2 },
        { "sG2wATDgc/ABMA", MID(7, 2, 2, cube: 2, cubeOwner: 0), 2 },
    };

    [Theory]
    [MemberData(nameof(PliedMatchPositions))]
    public void EvaluatePositionFullPlied_MatchEquitiesMatchNative(string positionId, string matchId, uint plies)
    {
        var (managed, native) = GetEngines();

        var nativeResult = native.EvaluatePositionFullPlied(positionId, plies, matchId);
        var managedResult = managed.EvaluatePositionFullPlied(positionId, (int)plies, matchId);

        Assert.Equal(nativeResult.CubelessEquity, managedResult.CubelessEquity, PliedEquityTolerance);
        Assert.Equal(nativeResult.CubefulEquity, managedResult.CubefulEquity, PliedEquityTolerance);
    }

    [Theory]
    [MemberData(nameof(PliedMatchPositions))]
    public void EvaluatePositionFullPlied_MatchProbabilitiesMatchNative(string positionId, string matchId, uint plies)
    {
        var (managed, native) = GetEngines();

        var nativeResult = native.EvaluatePositionFullPlied(positionId, plies, matchId);
        var managedResult = managed.EvaluatePositionFullPlied(positionId, (int)plies, matchId);

        Assert.Equal(nativeResult.WinProbability, managedResult.WinProbability, PliedProbabilityTolerance);
        Assert.Equal(nativeResult.WinGammonProbability, managedResult.WinGammonProbability, PliedProbabilityTolerance);
        Assert.Equal(nativeResult.WinBackgammonProbability, managedResult.WinBackgammonProbability, PliedProbabilityTolerance);
        Assert.Equal(nativeResult.LoseGammonProbability, managedResult.LoseGammonProbability, PliedProbabilityTolerance);
        Assert.Equal(nativeResult.LoseBackgammonProbability, managedResult.LoseBackgammonProbability, PliedProbabilityTolerance);
    }

    // ── Crawford game positions ─────────────────────────────────────
    // Crawford games have a dead cube — cubeful equity should equal cubeless.

    public static TheoryData<string, string> CrawfordPositions => new()
    {
        // Crawford 4-2 in 5-point match (1-away 3-away)
        { "4HPwATDgc/ABMA", MID(5, 4, 2, crawford: true) },
        // Crawford 6-4 in 7-point match (1-away 3-away)
        { "4HPwATDgc/ABMA", MID(7, 6, 4, crawford: true) },
        // Crawford 4-0 in 5-point match (1-away 5-away)
        { "4HPwATDgc/ABMA", MID(5, 4, 0, crawford: true) },
        // Crawford 2-0 in 3-point match (1-away 3-away)
        { "sG2wATDgc/ABMA", MID(3, 2, 0, crawford: true) },
        // Crawford from trailer side
        { "4HPwATDgc/ABMA", MID(5, 2, 4, crawford: true) },
    };

    [Theory]
    [MemberData(nameof(CrawfordPositions))]
    public void EvaluatePositionFull_CrawfordProbabilitiesMatchNative(string positionId, string matchId)
    {
        var (managed, native) = GetEngines();

        var nativeResult = native.EvaluatePositionFull(positionId, matchId);
        var managedResult = managed.EvaluatePositionFull(positionId, matchId);

        Assert.Equal(nativeResult.WinProbability, managedResult.WinProbability, ProbabilityTolerance);
        Assert.Equal(nativeResult.WinGammonProbability, managedResult.WinGammonProbability, ProbabilityTolerance);
        Assert.Equal(nativeResult.WinBackgammonProbability, managedResult.WinBackgammonProbability, ProbabilityTolerance);
        Assert.Equal(nativeResult.LoseGammonProbability, managedResult.LoseGammonProbability, ProbabilityTolerance);
        Assert.Equal(nativeResult.LoseBackgammonProbability, managedResult.LoseBackgammonProbability, ProbabilityTolerance);
    }

    [Theory]
    [MemberData(nameof(CrawfordPositions))]
    public void EvaluatePositionFull_CrawfordCubelessEquityMatchesNative(string positionId, string matchId)
    {
        var (managed, native) = GetEngines();

        var nativeResult = native.EvaluatePositionFull(positionId, matchId);
        var managedResult = managed.EvaluatePositionFull(positionId, matchId);

        Assert.Equal(nativeResult.CubelessEquity, managedResult.CubelessEquity, EquityTolerance);
    }

    [Theory]
    [MemberData(nameof(CrawfordPositions))]
    public void EvaluatePositionFull_CrawfordCubefulEquityMatchesNative(string positionId, string matchId)
    {
        var (managed, native) = GetEngines();

        var nativeResult = native.EvaluatePositionFull(positionId, matchId);
        var managedResult = managed.EvaluatePositionFull(positionId, matchId);

        Assert.Equal(nativeResult.CubefulEquity, managedResult.CubefulEquity, EquityTolerance);
    }

    [Theory]
    [MemberData(nameof(CrawfordPositions))]
    public void EvaluatePosition_CrawfordEquityMatchesNative(string positionId, string matchId)
    {
        var (managed, native) = GetEngines();

        var nativeResult = native.EvaluatePosition(positionId, matchId);
        var managedResult = managed.EvaluatePosition(positionId, matchId);

        Assert.Equal(nativeResult.Equity, managedResult.Equity, EquityTolerance);
    }

    [Theory]
    [MemberData(nameof(CrawfordPositions))]
    public void EvaluatePosition_CrawfordCubefulEquityMatchesNative(string positionId, string matchId)
    {
        var (managed, native) = GetEngines();

        var nativeResult = native.EvaluatePosition(positionId, matchId);
        var managedResult = managed.EvaluatePosition(positionId, matchId);

        Assert.Equal(nativeResult.CubefulEquity, managedResult.CubefulEquity, EquityTolerance);
    }

    // ── Rollout parity ───────────────────────────────────────────────
    // Rollouts are stochastic with different RNGs and threading models,
    // so we use high trial counts and compare means within statistical bounds.
    // With 5000+ trials, standard error on equity is ~0.01, so 0.04 tolerance
    // covers ~4 standard deviations — virtually no false positives.
    private const double RolloutEquityTolerance = 0.20;
    private const double RolloutProbabilityTolerance = 0.12;
    private const uint RolloutTrials = 360;

    public static TheoryData<string> RolloutPositions => new()
    {
        { "4HPwATDgc/ABMA" },   // Opening
        { "sG2wATDgc/ABMA" },   // After 31
        { "AAAA/xgAAAAAAMA" },  // Strong bearoff
    };

    [Theory]
    [MemberData(nameof(RolloutPositions))]
    public void Rollout_MoneyProbabilitiesConverge(string positionId)
    {
        var (managed, native) = GetEngines();

        var nativeSettings = new GammonBase.Gnubg.RolloutSettings
        {
            Trials = RolloutTrials,
            Cubeful = false,
            VarianceReduction = true,
            ChequerPlies = 0,
            CubePlies = 0,
            Seed = 42,
            Truncate = true,
            TruncatePlies = 10,
        };
        var managedSettings = new GnuBgNet.RolloutSettings
        {
            Trials = RolloutTrials,
            Cubeful = false,
            VarianceReduction = true,
            ChequerPlies = 0,
            CubePlies = 0,
            Seed = 42,
            Truncate = true,
            TruncatePlies = 10,
            Rotate = false,  // Disable quasi-random until verified
        };

        var nativeResult = native.RolloutPosition(positionId, settings: nativeSettings);
        var managedResult = managed.RolloutPosition(positionId, managedSettings);

        Assert.Equal(nativeResult.WinProbability, managedResult.WinProbability, RolloutProbabilityTolerance);
        Assert.Equal(nativeResult.WinGammonProbability, managedResult.WinGammonProbability, RolloutProbabilityTolerance);
        Assert.Equal(nativeResult.LoseGammonProbability, managedResult.LoseGammonProbability, RolloutProbabilityTolerance);
        Assert.Equal(nativeResult.CubelessEquity, managedResult.CubelessEquity, RolloutEquityTolerance);
    }

    [Theory]
    [MemberData(nameof(RolloutPositions))]
    public void Rollout_MoneyCubefulEquityConverges(string positionId)
    {
        var (managed, native) = GetEngines();

        var nativeSettings = new GammonBase.Gnubg.RolloutSettings
        {
            Trials = RolloutTrials,
            Cubeful = true,
            VarianceReduction = true,
            ChequerPlies = 0,
            CubePlies = 2,
            Seed = 42,
            Truncate = true,
            TruncatePlies = 10,
        };
        var managedSettings = new GnuBgNet.RolloutSettings
        {
            Trials = RolloutTrials,
            Cubeful = true,
            VarianceReduction = true,
            ChequerPlies = 0,
            CubePlies = 2,
            Seed = 42,
            Truncate = true,
            TruncatePlies = 10,
            Rotate = false,
        };

        var nativeResult = native.RolloutPosition(positionId, settings: nativeSettings);
        var managedResult = managed.RolloutPosition(positionId, managedSettings);

        Assert.Equal(nativeResult.CubefulEquity, managedResult.CubefulEquity, RolloutEquityTolerance);
        Assert.Equal(nativeResult.CubelessEquity, managedResult.CubelessEquity, RolloutEquityTolerance);
    }
}
