// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2019 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later

using GnuBgNet.Bearoff;
using GnuBgNet.Encoding;
using GnuBgNet.Evaluation;
using GnuBgNet.Formatting;
using GnuBgNet.MatchEquity;
using GnuBgNet.MoveGeneration;
using GnuBgNet.NeuralNet;
using GnuBgNet.Rollout;

namespace GnuBgNet;

/// <summary>
/// Top-level evaluation engine. Loads data files and provides evaluation/move-finding API.
/// This is the primary public API for consuming the ported gnubg engine.
/// </summary>
public sealed class Engine : IDisposable
{
    private readonly NetworkSet _nets;
    private readonly IBearoffDatabase? _oneSidedBearoff;
    private readonly IBearoffDatabase? _twoSidedBearoff;
    private readonly IBearoffDatabase? _hyper1;
    private readonly IBearoffDatabase? _hyper2;
    private readonly IBearoffDatabase? _hyper3;
    private readonly IPositionEvaluator _evaluator;
    internal IPositionEvaluator Evaluator => _evaluator;
    private readonly IMatchEquityTable _met;

    private Engine(NetworkSet nets, IBearoffDatabase? osBearoff, IBearoffDatabase? tsBearoff,
        IBearoffDatabase? hyper1, IBearoffDatabase? hyper2, IBearoffDatabase? hyper3,
        IMatchEquityTable met,
        IEvalCache? mainCache = null, IEvalCache? pruneCache = null,
        IMoveGenerator? moveGenerator = null, IInputCalculator? inputCalculator = null)
    {
        _nets = nets;
        _oneSidedBearoff = osBearoff;
        _twoSidedBearoff = tsBearoff;
        _hyper1 = hyper1;
        _hyper2 = hyper2;
        _hyper3 = hyper3;
        _evaluator = new Evaluator(nets, osBearoff, tsBearoff, hyper1, hyper2, hyper3,
            met, mainCache, pruneCache, moveGenerator, inputCalculator);
        _met = met;
    }

    /// <summary>
    /// Create an engine instance by loading data files from the specified directory.
    /// Expects: gnubg.wd (required), gnubg_os0.bd (optional), gnubg_ts0.bd (optional),
    /// hyper1.bd/hyper2.bd/hyper3.bd (optional hypergammon databases).
    /// </summary>
    public static Engine Create(string dataDir)
    {
        var weightsPath = Path.Combine(dataDir, "gnubg.wd");
        if (!File.Exists(weightsPath))
            throw new FileNotFoundException($"Weights file not found: {weightsPath}");

        var nets = NetworkSet.LoadBinary(weightsPath);
        var (os, ts, h1, h2, h3) = LoadBearoffDatabases(dataDir);

        // Try loading Kazaross-XG2 MET from data directory; fall back to computed default.
        var metXmlPath = Path.Combine(dataDir, "met", "Kazaross-XG2.xml");
        var met = File.Exists(metXmlPath)
            ? MetXmlLoader.LoadFromFile(metXmlPath)
            : MatchEquityTable.ComputeDefault();

        return new Engine(nets, os, ts, h1, h2, h3, met);
    }

    /// <summary>
    /// Create an engine instance with custom neural networks and components.
    /// </summary>
    public static Engine Create(string dataDir, NetworkSet nets,
        IMoveGenerator? moveGenerator = null, IInputCalculator? inputCalculator = null,
        IEvalCache? mainCache = null, IEvalCache? pruneCache = null)
    {
        var (os, ts, h1, h2, h3) = LoadBearoffDatabases(dataDir);

        var metXmlPath = Path.Combine(dataDir, "met", "Kazaross-XG2.xml");
        var met = File.Exists(metXmlPath)
            ? MetXmlLoader.LoadFromFile(metXmlPath)
            : MatchEquityTable.ComputeDefault();

        return new Engine(nets, os, ts, h1, h2, h3, met, mainCache, pruneCache,
            moveGenerator, inputCalculator);
    }

    /// <summary>
    /// Create an engine instance with a custom match equity table loaded from an XML file.
    /// </summary>
    public static Engine Create(string dataDir, string metXmlPath)
    {
        var weightsPath = Path.Combine(dataDir, "gnubg.wd");
        if (!File.Exists(weightsPath))
            throw new FileNotFoundException($"Weights file not found: {weightsPath}");

        var nets = NetworkSet.LoadBinary(weightsPath);
        var (os, ts, h1, h2, h3) = LoadBearoffDatabases(dataDir);
        var met = MetXmlLoader.LoadFromFile(metXmlPath);

        return new Engine(nets, os, ts, h1, h2, h3, met);
    }

    private static (BearoffDatabase? os, BearoffDatabase? ts,
        BearoffDatabase? h1, BearoffDatabase? h2, BearoffDatabase? h3)
        LoadBearoffDatabases(string dataDir)
    {
        BearoffDatabase? os = null, ts = null, h1 = null, h2 = null, h3 = null;

        var osPath = Path.Combine(dataDir, "gnubg_os0.bd");
        if (File.Exists(osPath)) os = BearoffDatabase.Load(osPath);

        var tsPath = Path.Combine(dataDir, "gnubg_ts0.bd");
        if (File.Exists(tsPath)) ts = BearoffDatabase.Load(tsPath);

        var h1Path = Path.Combine(dataDir, "hyper1.bd");
        if (File.Exists(h1Path)) h1 = BearoffDatabase.Load(h1Path);

        var h2Path = Path.Combine(dataDir, "hyper2.bd");
        if (File.Exists(h2Path)) h2 = BearoffDatabase.Load(h2Path);

        var h3Path = Path.Combine(dataDir, "hyper3.bd");
        if (File.Exists(h3Path)) h3 = BearoffDatabase.Load(h3Path);

        return (os, ts, h1, h2, h3);
    }

    /// <summary>
    /// Load a match equity table from an XML file (gnubg MET format).
    /// </summary>
    public static MatchEquityTable LoadMatchEquityTable(string xmlPath)
        => MetXmlLoader.LoadFromFile(xmlPath);

    /// <summary>
    /// Load a match equity table from XML string content.
    /// </summary>
    public static MatchEquityTable LoadMatchEquityTableFromXml(string xml)
        => MetXmlLoader.LoadFromXml(xml);

    /// <summary>
    /// Evaluate a position at 0-ply. Returns 5 output probabilities.
    /// </summary>
    public EvaluationResult EvaluatePosition(string positionId, string? matchId = null)
    {
        var board = PositionId.Decode(positionId)
            ?? throw new ArgumentException($"Invalid position ID: {positionId}");

        var ci = BuildCubeInfo(matchId);
        return BuildEvaluationResult(board, ci, 0);
    }

    /// <summary>
    /// Evaluate a position from a Board directly.
    /// </summary>
    public EvaluationResult EvaluatePosition(Board board, CubeInfo? cubeInfo = null)
    {
        cubeInfo ??= CubeInfo.Money();
        return BuildEvaluationResult(board, cubeInfo, 0);
    }

    /// <summary>
    /// Evaluate a position at the specified number of plies.
    /// Higher plies = stronger but slower. 0=instant, 1=fast, 2=world-class.
    /// </summary>
    public EvaluationResult EvaluatePositionPlied(string positionId, int plies, string? matchId = null)
    {
        var board = PositionId.Decode(positionId)
            ?? throw new ArgumentException($"Invalid position ID: {positionId}");

        var ci = BuildCubeInfo(matchId);
        return BuildEvaluationResult(board, ci, plies);
    }

    /// <summary>
    /// Evaluate a position at the specified number of plies from a Board.
    /// </summary>
    public EvaluationResult EvaluatePositionPlied(Board board, int plies, CubeInfo? cubeInfo = null)
    {
        cubeInfo ??= CubeInfo.Money();
        return BuildEvaluationResult(board, cubeInfo, plies);
    }

    /// <summary>
    /// Build an EvaluationResult with cubeful equity.
    /// Port of gnubgapi_evaluate_position/plied from gnubgapi.c:
    /// first eval cubeless to get equity, then a second eval with fCubeful=TRUE
    /// for cubeful equity via GeneralEvaluationEPlied.
    /// </summary>
    private EvaluationResult BuildEvaluationResult(Board board, CubeInfo ci, int plies)
    {
        // First evaluation: cubeless (matches C: GeneralEvaluationE with ecBasic)
        Span<float> output = stackalloc float[Constants.NumOutputs];
        if (plies > 0)
            _evaluator.EvaluatePositionPlied(board, output, plies, true, null, ci);
        else
            _evaluator.EvaluatePosition(board, output);

        float equity = ci.MatchTo > 0
            ? CubeDecision.UtilityMatch(output, ci, _met)
            : MatchEquityTable.MoneyEquity(output);

        // Second evaluation: cubeful (matches C: ec.fCubeful = 1; GeneralEvaluationE)
        var ec = new EvalContext
        {
            Cubeful = true,
            Plies = plies,
            UsePrune = true,
            Deterministic = true,
        };
        Span<float> arCubeful = stackalloc float[Constants.NumRolloutOutputs];
        _evaluator.GeneralEvaluationEPlied(board, arCubeful, ci, ec, plies);

        return new EvaluationResult(
            Win: output[Constants.OutputWin],
            WinGammon: output[Constants.OutputWinGammon],
            WinBackgammon: output[Constants.OutputWinBackgammon],
            LoseGammon: output[Constants.OutputLoseGammon],
            LoseBackgammon: output[Constants.OutputLoseBackgammon],
            Equity: equity,
            CubefulEquity: arCubeful[Constants.OutputCubefulEquity]);
    }

    /// <summary>
    /// Evaluate a position returning full 7-output result (5 probs + cubeless + cubeful equity).
    /// Port of gnubgapi_evaluate_position_full from gnubgapi.c.
    /// </summary>
    public FullEvaluationResult EvaluatePositionFull(string positionId, string? matchId = null)
    {
        var board = PositionId.Decode(positionId)
            ?? throw new ArgumentException($"Invalid position ID: {positionId}");

        var ci = BuildCubeInfo(matchId);
        return EvaluatePositionFull(board, ci);
    }

    /// <summary>
    /// Evaluate a position returning full 7-output result from a Board.
    /// </summary>
    public FullEvaluationResult EvaluatePositionFull(Board board, CubeInfo? cubeInfo = null)
    {
        cubeInfo ??= CubeInfo.Money();
        return BuildFullResult(board, cubeInfo, 0);
    }

    /// <summary>
    /// Evaluate a position at the specified number of plies returning full 7-output result.
    /// Port of gnubgapi_evaluate_position_full_plied from gnubgapi.c.
    /// </summary>
    public FullEvaluationResult EvaluatePositionFullPlied(string positionId, int plies, string? matchId = null)
    {
        var board = PositionId.Decode(positionId)
            ?? throw new ArgumentException($"Invalid position ID: {positionId}");

        var ci = BuildCubeInfo(matchId);
        return EvaluatePositionFullPlied(board, plies, ci);
    }

    /// <summary>
    /// Evaluate a position at the specified number of plies returning full 7-output result from a Board.
    /// </summary>
    public FullEvaluationResult EvaluatePositionFullPlied(Board board, int plies, CubeInfo? cubeInfo = null)
    {
        cubeInfo ??= CubeInfo.Money();
        return BuildFullResult(board, cubeInfo, plies);
    }

    /// <summary>
    /// Build a FullEvaluationResult using GeneralEvaluationEPlied.
    /// Port of gnubgapi_evaluate_position_full/full_plied from gnubgapi.c.
    /// </summary>
    private FullEvaluationResult BuildFullResult(Board board, CubeInfo ci, int plies)
    {
        var ec = new EvalContext
        {
            Cubeful = true,
            Plies = plies,
            UsePrune = true,
            Deterministic = true,
        };

        float[] arOutput = new float[Constants.NumRolloutOutputs];
        _evaluator.GeneralEvaluationEPlied(board, arOutput, ci, ec, plies);

        return new FullEvaluationResult(
            WinProbability: arOutput[Constants.OutputWin],
            WinGammonProbability: arOutput[Constants.OutputWinGammon],
            WinBackgammonProbability: arOutput[Constants.OutputWinBackgammon],
            LoseGammonProbability: arOutput[Constants.OutputLoseGammon],
            LoseBackgammonProbability: arOutput[Constants.OutputLoseBackgammon],
            CubelessEquity: arOutput[Constants.OutputEquity],
            CubefulEquity: arOutput[Constants.OutputCubefulEquity]);
    }

    /// <summary>
    /// Build a CubeInfo from a match ID string.
    /// Port of parse_position_and_cubeinfo() cube info logic from gnubgapi.c.
    /// </summary>
    private static CubeInfo BuildCubeInfo(string? matchId)
    {
        if (string.IsNullOrEmpty(matchId))
            return CubeInfo.Money();

        var mi = MatchId.Decode(matchId);
        if (mi == null)
            throw new ArgumentException($"Invalid match ID: {matchId}");

        if (mi.MatchTo > 0)
        {
            return new CubeInfo
            {
                Cube = mi.Cube,
                CubeOwner = mi.CubeOwner,
                Move = mi.Move,
                MatchTo = mi.MatchTo,
                Score0 = mi.Score0,
                Score1 = mi.Score1,
                Crawford = mi.Crawford,
                Jacoby = mi.Jacoby,
                Beavers = false,
            };
        }

        return new CubeInfo
        {
            Cube = mi.Cube,
            CubeOwner = mi.CubeOwner,
            Move = mi.Move,
            MatchTo = 0,
            Score0 = 0,
            Score1 = 0,
            Crawford = false,
            Jacoby = mi.Jacoby,
            Beavers = false,
        };
    }

    /// <summary>
    /// Generate all legal moves and score each at 0-ply.
    /// Returns moves sorted by equity (best first).
    /// </summary>
    public IReadOnlyList<ScoredMove> GenerateMovesWithEval(string positionId, int die1, int die2)
    {
        var board = PositionId.Decode(positionId)
            ?? throw new ArgumentException($"Invalid position ID: {positionId}");

        return GenerateMovesWithEval(board, die1, die2);
    }

    /// <summary>
    /// Generate all legal moves and score each at 0-ply from a Board.
    /// </summary>
    public IReadOnlyList<ScoredMove> GenerateMovesWithEval(Board board, int die1, int die2)
    {
        return GenerateMovesWithEvalPlied(board, die1, die2, 0);
    }

    /// <summary>
    /// Generate all legal moves and score each at the specified ply depth.
    /// Returns moves sorted by equity (best first).
    /// </summary>
    public IReadOnlyList<ScoredMove> GenerateMovesWithEvalPlied(Board board, int die1, int die2, int plies)
    {
        var ml = MoveGenerator.GenerateMoves(board, die1, die2);

        var scoredMoves = new List<ScoredMove>(ml.Moves.Count);
        float[] output = new float[Constants.NumOutputs];

        foreach (var move in ml.Moves)
        {
            // Apply move and swap sides for evaluation
            var newBoard = MoveGenerator.ApplyMove(board, move);
            var swapped = newBoard.Swapped();

            if (plies > 0)
                _evaluator.EvaluatePositionPlied(swapped, output, plies - 1);
            else
                _evaluator.EvaluatePosition(swapped, output);

            // Invert: we evaluated from opponent's perspective
            Evaluation.Evaluator.InvertEvaluation(output);

            float equity = MatchEquityTable.MoneyEquity(output);
            string resultPosId = PositionId.Encode(newBoard);

            scoredMoves.Add(new ScoredMove(
                AnMove: (int[])move.AnMove.Clone(),
                ResultPositionId: resultPosId,
                SubMoveCount: (int)move.SubMoveCount,
                Pips: (int)move.Pips,
                Equity: equity,
                WinProbability: output[Constants.OutputWin],
                WinGammonProbability: output[Constants.OutputWinGammon],
                WinBackgammonProbability: output[Constants.OutputWinBackgammon],
                LoseGammonProbability: output[Constants.OutputLoseGammon],
                LoseBackgammonProbability: output[Constants.OutputLoseBackgammon]));
        }

        scoredMoves.Sort((a, b) => b.Equity.CompareTo(a.Equity));
        return scoredMoves;
    }

    /// <summary>
    /// Generate and score moves using progressive move filtering (FindnSaveBestMoves).
    /// Uses the specified EvalContext and optional move filter preset.
    /// Returns scored moves sorted by equity (best first).
    /// </summary>
    public IReadOnlyList<ScoredMove> GenerateMovesFiltered(Board board, int die1, int die2,
        EvalContext? ec = null, MoveFilter[,]? moveFilters = null)
    {
        ec ??= EvalContext.WorldClass();
        var ml = new MoveList();
        _evaluator.FindnSaveBestMoves(ml, board, die1, die2, ec.Value, moveFilters);

        var result = new List<ScoredMove>(ml.Moves.Count);
        foreach (var move in ml.Moves)
        {
            string resultPosId = PositionId.Encode(PositionId.FromKey(move.Key));
            result.Add(new ScoredMove(
                AnMove: (int[])move.AnMove.Clone(),
                ResultPositionId: resultPosId,
                SubMoveCount: (int)move.SubMoveCount,
                Pips: (int)move.Pips,
                Equity: move.EvalOutputs[Constants.OutputEquity],
                WinProbability: move.EvalOutputs[Constants.OutputWin],
                WinGammonProbability: move.EvalOutputs[Constants.OutputWinGammon],
                WinBackgammonProbability: move.EvalOutputs[Constants.OutputWinBackgammon],
                LoseGammonProbability: move.EvalOutputs[Constants.OutputLoseGammon],
                LoseBackgammonProbability: move.EvalOutputs[Constants.OutputLoseBackgammon]));
        }
        return result;
    }

    /// <summary>
    /// Find the best move for a given position and dice roll.
    /// </summary>
    public MoveResult FindBestMove(string positionId, int die1, int die2)
    {
        var board = PositionId.Decode(positionId)
            ?? throw new ArgumentException($"Invalid position ID: {positionId}");

        return FindBestMove(board, die1, die2);
    }

    /// <summary>
    /// Find the best move from a Board.
    /// </summary>
    public MoveResult FindBestMove(Board board, int die1, int die2)
    {
        var moves = GenerateMovesWithEval(board, die1, die2);

        if (moves.Count == 0)
            return new MoveResult([-1, -1, -1, -1, -1, -1, -1, -1], "", 0, 0);

        var best = moves[0]; // already sorted by equity
        return new MoveResult(best.AnMove, best.ResultPositionId, best.SubMoveCount, best.Pips);
    }

    /// <summary>
    /// Find the best move at the specified ply depth.
    /// </summary>
    public MoveResult FindBestMovePlied(string positionId, int die1, int die2, int plies)
    {
        var board = PositionId.Decode(positionId)
            ?? throw new ArgumentException($"Invalid position ID: {positionId}");

        return FindBestMovePlied(board, die1, die2, plies);
    }

    /// <summary>
    /// Find the best move at the specified ply depth from a Board.
    /// </summary>
    public MoveResult FindBestMovePlied(Board board, int die1, int die2, int plies)
    {
        var moves = GenerateMovesWithEvalPlied(board, die1, die2, plies);

        if (moves.Count == 0)
            return new MoveResult([-1, -1, -1, -1, -1, -1, -1, -1], "", 0, 0);

        var best = moves[0];
        return new MoveResult(best.AnMove, best.ResultPositionId, best.SubMoveCount, best.Pips);
    }

    /// <summary>
    /// Classify a position.
    /// </summary>
    public PositionClass ClassifyPosition(Board board)
    {
        return _evaluator.ClassifyPosition(board);
    }

    /// <summary>
    /// Analyse the cube decision for a money game position.
    /// Returns recommended action plus equities for no-double, double/take, and double/pass.
    /// </summary>
    public CubeDecisionResult AnalyseCubeDecision(string positionId, bool jacoby = false)
    {
        var board = PositionId.Decode(positionId)
            ?? throw new ArgumentException($"Invalid position ID: {positionId}");

        return AnalyseCubeDecision(board, jacoby: jacoby);
    }

    /// <summary>
    /// Analyse the cube decision for a money game position from a Board.
    /// </summary>
    public CubeDecisionResult AnalyseCubeDecision(Board board, int cubeOwner = -1, bool jacoby = false)
    {
        float[] output = new float[Constants.NumOutputs];
        _evaluator.EvaluatePosition(board, output);

        return CubeDecision.AnalyseMoney(output, cubeOwner, jacoby);
    }

    /// <summary>
    /// Analyse the cube decision with full cube/match state.
    /// Supports both money game (matchTo=0) and match play.
    /// </summary>
    public CubeDecisionResult AnalyseCubeDecision(Board board, CubeInfo cubeInfo)
    {
        float[] output = new float[Constants.NumOutputs];
        _evaluator.EvaluatePosition(board, output);

        var pc = _evaluator.ClassifyPosition(board);
        float cubeEff = CubeEfficiency.Compute(board, pc, 0);
        return CubeDecision.Analyse(output, cubeInfo, _met, cubeEff);
    }

    /// <summary>
    /// Analyse the cube decision using plied evaluation.
    /// </summary>
    public CubeDecisionResult AnalyseCubeDecisionPlied(Board board, int plies, int cubeOwner = -1, bool jacoby = false)
    {
        float[] output = new float[Constants.NumOutputs];
        _evaluator.EvaluatePositionPlied(board, output, plies);

        var pc = _evaluator.ClassifyPosition(board);
        float cubeEff = CubeEfficiency.Compute(board, pc, plies);
        var ci = new CubeInfo
        {
            CubeOwner = cubeOwner,
            Move = 0,
            MatchTo = 0,
            Jacoby = jacoby,
            Beavers = true,
        };
        return CubeDecision.AnalyseMoney(output, ci, cubeEff);
    }

    /// <summary>
    /// Analyse the cube decision with full cube/match state, using plied evaluation.
    /// </summary>
    public CubeDecisionResult AnalyseCubeDecisionPlied(Board board, int plies, CubeInfo cubeInfo)
    {
        float[] output = new float[Constants.NumOutputs];
        _evaluator.EvaluatePositionPlied(board, output, plies);

        var pc = _evaluator.ClassifyPosition(board);
        float cubeEff = CubeEfficiency.Compute(board, pc, plies);
        return CubeDecision.Analyse(output, cubeInfo, _met, cubeEff);
    }

    /// <summary>
    /// Perform a Monte Carlo rollout of the given position.
    /// </summary>
    public RolloutResult RolloutPosition(string positionId, RolloutSettings? settings = null)
    {
        var board = PositionId.Decode(positionId)
            ?? throw new ArgumentException($"Invalid position ID: {positionId}");

        return RolloutPosition(board, settings);
    }

    /// <summary>
    /// Perform a Monte Carlo rollout of the given position from a Board.
    /// </summary>
    public RolloutResult RolloutPosition(Board board, RolloutSettings? settings = null)
    {
        settings ??= RolloutSettings.Default;
        var engine = new RolloutEngine(_evaluator, _met);
        return engine.Rollout(board, settings);
    }

    /// <summary>
    /// Roll out multiple candidate moves and compare them.
    /// Returns results for each legal move, sorted by equity.
    /// Supports JSD-based early stopping for inferior moves.
    /// </summary>
    public RolloutResult[] RolloutMoves(Board board, int die1, int die2,
        RolloutSettings? settings = null)
    {
        settings ??= RolloutSettings.Default;
        var engine = new RolloutEngine(_evaluator, _met);
        return engine.RolloutMoves(board, die1, die2, settings, out _);
    }

    /// <summary>
    /// One-sided rollout: compute race probabilities using Monte Carlo + bearoff.
    /// Returns 5 output probabilities plus average rolls for each side.
    /// </summary>
    public EvaluationResult OneSidedRollout(Board board, uint nGames = 5760)
    {
        var osr = new OneSidedRollout(_oneSidedBearoff);
        float[] output = new float[Constants.NumOutputs];
        osr.RaceProbs(board, nGames, output, out _, out _);
        float equity = MatchEquityTable.MoneyEquity(output);
        return new EvaluationResult(output[0], output[1], output[2], output[3], output[4], equity, equity);
    }

    /// <summary>
    /// Get match equity for a given score.
    /// </summary>
    public float GetMatchEquity(int playerAway, int opponentAway)
    {
        return _met.GetEquity(playerAway, opponentAway);
    }

    /// <summary>
    /// Evaluate a bearoff position with perfect cubeful equities.
    /// Returns 4 values: [cubeless, cube-owned, cube-centered, cube-opponent].
    /// Returns null if the position is not a bearoff position or cubeful DB is unavailable.
    /// </summary>
    public float[]? EvaluatePerfectCubeful(Board board)
    {
        float[] equities = new float[4];
        if (_evaluator.EvaluatePerfectCubeful(board, equities))
            return equities;
        return null;
    }

    /// <summary>
    /// Evaluate a bearoff position with perfect cubeful equities from a position ID.
    /// </summary>
    public float[]? EvaluatePerfectCubeful(string positionId)
    {
        var board = PositionId.Decode(positionId)
            ?? throw new ArgumentException($"Invalid position ID: {positionId}");
        return EvaluatePerfectCubeful(board);
    }

    /// <summary>
    /// Format a move in human-readable notation with hit markers.
    /// </summary>
    public string FormatMove(Board board, int[] anMove)
        => MoveFormatter.FormatMove(board, anMove);

    /// <summary>
    /// Format a move in plain notation (no hit markers).
    /// </summary>
    public static string FormatMovePlain(int[] anMove)
        => MoveFormatter.FormatMovePlain(anMove);

    /// <summary>
    /// Parse a human-readable move string into internal int[8] format.
    /// Returns -1 on error, otherwise the number of sub-moves.
    /// </summary>
    public static int ParseMove(string input, int[] an)
        => MoveFormatter.ParseMove(input, an);

    /// <summary>
    /// Count total pips for each side.
    /// Returns (playerPips, opponentPips).
    /// </summary>
    public static (int Player, int Opponent) GetPipCount(Board board)
        => PipCount.Count(board);

    /// <summary>
    /// Analyse a game: evaluate each turn, find the best move, and compute equity loss.
    /// Port of AnalyzeGame() from analysis.c.
    /// </summary>
    public GameAnalysis AnalyseGame(IReadOnlyList<GameTurn> turns, int plies = 0)
    {
        var analyses = new List<TurnAnalysis>(turns.Count);
        double totalLoss = 0;
        int errors = 0, blunders = 0;

        foreach (var turn in turns)
        {
            if (turn.IsCubeAction) continue; // Skip cube actions for now

            var board = PositionId.Decode(turn.PositionId);
            if (board == null) continue;

            // Generate and evaluate all moves
            var rankedMoves = plies > 0
                ? GenerateMovesWithEvalPlied(board.Value, turn.Die1, turn.Die2, plies)
                : GenerateMovesWithEval(board.Value, turn.Die1, turn.Die2);

            if (rankedMoves.Count == 0) continue;

            double bestEquity = rankedMoves[0].Equity;

            // Find the played move's rank and equity
            int playedRank = 0;
            double playedEquity = bestEquity;

            for (int i = 0; i < rankedMoves.Count; i++)
            {
                if (MovesMatch(rankedMoves[i].AnMove, turn.PlayedMove))
                {
                    playedRank = i;
                    playedEquity = rankedMoves[i].Equity;
                    break;
                }
            }

            double equityLoss = bestEquity - playedEquity;
            var classification = ClassifyMove(equityLoss);

            if (classification >= MoveClassification.Bad) errors++;
            if (classification >= MoveClassification.Blunder) blunders++;
            totalLoss += equityLoss;

            analyses.Add(new TurnAnalysis
            {
                PositionId = turn.PositionId,
                EquityBefore = bestEquity,
                EquityAfterBestMove = bestEquity,
                EquityAfterPlayedMove = playedEquity,
                EquityLoss = equityLoss,
                RankedMoves = rankedMoves,
                PlayedMoveRank = playedRank,
                Classification = classification,
            });
        }

        return new GameAnalysis
        {
            Turns = analyses,
            TotalEquityLost = totalLoss,
            AverageEquityLoss = analyses.Count > 0 ? totalLoss / analyses.Count : 0,
            ErrorCount = errors,
            BlunderCount = blunders,
        };
    }

    private static bool MovesMatch(int[] a, int[] b)
    {
        for (int i = 0; i < 8; i++)
        {
            int va = i < a.Length ? a[i] : -1;
            int vb = i < b.Length ? b[i] : -1;
            if (va != vb) return false;
        }
        return true;
    }

    private static MoveClassification ClassifyMove(double equityLoss)
    {
        if (equityLoss < 0.001) return MoveClassification.Best;
        if (equityLoss < 0.02) return MoveClassification.Good;
        if (equityLoss < 0.04) return MoveClassification.Inaccuracy;
        if (equityLoss < 0.08) return MoveClassification.Doubtful;
        if (equityLoss < 0.16) return MoveClassification.Bad;
        return MoveClassification.Blunder;
    }

    /// <summary>
    /// Analyse a Jellyfish .mat file.
    /// Port of gnubgapi_analyse_mat() from gnubgapi.c lines 1244-1277.
    /// Parses the file, then analyses all games found within.
    /// </summary>
    public MatAnalysisResult AnalyseMat(string matPath, int plies = 0)
    {
        var turns = MatParser.ParseFile(matPath);
        if (turns.Count == 0)
            throw new ArgumentException("No turns found in .mat file");

        return AnalyseGameFromBoard(turns, plies);
    }

    /// <summary>
    /// Analyse a game from parsed turns, tracking board state from the opening position.
    /// Port of gnubgapi_analyse_game() from gnubgapi.c lines 885-1017.
    /// Unlike AnalyseGame (which takes position IDs), this tracks the board internally
    /// and uses the C-style skill classification system.
    /// </summary>
    public MatAnalysisResult AnalyseGameFromBoard(IReadOnlyList<GameTurn> turns, int plies = 0)
    {
        var board = Board.Opening();
        var ci = CubeInfo.Money();

        var analyses = new List<TurnAnalysis>();
        int[] totalMoves = new int[2];
        int[] unforcedMoves = new int[2];
        int[,] skillCounts = new int[2, 4]; // [player, skill]
        float[] totalError = new float[2];

        foreach (var turn in turns)
        {
            if (turn.IsCubeAction) continue;
            if (turn.Die1 < 1 || turn.Die1 > 6 || turn.Die2 < 1 || turn.Die2 > 6)
                continue;

            int player = turn.Player;

            // Generate all legal moves
            var ml = MoveGenerator.GenerateMoves(board, turn.Die1, turn.Die2);

            if (ml.Moves.Count == 0)
            {
                // No legal moves — forced pass, swap sides
                board = board.Swapped();
                continue;
            }

            totalMoves[player]++;

            if (ml.Moves.Count == 1)
            {
                // Forced move — no decision to analyse
                skillCounts[player, (int)SkillLevel.None]++;
                var newBoard = MoveGenerator.ApplyMoveRaw(board, turn.PlayedMove);
                board = newBoard.Swapped();
                continue;
            }

            // Multiple legal moves — decision point
            unforcedMoves[player]++;

            // Evaluate all candidates
            var rankedMoves = plies > 0
                ? GenerateMovesWithEvalPlied(board, turn.Die1, turn.Die2, plies)
                : GenerateMovesWithEval(board, turn.Die1, turn.Die2);

            if (rankedMoves.Count == 0)
            {
                // Fallback: apply move and continue
                var nb = MoveGenerator.ApplyMoveRaw(board, turn.PlayedMove);
                board = nb.Swapped();
                continue;
            }

            // Find played move by comparing result position IDs
            var afterBoard = MoveGenerator.ApplyMoveRaw(board, turn.PlayedMove);
            string? afterPosId = PositionId.Encode(afterBoard);

            int playedIdx = -1;
            if (afterPosId != null)
            {
                for (int i = 0; i < rankedMoves.Count; i++)
                {
                    if (rankedMoves[i].ResultPositionId == afterPosId)
                    {
                        playedIdx = i;
                        break;
                    }
                }
            }

            // Compute skill (played score - best score)
            float rSkill = 0.0f;
            if (playedIdx >= 0 && rankedMoves.Count > 0)
                rSkill = (float)(rankedMoves[playedIdx].Equity - rankedMoves[0].Equity);

            var skill = ClassifySkill(rSkill);
            skillCounts[player, (int)skill]++;

            if (rSkill < 0.0f)
                totalError[player] -= rSkill;

            string posId = PositionId.Encode(board) ?? "";

            var classification = ClassifyMove(-rSkill); // ClassifyMove expects positive loss

            analyses.Add(new TurnAnalysis
            {
                PositionId = posId,
                EquityBefore = rankedMoves[0].Equity,
                EquityAfterBestMove = rankedMoves[0].Equity,
                EquityAfterPlayedMove = playedIdx >= 0 ? rankedMoves[playedIdx].Equity : rankedMoves[0].Equity,
                EquityLoss = rSkill < 0 ? -rSkill : 0,
                RankedMoves = rankedMoves,
                PlayedMoveRank = playedIdx >= 0 ? playedIdx : 0,
                Classification = classification,
            });

            // Apply the played move and swap sides
            board = afterBoard.Swapped();
        }

        // Compute derived statistics
        float[] errorPerMove = new float[2];
        float[] mpr = new float[2];
        string[] rating = new string[2];

        for (int p = 0; p < 2; p++)
        {
            if (unforcedMoves[p] > 0)
                errorPerMove[p] = totalError[p] / unforcedMoves[p];
            mpr[p] = errorPerMove[p] * 1000.0f;
            rating[p] = GetRating(errorPerMove[p]);
        }

        return new MatAnalysisResult
        {
            Turns = analyses,
            TotalMoves = [totalMoves[0], totalMoves[1]],
            UnforcedMoves = [unforcedMoves[0], unforcedMoves[1]],
            SkillCounts = new int[,]
            {
                { skillCounts[0, 0], skillCounts[0, 1], skillCounts[0, 2], skillCounts[0, 3] },
                { skillCounts[1, 0], skillCounts[1, 1], skillCounts[1, 2], skillCounts[1, 3] },
            },
            TotalError = [totalError[0], totalError[1]],
            ErrorPerMove = [errorPerMove[0], errorPerMove[1]],
            MillipointsPerMove = [mpr[0], mpr[1]],
            Rating = [rating[0], rating[1]],
        };
    }

    /// <summary>
    /// Skill classification matching GnuBG's arSkillLevel thresholds.
    /// Port of skill_classify() from gnubgapi.c.
    /// </summary>
    private static SkillLevel ClassifySkill(float rSkill)
    {
        if (rSkill < -0.12f) return SkillLevel.VeryBad;
        if (rSkill < -0.06f) return SkillLevel.Bad;
        if (rSkill < -0.03f) return SkillLevel.Doubtful;
        return SkillLevel.None;
    }

    /// <summary>
    /// Rating string from error per move.
    /// Port of get_rating() from gnubgapi.c.
    /// </summary>
    private static string GetRating(float errorPerMove)
    {
        if (errorPerMove < 0.005f) return "Super Grandmaster";
        if (errorPerMove < 0.008f) return "Grandmaster";
        if (errorPerMove < 0.013f) return "Master";
        if (errorPerMove < 0.020f) return "Advanced";
        if (errorPerMove < 0.032f) return "Intermediate";
        return "Beginner";
    }

    // ---- Resignation Analysis ----

    /// <summary>
    /// Determine whether a player should resign and at what level.
    /// Port of getResignation() from rollout.c.
    /// Returns 0 (no resign), 1 (normal), 2 (gammon), or 3 (backgammon).
    /// </summary>
    public int GetResignation(Board board, CubeInfo? cubeInfo = null)
    {
        cubeInfo ??= CubeInfo.Money();
        float[] output = new float[Constants.NumOutputs];
        _evaluator.EvaluatePosition(board, output);
        return Resignation.GetResignation(output, cubeInfo, _met);
    }

    /// <summary>
    /// Determine whether a player should resign from a position ID.
    /// </summary>
    public int GetResignation(string positionId, CubeInfo? cubeInfo = null)
    {
        var board = PositionId.Decode(positionId)
            ?? throw new ArgumentException($"Invalid position ID: {positionId}");
        return GetResignation(board, cubeInfo);
    }

    /// <summary>
    /// Calculate equity before and after resignation at a specific level.
    /// Port of getResignEquities() from rollout.c.
    /// </summary>
    public ResignationResult GetResignEquities(Board board, int resignLevel, CubeInfo? cubeInfo = null)
    {
        cubeInfo ??= CubeInfo.Money();
        float[] output = new float[Constants.NumOutputs];
        _evaluator.EvaluatePosition(board, output);
        Resignation.GetResignEquities(output, cubeInfo, resignLevel,
            out float eqBefore, out float eqAfter, _met);
        return new ResignationResult(resignLevel, eqBefore, eqAfter);
    }

    /// <summary>
    /// Check whether an opponent's resignation should be accepted.
    /// Returns the acceptable level (1-3) or 0 to reject.
    /// </summary>
    public int CheckResignation(Board board, int resignLevel, CubeInfo? cubeInfo = null, float maxCost = 0.05f)
    {
        cubeInfo ??= CubeInfo.Money();
        float[] output = new float[Constants.NumOutputs];
        _evaluator.EvaluatePosition(board, output);
        return Resignation.CheckResignation(output, cubeInfo, resignLevel, _met, maxCost);
    }

    // ---- Board Display ----

    /// <summary>
    /// Draw the board in ASCII art.
    /// Port of DrawBoard() from drawboard.c.
    /// </summary>
    public string DrawBoard(Board board, bool playerOnRoll = true, string[]? annotations = null,
        string? matchId = null, int nChequers = 15, bool clockwise = false)
        => BoardDisplay.DrawBoard(board, playerOnRoll, annotations, matchId, nChequers, clockwise);

    /// <summary>
    /// Generate FIBS "boardstyle 3" protocol representation.
    /// Port of FIBSBoard() from drawboard.c.
    /// </summary>
    public static string FIBSBoard(Board board, bool playerOnRoll, string playerName, string opponentName,
        int matchLength, int playerScore, int opponentScore, int die0, int die1,
        int cubeValue = 1, int cubeOwner = -1, bool crawford = false)
        => BoardDisplay.FIBSBoard(board, playerOnRoll, playerName, opponentName,
            matchLength, playerScore, opponentScore, die0, die1, cubeValue, cubeOwner, crawford);

    // ---- Feature Extraction ----

    /// <summary>
    /// Extract neural network input features from a board position.
    /// Returns 248 floats matching gnubgapi_position_to_features().
    /// Port of gnubgapi_position_to_features() from gnubgapi.c.
    /// </summary>
    public static float[] ExtractFeatures(Board board, bool isTopOnRoll = false)
    {
        float[] features = new float[NeuralNet.FeatureExtractor.FeatureDim];
        NeuralNet.FeatureExtractor.ExtractFeatures(board, isTopOnRoll, features);
        return features;
    }

    /// <summary>
    /// Extract neural network input features from a position ID.
    /// </summary>
    public static float[] ExtractFeatures(string positionId, bool isTopOnRoll = false)
    {
        var board = PositionId.Decode(positionId)
            ?? throw new ArgumentException($"Invalid position ID: {positionId}");
        return ExtractFeatures(board, isTopOnRoll);
    }

    /// <summary>
    /// Extract raw neural network inputs for a specific position class.
    /// Returns the full input array that would be fed to the neural net.
    /// </summary>
    public float[] ExtractRawInputs(Board board)
    {
        var pc = _evaluator.ClassifyPosition(board);
        int size = pc switch
        {
            PositionClass.Race => Constants.NumRaceInputs,
            PositionClass.Contact => Constants.NumContactInputs,
            PositionClass.Crashed => Constants.NumCrashedInputs,
            _ => Constants.NumContactInputs,
        };
        float[] inputs = new float[size];
        NeuralNet.FeatureExtractor.ExtractRawInputs(board, pc, inputs);
        return inputs;
    }

    public void Dispose()
    {
        // No unmanaged resources, but implement IDisposable for API compatibility
    }
}
