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
    private readonly BearoffDatabase? _oneSidedBearoff;
    private readonly BearoffDatabase? _twoSidedBearoff;
    private readonly Evaluator _evaluator;
    private readonly MatchEquityTable _met;

    private Engine(NetworkSet nets, BearoffDatabase? osBearoff, BearoffDatabase? tsBearoff, MatchEquityTable met)
    {
        _nets = nets;
        _oneSidedBearoff = osBearoff;
        _twoSidedBearoff = tsBearoff;
        _evaluator = new Evaluator(nets, osBearoff, tsBearoff);
        _met = met;
    }

    /// <summary>
    /// Create an engine instance by loading data files from the specified directory.
    /// Expects: gnubg.wd (required), gnubg_os0.bd (optional), gnubg_ts0.bd (optional).
    /// </summary>
    public static Engine Create(string dataDir)
    {
        var weightsPath = Path.Combine(dataDir, "gnubg.wd");
        if (!File.Exists(weightsPath))
            throw new FileNotFoundException($"Weights file not found: {weightsPath}");

        var nets = NetworkSet.LoadBinary(weightsPath);

        BearoffDatabase? os = null, ts = null;
        var osPath = Path.Combine(dataDir, "gnubg_os0.bd");
        if (File.Exists(osPath)) os = BearoffDatabase.Load(osPath);

        var tsPath = Path.Combine(dataDir, "gnubg_ts0.bd");
        if (File.Exists(tsPath)) ts = BearoffDatabase.Load(tsPath);

        var met = MatchEquityTable.ComputeDefault();

        return new Engine(nets, os, ts, met);
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

        BearoffDatabase? os = null, ts = null;
        var osPath = Path.Combine(dataDir, "gnubg_os0.bd");
        if (File.Exists(osPath)) os = BearoffDatabase.Load(osPath);

        var tsPath = Path.Combine(dataDir, "gnubg_ts0.bd");
        if (File.Exists(tsPath)) ts = BearoffDatabase.Load(tsPath);

        var met = MetXmlLoader.LoadFromFile(metXmlPath);

        return new Engine(nets, os, ts, met);
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

        float[] output = new float[Constants.NumOutputs];
        _evaluator.EvaluatePosition(board, output);

        float equity = MatchEquityTable.MoneyEquity(output);

        return new EvaluationResult(
            Win: output[Constants.OutputWin],
            WinGammon: output[Constants.OutputWinGammon],
            WinBackgammon: output[Constants.OutputWinBackgammon],
            LoseGammon: output[Constants.OutputLoseGammon],
            LoseBackgammon: output[Constants.OutputLoseBackgammon],
            Equity: equity);
    }

    /// <summary>
    /// Evaluate a position from a Board directly.
    /// </summary>
    public EvaluationResult EvaluatePosition(Board board)
    {
        float[] output = new float[Constants.NumOutputs];
        _evaluator.EvaluatePosition(board, output);
        float equity = MatchEquityTable.MoneyEquity(output);

        return new EvaluationResult(
            Win: output[Constants.OutputWin],
            WinGammon: output[Constants.OutputWinGammon],
            WinBackgammon: output[Constants.OutputWinBackgammon],
            LoseGammon: output[Constants.OutputLoseGammon],
            LoseBackgammon: output[Constants.OutputLoseBackgammon],
            Equity: equity);
    }

    /// <summary>
    /// Evaluate a position at the specified number of plies.
    /// Higher plies = stronger but slower. 0=instant, 1=fast, 2=world-class.
    /// </summary>
    public EvaluationResult EvaluatePositionPlied(string positionId, int plies, string? matchId = null)
    {
        var board = PositionId.Decode(positionId)
            ?? throw new ArgumentException($"Invalid position ID: {positionId}");

        return EvaluatePositionPlied(board, plies);
    }

    /// <summary>
    /// Evaluate a position at the specified number of plies from a Board.
    /// </summary>
    public EvaluationResult EvaluatePositionPlied(Board board, int plies)
    {
        float[] output = new float[Constants.NumOutputs];
        _evaluator.EvaluatePositionPlied(board, output, plies);
        float equity = MatchEquityTable.MoneyEquity(output);

        return new EvaluationResult(
            Win: output[Constants.OutputWin],
            WinGammon: output[Constants.OutputWinGammon],
            WinBackgammon: output[Constants.OutputWinBackgammon],
            LoseGammon: output[Constants.OutputLoseGammon],
            LoseBackgammon: output[Constants.OutputLoseBackgammon],
            Equity: equity);
    }

    /// <summary>
    /// Evaluate a position returning full 7-output result (5 probs + cubeless + cubeful equity).
    /// </summary>
    public FullEvaluationResult EvaluatePositionFull(string positionId, string? matchId = null)
    {
        var board = PositionId.Decode(positionId)
            ?? throw new ArgumentException($"Invalid position ID: {positionId}");

        float[] output = new float[Constants.NumOutputs];
        _evaluator.EvaluatePosition(board, output);
        float cubelessEquity = MatchEquityTable.MoneyEquity(output);

        return new FullEvaluationResult(
            WinProbability: output[Constants.OutputWin],
            WinGammonProbability: output[Constants.OutputWinGammon],
            WinBackgammonProbability: output[Constants.OutputWinBackgammon],
            LoseGammonProbability: output[Constants.OutputLoseGammon],
            LoseBackgammonProbability: output[Constants.OutputLoseBackgammon],
            CubelessEquity: cubelessEquity,
            CubefulEquity: cubelessEquity); // TODO: proper cubeful equity from cube decisions
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
            Evaluator.InvertEvaluation(output);

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
        _evaluator.FindnSaveBestMoves(ml, board, die1, die2, ec, moveFilters);

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
        return Classifier.Classify(board, _evaluator);
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

        var pc = Classifier.Classify(board, _evaluator);
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

        var pc = Classifier.Classify(board, _evaluator);
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

        var pc = Classifier.Classify(board, _evaluator);
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
        var engine = new RolloutEngine(_evaluator);
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
        var engine = new RolloutEngine(_evaluator);
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
        return new EvaluationResult(output[0], output[1], output[2], output[3], output[4], equity);
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
                ? GenerateMovesWithEvalPlied(board, turn.Die1, turn.Die2, plies)
                : GenerateMovesWithEval(board, turn.Die1, turn.Die2);

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

    public void Dispose()
    {
        // No unmanaged resources, but implement IDisposable for API compatibility
    }
}
