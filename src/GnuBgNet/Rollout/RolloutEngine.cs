// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2019 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Port of rollout.c (BasicCubefulRollout / RolloutGeneral)

using GnuBgNet.Evaluation;
using GnuBgNet.MatchEquity;
using GnuBgNet.MoveGeneration;
using GnuBgNet.Random;

namespace GnuBgNet.Rollout;

/// <summary>
/// Monte Carlo rollout engine. Plays out positions with random dice
/// and evaluates using neural nets at truncation point.
/// Port of BasicCubefulRollout / RolloutGeneral from rollout.c.
/// </summary>
public sealed class RolloutEngine
{
    private readonly Evaluator _evaluator;

    public RolloutEngine(Evaluator evaluator)
    {
        _evaluator = evaluator;
    }

    /// <summary>
    /// Roll out a position. Returns mean probabilities, cubeless equity, and standard deviations.
    /// </summary>
    public RolloutResult Rollout(Board board, RolloutSettings settings)
    {
        int trials = (int)settings.Trials;
        int truncPlies = settings.Truncate ? settings.TruncatePlies : int.MaxValue;
        int chequerPlies = settings.ChequerPlies;

        // Run trials in parallel with per-thread state
        var locker = new object();
        double[] totalSum = new double[7];
        double[] totalSumSq = new double[7];
        int totalCount = 0;

        Parallel.For(0, trials, () => new ThreadLocalState(), (trial, _, state) =>
        {
            state.Rng ??= new MersenneTwister(settings.Seed == 0 ? (uint)trial : settings.Seed + (uint)trial);
            state.Output ??= new float[Constants.NumOutputs];

            var trialResult = RunSingleTrial(board, state.Rng, truncPlies, chequerPlies,
                settings, state.Output, (uint)trial);

            for (int i = 0; i < 7; i++)
            {
                state.Sum[i] += trialResult[i];
                state.SumSq[i] += trialResult[i] * trialResult[i];
            }
            state.Count++;

            return state;
        },
        state =>
        {
            lock (locker)
            {
                for (int i = 0; i < 7; i++)
                {
                    totalSum[i] += state.Sum[i];
                    totalSumSq[i] += state.SumSq[i];
                }
                totalCount += state.Count;
            }
        });

        // Compute means and standard deviations
        double[] mean = new double[7];
        double[] stddev = new double[7];
        for (int i = 0; i < 7; i++)
        {
            mean[i] = totalSum[i] / totalCount;
            double variance = (totalSumSq[i] / totalCount) - (mean[i] * mean[i]);
            stddev[i] = Math.Sqrt(Math.Max(0, variance) / totalCount);
        }

        return new RolloutResult(
            WinProbability: mean[0],
            WinGammonProbability: mean[1],
            WinBackgammonProbability: mean[2],
            LoseGammonProbability: mean[3],
            LoseBackgammonProbability: mean[4],
            CubelessEquity: mean[5],
            CubefulEquity: mean[6],
            WinProbabilityStdDev: stddev[0],
            WinGammonProbabilityStdDev: stddev[1],
            WinBackgammonProbabilityStdDev: stddev[2],
            LoseGammonProbabilityStdDev: stddev[3],
            LoseBackgammonProbabilityStdDev: stddev[4],
            CubelessEquityStdDev: stddev[5],
            CubefulEquityStdDev: stddev[6]);
    }

    /// <summary>
    /// Roll out multiple candidate moves and return results for each.
    /// Supports JSD-based early stopping to skip clearly inferior moves.
    /// </summary>
    public RolloutResult[] RolloutMoves(Board board, int die1, int die2,
        RolloutSettings settings, out int gamesPlayed)
    {
        var ml = MoveGenerator.GenerateMoves(board, die1, die2);
        if (ml.Moves.Count == 0)
        {
            gamesPlayed = 0;
            return [];
        }

        int nAlts = ml.Moves.Count;
        int trials = (int)settings.Trials;

        // Prepare candidate boards
        var candidateBoards = new Board[nAlts];
        for (int i = 0; i < nAlts; i++)
            candidateBoards[i] = MoveGenerator.ApplyMove(board, ml.Moves[i]);

        // Per-alternative accumulators
        var altSum = new double[nAlts][];
        var altSumSq = new double[nAlts][];
        var altCount = new int[nAlts];
        var stopped = new bool[nAlts];

        for (int i = 0; i < nAlts; i++)
        {
            altSum[i] = new double[7];
            altSumSq[i] = new double[7];
        }

        int truncPlies = settings.Truncate ? settings.TruncatePlies : int.MaxValue;
        int chequerPlies = settings.ChequerPlies;

        for (int trial = 0; trial < trials; trial++)
        {
            var rng = new MersenneTwister(settings.Seed == 0 ? (uint)trial : settings.Seed + (uint)trial);
            float[] output = new float[Constants.NumOutputs];

            for (int alt = 0; alt < nAlts; alt++)
            {
                if (stopped[alt]) continue;

                var trialResult = RunSingleTrial(candidateBoards[alt], rng, truncPlies,
                    chequerPlies, settings, output, (uint)trial);

                for (int j = 0; j < 7; j++)
                {
                    altSum[alt][j] += trialResult[j];
                    altSumSq[alt][j] += trialResult[j] * trialResult[j];
                }
                altCount[alt]++;
            }

            // JSD stopping check
            if (settings.StopOnJsd && trial >= (int)settings.MinimumJsdGames && nAlts > 1)
            {
                CheckJsdStopping(altSum, altSumSq, altCount, stopped, nAlts, settings.JsdLimit);
            }
        }

        gamesPlayed = trials;

        // Build results
        var results = new RolloutResult[nAlts];
        for (int alt = 0; alt < nAlts; alt++)
        {
            int n = altCount[alt];
            if (n == 0) n = 1;
            double[] mean = new double[7];
            double[] stddev = new double[7];
            for (int j = 0; j < 7; j++)
            {
                mean[j] = altSum[alt][j] / n;
                double variance = (altSumSq[alt][j] / n) - (mean[j] * mean[j]);
                stddev[j] = Math.Sqrt(Math.Max(0, variance) / n);
            }

            results[alt] = new RolloutResult(
                mean[0], mean[1], mean[2], mean[3], mean[4], mean[5], mean[6],
                stddev[0], stddev[1], stddev[2], stddev[3], stddev[4], stddev[5], stddev[6]);
        }

        return results;
    }

    /// <summary>
    /// Run a single rollout trial: alternating moves until game over or truncation.
    /// Returns 7 values: 5 probabilities + cubeless equity + cubeful equity.
    /// Port of BasicCubefulRollout from rollout.c.
    /// </summary>
    private double[] RunSingleTrial(Board startBoard, MersenneTwister rng,
        int truncPlies, int chequerPlies, RolloutSettings settings, float[] output, uint trialIndex)
    {
        var board = startBoard.Clone();
        int ply = 0;
        bool playerOnRoll = true;
        double[] varRedn = settings.VarianceReduction ? new double[7] : null!;

        while (ply < truncPlies)
        {
            // Check for game over
            if (IsGameOver(board, playerOnRoll, out var gameOverResult))
            {
                if (settings.VarianceReduction)
                    ApplyVarianceReduction(gameOverResult, varRedn);
                return gameOverResult;
            }

            // Bearoff truncation
            if (TryBearoffTruncation(board, playerOnRoll, settings, output, out var bearoffResult))
            {
                if (settings.VarianceReduction)
                    ApplyVarianceReduction(bearoffResult, varRedn);
                return bearoffResult;
            }

            // Roll dice (quasi-random rotation for first roll if enabled)
            int d0, d1;
            if (settings.Rotate && ply == 0)
            {
                // Quasi-random: cycle through all 36 outcomes
                int diceIdx = (int)(trialIndex % 36);
                d0 = diceIdx / 6 + 1;
                d1 = diceIdx % 6 + 1;
                if (d0 < d1) (d0, d1) = (d1, d0);
            }
            else
            {
                (d0, d1) = rng.NextDiceRoll();
            }

            // Variance reduction: evaluate all 36 dice outcomes at 0-ply
            if (settings.VarianceReduction)
            {
                AccumulateVarianceReduction(board, playerOnRoll, d0, d1, output, varRedn);
            }

            // Find best move at the configured chequer ply depth
            var activeBoard = playerOnRoll ? board : board.Swapped();
            var ml = MoveGenerator.GenerateMoves(activeBoard, d0, d1);

            if (ml.Moves.Count > 0)
            {
                Board bestBoard = activeBoard;
                float bestScore = float.MinValue;

                foreach (var move in ml.Moves)
                {
                    var newBoard = MoveGenerator.ApplyMove(activeBoard, move);
                    var swapped = newBoard.Swapped();

                    if (chequerPlies > 0)
                        _evaluator.EvaluatePositionPlied(swapped, output, chequerPlies - 1);
                    else
                        _evaluator.EvaluatePosition(swapped, output);

                    Evaluator.InvertEvaluation(output);
                    float score = MatchEquityTable.MoneyEquity(output);

                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestBoard = newBoard;
                    }
                }

                board = playerOnRoll ? bestBoard : bestBoard.Swapped();
            }

            playerOnRoll = !playerOnRoll;
            ply++;
        }

        // Truncated: evaluate at current position
        var evalBoard = playerOnRoll ? board : board.Swapped();
        _evaluator.EvaluatePosition(evalBoard, output);
        if (!playerOnRoll)
            Evaluator.InvertEvaluation(output);

        float equity = MatchEquityTable.MoneyEquity(output);
        double[] result =
        [
            output[Constants.OutputWin],
            output[Constants.OutputWinGammon],
            output[Constants.OutputWinBackgammon],
            output[Constants.OutputLoseGammon],
            output[Constants.OutputLoseBackgammon],
            equity,
            equity,
        ];

        if (settings.VarianceReduction)
            ApplyVarianceReduction(result, varRedn);

        return result;
    }

    /// <summary>
    /// Attempt bearoff truncation: if position is a bearoff position, evaluate directly.
    /// Port of bearoff truncation from BasicCubefulRollout.
    /// </summary>
    private bool TryBearoffTruncation(Board board, bool playerOnRoll,
        RolloutSettings settings, float[] output, out double[] result)
    {
        result = null!;

        var evalBoard = playerOnRoll ? board : board.Swapped();
        var pc = Classifier.Classify(evalBoard, _evaluator);

        // Two-sided bearoff truncation
        if (settings.TruncateBearoff2 && pc == PositionClass.BearoffTwoSided)
        {
            _evaluator.EvaluatePositionByClass(evalBoard, output, pc);
            if (!playerOnRoll)
                Evaluator.InvertEvaluation(output);

            float eq = MatchEquityTable.MoneyEquity(output);
            result =
            [
                output[Constants.OutputWin],
                output[Constants.OutputWinGammon],
                output[Constants.OutputWinBackgammon],
                output[Constants.OutputLoseGammon],
                output[Constants.OutputLoseBackgammon],
                eq, eq,
            ];
            return true;
        }

        // One-sided bearoff truncation
        if (settings.TruncateBearoffOS && pc == PositionClass.BearoffOneSided)
        {
            _evaluator.EvaluatePositionByClass(evalBoard, output, pc);
            if (!playerOnRoll)
                Evaluator.InvertEvaluation(output);

            float eq = MatchEquityTable.MoneyEquity(output);
            result =
            [
                output[Constants.OutputWin],
                output[Constants.OutputWinGammon],
                output[Constants.OutputWinBackgammon],
                output[Constants.OutputLoseGammon],
                output[Constants.OutputLoseBackgammon],
                eq, eq,
            ];
            return true;
        }

        return false;
    }

    /// <summary>
    /// Variance reduction: evaluate all 36 dice outcomes at 0-ply, compute mean,
    /// then accumulate the correction (mean - actual_roll_value).
    /// Port of variance reduction from BasicCubefulRollout.
    /// </summary>
    private void AccumulateVarianceReduction(Board board, bool playerOnRoll,
        int actualD0, int actualD1, float[] output, double[] varRedn)
    {
        var activeBoard = playerOnRoll ? board : board.Swapped();
        double[] mean = new double[7];
        double[] actualRollValue = new double[7];

        // Evaluate all 21 distinct dice combinations
        for (int d0 = 1; d0 <= 6; d0++)
        {
            for (int d1 = 1; d1 <= d0; d1++)
            {
                float w = (d0 == d1) ? 1.0f : 2.0f;

                var ml = MoveGenerator.GenerateMoves(activeBoard, d0, d1);
                Board bestBoard;
                if (ml.Moves.Count > 0)
                {
                    bestBoard = activeBoard;
                    float bestScore = float.MinValue;
                    foreach (var move in ml.Moves)
                    {
                        var nb = MoveGenerator.ApplyMove(activeBoard, move);
                        var sw = nb.Swapped();
                        _evaluator.EvaluatePosition(sw, output);
                        Evaluator.InvertEvaluation(output);
                        float sc = MatchEquityTable.MoneyEquity(output);
                        if (sc > bestScore) { bestScore = sc; bestBoard = nb; }
                    }
                }
                else
                {
                    bestBoard = activeBoard.Clone();
                }

                var evalBoard2 = bestBoard.Swapped();
                _evaluator.EvaluatePosition(evalBoard2, output);
                Evaluator.InvertEvaluation(output);
                float eq = MatchEquityTable.MoneyEquity(output);

                double[] vals =
                [
                    output[Constants.OutputWin],
                    output[Constants.OutputWinGammon],
                    output[Constants.OutputWinBackgammon],
                    output[Constants.OutputLoseGammon],
                    output[Constants.OutputLoseBackgammon],
                    eq, eq,
                ];

                for (int i = 0; i < 7; i++)
                    mean[i] += w * vals[i];

                // Check if this is the actual roll
                if ((d0 == actualD0 && d1 == actualD1) || (d0 == actualD1 && d1 == actualD0))
                {
                    for (int i = 0; i < 7; i++)
                        actualRollValue[i] = vals[i];
                }
            }
        }

        // Normalize mean by 36
        for (int i = 0; i < 7; i++)
        {
            mean[i] /= 36.0;
            // Accumulate correction: mean - actual
            varRedn[i] += mean[i] - actualRollValue[i];
        }
    }

    private static void ApplyVarianceReduction(double[] result, double[] varRedn)
    {
        if (varRedn == null) return;
        for (int i = 0; i < 7; i++)
            result[i] += varRedn[i];
    }

    /// <summary>
    /// JSD stopping: check if any alternative is clearly worse than the best
    /// and can be stopped early.
    /// </summary>
    private static void CheckJsdStopping(double[][] altSum, double[][] altSumSq,
        int[] altCount, bool[] stopped, int nAlts, float jsdLimit)
    {
        // Find best alternative by equity (index 5)
        int bestAlt = -1;
        double bestEq = double.MinValue;
        for (int i = 0; i < nAlts; i++)
        {
            if (stopped[i]) continue;
            double eq = altSum[i][5] / altCount[i];
            if (eq > bestEq) { bestEq = eq; bestAlt = i; }
        }
        if (bestAlt < 0) return;

        double bestVar = (altSumSq[bestAlt][5] / altCount[bestAlt])
            - (bestEq * bestEq);
        double bestSigma = Math.Sqrt(Math.Max(0, bestVar) / altCount[bestAlt]);

        for (int i = 0; i < nAlts; i++)
        {
            if (i == bestAlt || stopped[i]) continue;
            int n = altCount[i];
            if (n < 2) continue;

            double eq = altSum[i][5] / n;
            double var2 = (altSumSq[i][5] / n) - (eq * eq);
            double sigma = Math.Sqrt(Math.Max(0, var2) / n);

            double combinedSigma = Math.Sqrt(bestSigma * bestSigma + sigma * sigma);
            if (combinedSigma > 0)
            {
                double jsd = (bestEq - eq) / combinedSigma;
                if (jsd > jsdLimit)
                    stopped[i] = true;
            }
        }
    }

    private static bool IsGameOver(Board board, bool playerOnRoll, out double[] result)
    {
        result = null!;

        // Check if player has borne off all pieces
        bool playerHasPieces = false;
        for (int i = 0; i < 25; i++)
            if (board.Player[i] > 0) { playerHasPieces = true; break; }

        if (!playerHasPieces)
        {
            // Player (original) has won
            int oppCount = 0;
            for (int i = 0; i < 25; i++) oppCount += (int)board.Opponent[i];
            bool gammon = oppCount == 15;
            bool backgammon = gammon && HasCheckersInHomeOrBar(board.Opponent);

            result = playerOnRoll
                ? [1, gammon ? 1 : 0, backgammon ? 1 : 0, 0, 0, ComputeEquity(1, gammon, backgammon, false, false), ComputeEquity(1, gammon, backgammon, false, false)]
                : [0, 0, 0, gammon ? 1 : 0, backgammon ? 1 : 0, ComputeEquity(0, false, false, gammon, backgammon), ComputeEquity(0, false, false, gammon, backgammon)];
            return true;
        }

        bool oppHasPieces = false;
        for (int i = 0; i < 25; i++)
            if (board.Opponent[i] > 0) { oppHasPieces = true; break; }

        if (!oppHasPieces)
        {
            // Opponent (original) has won
            int plCount = 0;
            for (int i = 0; i < 25; i++) plCount += (int)board.Player[i];
            bool gammon = plCount == 15;
            bool backgammon = gammon && HasCheckersInHomeOrBar(board.Player);

            result = playerOnRoll
                ? [0, 0, 0, gammon ? 1 : 0, backgammon ? 1 : 0, ComputeEquity(0, false, false, gammon, backgammon), ComputeEquity(0, false, false, gammon, backgammon)]
                : [1, gammon ? 1 : 0, backgammon ? 1 : 0, 0, 0, ComputeEquity(1, gammon, backgammon, false, false), ComputeEquity(1, gammon, backgammon, false, false)];
            return true;
        }

        return false;
    }

    private static bool HasCheckersInHomeOrBar(uint[] side)
    {
        for (int i = 18; i < 25; i++)
            if (side[i] > 0) return true;
        return false;
    }

    private static double ComputeEquity(double win, bool wg, bool wbg, bool lg, bool lbg)
    {
        return win * 2.0 - 1.0 + (wg ? 1 : 0) + (wbg ? 1 : 0) - (lg ? 1 : 0) - (lbg ? 1 : 0);
    }

    private sealed class ThreadLocalState
    {
        public MersenneTwister? Rng;
        public float[]? Output;
        public readonly double[] Sum = new double[7];
        public readonly double[] SumSq = new double[7];
        public int Count;
    }
}
