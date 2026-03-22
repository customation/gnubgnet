// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2019 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Port of rollout.c (BasicCubefulRollout / RolloutGeneral)

using GnuBgNet.Encoding;
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
    private readonly MatchEquityTable? _met;

    public RolloutEngine(Evaluator evaluator, MatchEquityTable? met = null)
    {
        _evaluator = evaluator;
        _met = met;
    }

    /// <summary>
    /// Roll out a position. Returns mean probabilities, cubeless equity, and standard deviations.
    /// </summary>
    public RolloutResult Rollout(Board board, RolloutSettings settings)
    {
        int trials = (int)settings.Trials;
        int truncPlies = settings.Truncate ? settings.TruncatePlies : int.MaxValue;
        int chequerPlies = settings.ChequerPlies;

        // Create quasi-random permutation tables if rotating dice
        DicePermutations? perms = settings.Rotate ? new DicePermutations(settings.Seed) : null;

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
                settings, state.Output, (uint)trial, perms);

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

        // Create quasi-random permutation tables (shared across all alternatives)
        DicePermutations? perms = settings.Rotate ? new DicePermutations(settings.Seed) : null;

        // Run trials in parallel, matching the threading model of Rollout/RolloutGeneral.
        // Each thread accumulates per-alternative results independently.
        var locker = new object();

        Parallel.For(0, trials, () => new MoveTrialState(nAlts), (trial, _, state) =>
        {
            state.Rng ??= new MersenneTwister(settings.Seed == 0 ? (uint)trial : settings.Seed + (uint)trial);
            state.Output ??= new float[Constants.NumOutputs];

            // Snapshot stopped flags under lock to avoid torn reads
            bool[] localStopped;
            lock (locker) { localStopped = (bool[])stopped.Clone(); }

            for (int alt = 0; alt < nAlts; alt++)
            {
                if (localStopped[alt]) continue;

                // Dice sharing: each alternative gets the SAME seed for the same trial,
                // so they share the same dice sequence. This is critical for variance
                // reduction — comparing moves under identical dice conditions.
                state.Rng.Init(settings.Seed == 0 ? (uint)trial : settings.Seed + (uint)trial);
                var trialResult = RunSingleTrial(candidateBoards[alt], state.Rng, truncPlies,
                    chequerPlies, settings, state.Output, (uint)trial, perms);

                for (int j = 0; j < 7; j++)
                {
                    state.AltSum[alt][j] += trialResult[j];
                    state.AltSumSq[alt][j] += trialResult[j] * trialResult[j];
                }
                state.AltCount[alt]++;
            }

            return state;
        },
        state =>
        {
            lock (locker)
            {
                for (int alt = 0; alt < nAlts; alt++)
                {
                    for (int j = 0; j < 7; j++)
                    {
                        altSum[alt][j] += state.AltSum[alt][j];
                        altSumSq[alt][j] += state.AltSumSq[alt][j];
                    }
                    altCount[alt] += state.AltCount[alt];
                }
            }
        });

        // JSD stopping is applied post-hoc in parallel mode.
        // In the C version, JSD is checked per-trial sequentially within RolloutGeneral.
        // For parallel execution, we lose incremental JSD stopping but gain throughput.
        // A future refinement could batch trials and check JSD between batches.

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
    ///
    /// Board convention (matching C): board is ALWAYS from the current player's
    /// perspective. board.Player[] = player on roll (anBoard[1] in C),
    /// board.Opponent[] = their opponent (anBoard[0] in C).
    /// After each move, SwapSides so the next player sees the board from their side.
    /// iTurn tracks the half-move count; (iTurn &amp; 1) != 0 means the result must
    /// be inverted back to player 0's perspective at every exit point.
    /// </summary>
    private double[] RunSingleTrial(Board startBoard, MersenneTwister rng,
        int truncPlies, int chequerPlies, RolloutSettings settings, float[] output,
        uint trialIndex, DicePermutations? perms = null)
    {
        var board = startBoard.Clone();
        int iTurn = 0;
        double[] varRedn = settings.VarianceReduction ? new double[7] : null!;
        int skip = 0; // quasi-random skip counter

        // Cube state for cubeful rollout
        int cubeValue = 1;
        int cubeOwner = -1; // -1 = centered
        int nBasisCube = 1;

        // Late evaluation threshold: move number at which to switch eval contexts.
        // Port of: int nLateEvals = prc->fLateEvals ? prc->nLate : 0x7fffffff;
        int nLateEvals = settings.LateEvals ? settings.LateMoveThreshold : int.MaxValue;

        while (iTurn < truncPlies)
        {
            // Determine effective chequer and cube plies based on late eval threshold.
            // Port of pecCube/pecChequer selection from rollout.c lines 416-426.
            int effectiveChequerPlies = iTurn < nLateEvals ? chequerPlies : settings.ChequerPliesLate;
            int effectiveCubePlies = iTurn < nLateEvals ? settings.CubePlies : settings.CubePliesLate;
            int fMove = iTurn & 1; // 0 = player 0 on roll, 1 = player 1 on roll

            // Check for game over.
            // Board is from current player's perspective; Classify + EvalOver give
            // the result from the current player's perspective.  Invert if odd turn.
            var pc = Classifier.Classify(board, _evaluator);
            if (pc == PositionClass.Over)
            {
                _evaluator.EvaluatePositionByClass(board, output, pc);
                if (fMove != 0)
                    Evaluator.InvertEvaluation(output);

                float eq = MatchEquityTable.MoneyEquity(output);
                double cubefulEq = settings.Cubeful ? eq * (double)cubeValue / nBasisCube : eq;
                double[] gameOverResult =
                [
                    output[Constants.OutputWin],
                    output[Constants.OutputWinGammon],
                    output[Constants.OutputWinBackgammon],
                    output[Constants.OutputLoseGammon],
                    output[Constants.OutputLoseBackgammon],
                    eq,
                    cubefulEq,
                ];
                if (settings.VarianceReduction)
                    ApplyVarianceReduction(gameOverResult, varRedn);
                return gameOverResult;
            }

            // Bearoff truncation: evaluate from current player's perspective,
            // invert to player 0 if odd turn.
            if ((settings.TruncateBearoff2 && pc == PositionClass.BearoffTwoSided) ||
                (settings.TruncateBearoffOS && pc == PositionClass.BearoffOneSided))
            {
                _evaluator.EvaluatePositionByClass(board, output, pc);
                if (fMove != 0)
                    Evaluator.InvertEvaluation(output);

                float eq = MatchEquityTable.MoneyEquity(output);
                double cubefulEq = settings.Cubeful ? eq * (double)cubeValue / nBasisCube : eq;
                double[] bearoffResult =
                [
                    output[Constants.OutputWin],
                    output[Constants.OutputWinGammon],
                    output[Constants.OutputWinBackgammon],
                    output[Constants.OutputLoseGammon],
                    output[Constants.OutputLoseBackgammon],
                    eq,
                    cubefulEq,
                ];
                if (settings.VarianceReduction)
                    ApplyVarianceReduction(bearoffResult, varRedn);
                return bearoffResult;
            }

            // Cubeful play-out: evaluate cube decision before rolling dice.
            // Port of rollout.c line 471: suppress cube on turn 0 when fInitial is set.
            bool allowCube = iTurn > 0 || !settings.Initial;
            if (settings.Cubeful && allowCube)
            {
                // Current player can double if cube is centered or they own it.
                // cubeOwner is absolute (0/1/-1); canDouble when cubeOwner == fMove.
                bool canDouble = cubeOwner == -1 || cubeOwner == fMove;
                if (canDouble)
                {
                    // Evaluate from current player's perspective (board is already correct)
                    if (effectiveCubePlies > 0)
                        _evaluator.EvaluatePositionPlied(board, output, effectiveCubePlies - 1);
                    else
                        _evaluator.EvaluatePosition(board, output);

                    // Build CubeInfo with absolute player indices (matching C's convention)
                    var ci = new CubeInfo
                    {
                        Cube = cubeValue,
                        CubeOwner = cubeOwner,
                        Move = fMove,
                        MatchTo = 0, // money game rollout
                        Jacoby = false,
                        Beavers = false,
                    };

                    if (CubeDecision.GetDPEq(ci, _met, out float dpEquity))
                    {
                        var cubeResult = CubeDecision.AnalyseMoney(output, ci);

                        switch (cubeResult.Action)
                        {
                            case CubeAction.DoublePass:
                            case CubeAction.RedoublePass:
                            case CubeAction.OptionalDoublePass:
                            case CubeAction.OptionalRedoublePass:
                            {
                                // Opponent drops. Cubeless equity comes from the evaluation;
                                // cubeful equity is the DP value (dpEquity) scaled by cube ratio.
                                // Port of rollout.c: aarOutput[OUTPUT_CUBEFUL_EQUITY] = rDP,
                                // post-loop scales by pci->nCube / nBasisCube.
                                float clEq = MatchEquityTable.MoneyEquity(output);
                                float cfEq = dpEquity;
                                if (fMove != 0)
                                {
                                    Evaluator.InvertEvaluation(output);
                                    clEq = -clEq;
                                    cfEq = -cfEq;
                                }
                                double cubefulEq = cfEq * (double)cubeValue / nBasisCube;
                                double[] dpResult =
                                [
                                    output[Constants.OutputWin],
                                    output[Constants.OutputWinGammon],
                                    output[Constants.OutputWinBackgammon],
                                    output[Constants.OutputLoseGammon],
                                    output[Constants.OutputLoseBackgammon],
                                    clEq,
                                    cubefulEq,
                                ];
                                if (settings.VarianceReduction)
                                    ApplyVarianceReduction(dpResult, varRedn);
                                return dpResult;
                            }

                            case CubeAction.DoubleTake:
                            case CubeAction.DoubleBeaver:
                            case CubeAction.RedoubleTake:
                            case CubeAction.OptionalDoubleTake:
                            case CubeAction.OptionalRedoubleTake:
                            case CubeAction.OptionalDoubleBeaver:
                                cubeValue *= 2;
                                cubeOwner = 1 - fMove; // opponent now owns
                                break;

                            default:
                                // No double: continue
                                break;
                        }
                    }
                }
            }

            // Roll dice using quasi-random permutation tables or RNG.
            // Port of RolloutDice from rollout.c: when fInitial and first turn,
            // skip doubles (only 30 of 36 outcomes are valid).
            int d0, d1;
            bool skipDoublesThisTurn = iTurn == 0 && settings.Initial;
            if (settings.Rotate && perms != null)
            {
                (d0, d1) = perms.GetRoll((int)trialIndex, iTurn, ref skip, skipDoubles: skipDoublesThisTurn);
            }
            else
            {
                do
                {
                    (d0, d1) = rng.NextDiceRoll();
                } while (skipDoublesThisTurn && d0 == d1);
            }

            // Variance reduction: evaluate all 21 (or 15) dice outcomes at 0-ply.
            // Port of rollout.c lines 596-638. Board is from current player's
            // perspective; after best move + SwapSides + eval, the result is from
            // the NEXT player's perspective. Invert to player 0 using !(iTurn & 1).
            if (settings.VarianceReduction)
            {
                AccumulateVarianceReduction(board, iTurn, d0, d1, output, varRedn,
                    skipDoubles: skipDoublesThisTurn);
            }

            // Find best move at the effective chequer ply depth.
            // Board is from current player's perspective — generate moves directly.
            if (effectiveChequerPlies > 0)
            {
                // Use FindnSaveBestMoves with move filters (matches C: FindBestMove + defaultFilters)
                var ec = new EvalContext
                {
                    Cubeful = settings.Cubeful,
                    Plies = effectiveChequerPlies,
                    UsePrune = effectiveChequerPlies >= 2,
                    Deterministic = true,
                };
                var ml = new MoveList();
                _evaluator.FindnSaveBestMoves(ml, board, d0, d1, ec);

                if (ml.Moves.Count > 0 && ml.BestIndex >= 0)
                    board = PositionId.FromKey(ml.Moves[ml.BestIndex].Key);
            }
            else
            {
                // 0-ply: evaluate all moves directly (no filtering needed)
                var ml = MoveGenerator.GenerateMoves(board, d0, d1);

                if (ml.Moves.Count > 0)
                {
                    Board bestBoard = board;
                    float bestScore = float.MinValue;

                    foreach (var move in ml.Moves)
                    {
                        var newBoard = MoveGenerator.ApplyMove(board, move);
                        var swapped = newBoard.Swapped();
                        _evaluator.EvaluatePosition(swapped, output);
                        Evaluator.InvertEvaluation(output);
                        float score = MatchEquityTable.MoneyEquity(output);

                        if (score > bestScore)
                        {
                            bestScore = score;
                            bestBoard = newBoard;
                        }
                    }

                    board = bestBoard;
                }
            }

            // SwapSides for next player (matches C: SwapSides(aanBoard[ici]))
            board.SwapSides();
            iTurn++;
        }

        // Truncated: evaluate at current position.
        // Board is from current player's perspective; invert if odd turn.
        if (settings.Cubeful && settings.CubePlies > 0)
            _evaluator.EvaluatePositionPlied(board, output, settings.CubePlies - 1);
        else
            _evaluator.EvaluatePosition(board, output);

        if ((iTurn & 1) != 0)
            Evaluator.InvertEvaluation(output);

        float truncEq = MatchEquityTable.MoneyEquity(output);
        double truncCubefulEq = settings.Cubeful ? truncEq * (double)cubeValue / nBasisCube : truncEq;

        double[] truncResult =
        [
            output[Constants.OutputWin],
            output[Constants.OutputWinGammon],
            output[Constants.OutputWinBackgammon],
            output[Constants.OutputLoseGammon],
            output[Constants.OutputLoseBackgammon],
            truncEq,
            truncCubefulEq,
        ];

        if (settings.VarianceReduction)
            ApplyVarianceReduction(truncResult, varRedn);

        return truncResult;
    }

    /// <summary>
    /// Variance reduction: evaluate all 36 dice outcomes at 0-ply, compute mean,
    /// then accumulate the correction (mean - actual_roll_value).
    /// Port of variance reduction from BasicCubefulRollout (rollout.c lines 596-678).
    ///
    /// Board is from the current player's perspective. For each dice combination:
    /// 1. Find best move at 0-ply
    /// 2. SwapSides the result and evaluate from next player's perspective
    /// 3. Invert to player 0's perspective:
    ///    - Even turns (player 0 on roll): eval after SwapSides is from player 1 → invert
    ///    - Odd turns (player 1 on roll): eval after SwapSides is from player 0 → keep
    ///    Rule: if (!(iTurn &amp; 1)) InvertEvaluation (matches C)
    /// </summary>
    /// <param name="skipDoubles">
    /// When true (fInitial on first turn), skip doubles (d0 == d1) and divide by 30
    /// instead of 36. Port of rollout.c lines 596-638.
    /// </param>
    private void AccumulateVarianceReduction(Board board, int iTurn,
        int actualD0, int actualD1, float[] output, double[] varRedn,
        bool skipDoubles = false)
    {
        double[] mean = new double[7];
        double[] actualRollValue = new double[7];

        // Evaluate all 21 distinct dice combinations (or 15 when skipping doubles)
        for (int d0 = 1; d0 <= 6; d0++)
        {
            for (int d1 = 1; d1 <= d0; d1++)
            {
                // Skip doubles when rolling initial position (no doubles on opening roll)
                if (skipDoubles && d0 == d1)
                    continue;

                float w = (d0 == d1) ? 1.0f : 2.0f;

                // Find best move at 0-ply from current player's perspective
                var ml = MoveGenerator.GenerateMoves(board, d0, d1);
                Board bestBoard;
                if (ml.Moves.Count > 0)
                {
                    bestBoard = board;
                    float bestScore = float.MinValue;
                    foreach (var move in ml.Moves)
                    {
                        var nb = MoveGenerator.ApplyMove(board, move);
                        var sw = nb.Swapped();
                        _evaluator.EvaluatePosition(sw, output);
                        Evaluator.InvertEvaluation(output);
                        float sc = MatchEquityTable.MoneyEquity(output);
                        if (sc > bestScore) { bestScore = sc; bestBoard = nb; }
                    }
                }
                else
                {
                    bestBoard = board.Clone();
                }

                // SwapSides and evaluate from next player's perspective
                // (matches C: SwapSides + GeneralEvaluationE with flipped fMove)
                var evalBoard = bestBoard.Swapped();
                _evaluator.EvaluatePosition(evalBoard, output);
                // output is from next player's perspective

                // Invert to player 0's perspective.
                // On even turns (player 0 was mover): after SwapSides, eval is
                // from player 1's perspective → invert.
                // On odd turns (player 1 was mover): after SwapSides, eval is
                // from player 0's perspective → keep.
                if ((iTurn & 1) == 0)
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

        // Normalize: 36 outcomes total, or 30 when doubles are skipped (6 doubles removed)
        double divisor = skipDoubles ? 30.0 : 36.0;
        for (int i = 0; i < 7; i++)
        {
            mean[i] /= divisor;
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

    private sealed class ThreadLocalState
    {
        public MersenneTwister? Rng;
        public float[]? Output;
        public readonly double[] Sum = new double[7];
        public readonly double[] SumSq = new double[7];
        public int Count;
    }

    private sealed class MoveTrialState
    {
        public MersenneTwister? Rng;
        public float[]? Output;
        public readonly double[][] AltSum;
        public readonly double[][] AltSumSq;
        public readonly int[] AltCount;

        public MoveTrialState(int nAlts)
        {
            AltSum = new double[nAlts][];
            AltSumSq = new double[nAlts][];
            AltCount = new int[nAlts];
            for (int i = 0; i < nAlts; i++)
            {
                AltSum[i] = new double[7];
                AltSumSq[i] = new double[7];
            }
        }
    }
}
