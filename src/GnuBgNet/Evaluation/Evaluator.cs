// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2019 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Ported from eval.c (EvaluatePosition, EvaluatePositionFull, EvaluatePositionCache, FindBestMoveInEval)

using GnuBgNet.Bearoff;
using GnuBgNet.Encoding;
using GnuBgNet.MatchEquity;
using GnuBgNet.MoveGeneration;
using GnuBgNet.NeuralNet;

namespace GnuBgNet.Evaluation;

/// <summary>
/// Core evaluation engine. Dispatches to neural nets, bearoff databases,
/// and performs n-ply search with eval caching.
/// Port of EvaluatePosition/EvaluatePositionFull/EvaluatePositionCache from eval.c.
/// </summary>
public sealed class Evaluator : IPositionEvaluator
{
    private const int MinPruneMoves = 5;

    private readonly NetworkSet _nets;
    private readonly IEvalCache _mainCache;
    private readonly IEvalCache _pruneCache;
    private readonly IMatchEquityTable _met;
    private readonly IMoveGenerator _moveGen;
    private readonly IInputCalculator _inputCalc;

    // Board is now a stack-allocated struct — no pooling needed.

    internal IBearoffDatabase? OneSidedBearoff { get; }
    internal IBearoffDatabase? TwoSidedBearoff { get; }
    internal IBearoffDatabase? HypergammonBearoff1 { get; }
    internal IBearoffDatabase? HypergammonBearoff2 { get; }
    internal IBearoffDatabase? HypergammonBearoff3 { get; }

    internal bool HasOneSidedBearoff => OneSidedBearoff != null;
    internal bool HasTwoSidedBearoff => TwoSidedBearoff != null;
    internal bool HasHypergammon1 => HypergammonBearoff1 != null;
    internal bool HasHypergammon2 => HypergammonBearoff2 != null;
    internal bool HasHypergammon3 => HypergammonBearoff3 != null;

    public Evaluator(NetworkSet nets, IBearoffDatabase? oneSidedBearoff = null,
        IBearoffDatabase? twoSidedBearoff = null,
        IBearoffDatabase? hypergammon1 = null,
        IBearoffDatabase? hypergammon2 = null,
        IBearoffDatabase? hypergammon3 = null,
        IMatchEquityTable? met = null,
        IEvalCache? mainCache = null,
        IEvalCache? pruneCache = null,
        IMoveGenerator? moveGenerator = null,
        IInputCalculator? inputCalculator = null)
    {
        _nets = nets;
        OneSidedBearoff = oneSidedBearoff;
        TwoSidedBearoff = twoSidedBearoff;
        HypergammonBearoff1 = hypergammon1;
        HypergammonBearoff2 = hypergammon2;
        HypergammonBearoff3 = hypergammon3;
        _met = met ?? MatchEquity.MatchEquityTable.ComputeDefault();
        _mainCache = mainCache ?? new EvalCache(Constants.CacheSizeMainLog2);
        _pruneCache = pruneCache ?? new EvalCache(Constants.CacheSizePruneLog2);
        _moveGen = moveGenerator ?? MoveGeneration.DefaultMoveGenerator.Instance;
        _inputCalc = inputCalculator ?? NeuralNet.DefaultInputCalculator.Instance;
    }

    /// <inheritdoc />
    public PositionClass ClassifyPosition(Board board) => Classifier.Classify(board, this);

    /// <summary>
    /// Evaluate a position at 0-ply (neural net or bearoff lookup).
    /// Port of EvaluatePosition() from eval.c.
    /// </summary>
    public void EvaluatePosition(Board board, Span<float> output)
    {
        var pc = Classifier.Classify(board, this);
        EvaluatePositionByClass(board, output, pc);
    }

    /// <summary>
    /// Evaluate a position at the specified number of plies, using the cache.
    /// Port of EvaluatePositionCache() from eval.c.
    /// </summary>
    public void EvaluatePositionPlied(Board board, Span<float> output, int nPlies, bool usePrune = true)
    {
        EvaluatePositionPlied(board, output, nPlies, usePrune, null, null);
    }

    /// <summary>
    /// Evaluate a position at the specified number of plies, with optional noise and cube context.
    /// Port of EvaluatePositionCache() from eval.c.
    /// When ec has noise, cache is bypassed (non-deterministic results cannot be cached).
    /// The cubeinfo is used for move scoring during search (gammon prices) and included in the cache key.
    /// </summary>
    public void EvaluatePositionPlied(Board board, Span<float> output, int nPlies, bool usePrune, EvalContext? ec,
        CubeInfo? ci = null)
    {
        bool hasNoise = ec != null && ec.Value.Noise != 0.0f;

        if (!hasNoise)
        {
            var key = PositionId.ToKey(board);
            int evalContext = EvalCache.ComputeEvalKey(nPlies, false, usePrune, ci, false);

            uint l = _mainCache.Lookup(key, evalContext, output);
            if (l == EvalCache.CacheHit)
                return;

            var pc = Classifier.Classify(board, this);
            EvaluatePositionFull(board, output, nPlies, pc, usePrune, ec, ci);

            _mainCache.Add(key, evalContext, output, l);
        }
        else
        {
            // Noisy evaluations cannot be cached
            var pc = Classifier.Classify(board, this);
            EvaluatePositionFull(board, output, nPlies, pc, usePrune, ec, ci);
        }
    }

    /// <summary>
    /// Full evaluation: leaf (0-ply) or recursive (n-ply).
    /// Port of EvaluatePositionFull() from eval.c.
    /// </summary>
    internal void EvaluatePositionFull(Board board, Span<float> output, int nPlies, PositionClass pc, bool usePrune,
        EvalContext? ec = null, CubeInfo? ci = null)
    {
        if (pc > PositionClass.BearoffTwoSided && nPlies > 0)
        {
            // Internal node: recurse over all 21 dice rolls
            for (int i = 0; i < Constants.NumOutputs; i++)
                output[i] = 0.0f;

            Span<float> variationOutput = stackalloc float[Constants.NumOutputs];

            // Create opponent cubeinfo once (same for all 21 rolls)
            CubeInfo? ciOpp = ci.HasValue ? new CubeInfo
            {
                Cube = ci.Value.Cube,
                CubeOwner = ci.Value.CubeOwner,
                Move = 1 - ci.Value.Move,
                MatchTo = ci.Value.MatchTo,
                Score0 = ci.Value.Score0, Score1 = ci.Value.Score1,
                Crawford = ci.Value.Crawford,
                Jacoby = ci.Value.Jacoby,
                Beavers = ci.Value.Beavers,
                Variation = ci.Value.Variation,
            } : null;

            for (int n0 = 1; n0 <= 6; n0++)
            {
                for (int n1 = 1; n1 <= n0; n1++)
                {
                    float w = (n0 == n1) ? 1.0f : 2.0f;

                    // Find best move for this roll
                    var newBoard = FindBestMoveForRoll(board, n0, n1, usePrune, ci);

                    // Swap sides and evaluate at n-1 ply
                    newBoard.SwapSides();

                    EvaluatePositionPlied(newBoard, variationOutput, nPlies - 1, usePrune, ec, ciOpp);

                    for (int i = 0; i < Constants.NumOutputs; i++)
                        output[i] += w * variationOutput[i];
                }
            }

            // Normalize by 36
            for (int i = 0; i < Constants.NumOutputs; i++)
                output[i] /= 36.0f;

            // Invert (we evaluated from opponent's perspective)
            InvertEvaluation(output);
        }
        else
        {
            // Leaf node: static evaluation
            EvaluatePositionByClass(board, output, pc);

            // Apply noise at leaf nodes (matching C: pec->rNoise check after acef[pc]())
            if (ec != null && ec.Value.Noise > 0.0f && pc != PositionClass.Over)
            {
                for (int i = 0; i < Constants.NumOutputs; i++)
                {
                    output[i] += Noise(ec.Value, board, i);
                    output[i] = Math.Clamp(output[i], 0.0f, 1.0f);
                }
            }
        }
    }

    /// <summary>
    /// Find the best move for a given dice roll during n-ply search.
    /// Uses pruning nets when possible for speed.
    /// Port of FindBestMoveInEval() from eval.c.
    /// </summary>
    private Board FindBestMoveForRoll(Board board, int n0, int n1, bool usePrune,
        CubeInfo? ci = null, bool cubeful = false)
    {
        var ml = _moveGen.GenerateMoves(board, n0, n1);

        if (ml.Moves.Count == 0)
        {
            _moveGen.ReturnMoves(ml);
            return board.Clone();
        }

        if (ml.Moves.Count == 1)
        {
            var key = ml.Moves[0].Key;
            _moveGen.ReturnMoves(ml);
            return PositionId.FromKey(key);
        }

        int pruneMoves = MinPruneMoves + FloorLog2(ml.Moves.Count);

        if (usePrune && ml.Moves.Count > pruneMoves)
        {
            if (TryScoreWithPruningNets(ml, board, ci))
            {
                var candidates = SelectTopMoves(ml, pruneMoves);
                var bestKey = FindBestKeyFromCandidates(candidates, ci, cubeful);
                _moveGen.ReturnMoves(ml);
                return PositionId.FromKey(bestKey);
            }
        }

        {
            var bestKey = FindBestKeyFromAll(ml, ci, cubeful);
            _moveGen.ReturnMoves(ml);
            return PositionId.FromKey(bestKey);
        }
    }

    /// <summary>
    /// Find and score best moves with progressive filtering.
    /// Port of FindnSaveBestMoves() from eval.c.
    /// Uses move filters to progressively prune candidates at each intermediate ply.
    /// </summary>
    public void FindnSaveBestMoves(MoveList ml, Board board, int nDice0, int nDice1,
        EvalContext ec, MoveFilter[,]? moveFilters = null)
    {
        _moveGen.GenerateMovesInto(ml, board, nDice0, nDice1);

        if (ml.Moves.Count == 0)
            return;

        moveFilters ??= MoveFilterPresets.Default;

        int filterRow = Math.Min(ec.Plies, MoveFilterPresets.MaxFilterPlies) - 1;
        if (filterRow < 0) filterRow = 0;

        for (int iPly = 0; iPly < ec.Plies; iPly++)
        {
            var filter = iPly < MoveFilterPresets.MaxFilterPlies
                ? moveFilters[filterRow, iPly]
                : MoveFilter.Null;

            if (filter.Accept < 0)
                continue;

            ScoreMoves(ml, iPly);
            SortMovesByScore(ml);

            // Apply filter: keep Accept + Extra moves within Threshold
            int keep = Math.Min(filter.Accept, ml.Moves.Count);
            int limit = Math.Min(ml.Moves.Count, keep + filter.Extra);

            for (int i = keep; i < limit; i++)
            {
                if (ml.Moves[i].Score < ml.Moves[0].Score - filter.Threshold)
                    break;
                keep = i + 1;
            }

            if (keep < ml.Moves.Count)
                ml.Moves.RemoveRange(keep, ml.Moves.Count - keep);

            if (ml.Moves.Count == 1 && filter.Accept != 1)
                break;
        }

        // Final evaluation at top ply
        ScoreMoves(ml, ec.Plies);
        SortMovesByScore(ml);
    }

    /// <summary>
    /// Score all moves in the list at the given ply.
    /// Port of ScoreMoves() from eval.c.
    /// </summary>
    internal void ScoreMoves(MoveList ml, int nPlies)
    {
        ml.BestScore = -99999.9f;

        for (int i = 0; i < ml.Moves.Count; i++)
        {
            ScoreMove(ml.Moves[i], nPlies);

            if (ml.Moves[i].Score > ml.BestScore ||
                (ml.Moves[i].Score == ml.BestScore &&
                 ml.Moves[i].Score2 > ml.Moves[ml.BestIndex].Score2))
            {
                ml.BestIndex = i;
                ml.BestScore = ml.Moves[i].Score;
            }
        }
    }

    /// <summary>
    /// Score a single move at the given ply (cubeless only).
    /// Port of ScoreMove() from eval.c.
    /// </summary>
    internal void ScoreMove(Move move, int nPlies)
    {
        ScoreMove(move, nPlies, null, false);
    }

    /// <summary>
    /// Score a single move at the given ply, optionally with cubeful equity.
    /// Port of ScoreMove() from eval.c.
    /// </summary>
    internal void ScoreMove(Move move, int nPlies, CubeInfo? ci, bool cubeful)
    {
        Span<float> arEval = stackalloc float[Constants.NumRolloutOutputs];
        var moveBoard = new Board();
        PositionId.FromKeySwappedInto(move.Key, ref moveBoard);

        if (cubeful && ci.HasValue)
        {
            var ciVal = ci.Value;
            // Create opponent-perspective cube info for after the move
            var ciOpp = new CubeInfo
            {
                Cube = ciVal.Cube,
                CubeOwner = ciVal.CubeOwner,
                Move = 1 - ciVal.Move,
                MatchTo = ciVal.MatchTo,
                Score0 = ciVal.Score0, Score1 = ciVal.Score1,
                Crawford = ciVal.Crawford,
                Jacoby = ciVal.Jacoby,
                Beavers = ciVal.Beavers,
                Variation = ciVal.Variation,
            };

            var ec = new EvalContext { Plies = nPlies, Cubeful = true, UsePrune = true };
            GeneralEvaluationEPlied(moveBoard, arEval, ciOpp, ec, nPlies);
            InvertEvaluationR(arEval, ciVal.MatchTo > 0);

            if (ciVal.MatchTo > 0)
            {
                arEval[Constants.OutputCubefulEquity] =
                    CubeDecision.Mwc2Eq(arEval[Constants.OutputCubefulEquity], ciVal, _met);
            }
        }
        else
        {
            if (nPlies == 0)
                EvaluatePosition(moveBoard, arEval);
            else
                EvaluatePositionPlied(moveBoard, arEval, nPlies);

            InvertEvaluation(arEval);

            arEval[Constants.OutputEquity] = ComputeEquity(arEval, ci);
            arEval[Constants.OutputCubefulEquity] = arEval[Constants.OutputEquity];
        }

        arEval.Slice(0, Constants.NumRolloutOutputs).CopyTo(move.EvalOutputs);

        move.MoveEvalSetup = new EvalSetup
        {
            Type = EvalType.Eval,
            Context = new EvalContext { Plies = nPlies, Cubeful = cubeful },
        };

        move.Score = cubeful
            ? arEval[Constants.OutputCubefulEquity]
            : arEval[Constants.OutputEquity];
        move.Score2 = arEval[Constants.OutputEquity];
    }

    /// <summary>Sort moves by score descending (best first).</summary>
    private static void SortMovesByScore(MoveList ml)
    {
        ml.Moves.Sort((a, b) =>
        {
            int cmp = b.Score.CompareTo(a.Score);
            return cmp != 0 ? cmp : b.Score2.CompareTo(a.Score2);
        });
        ml.BestIndex = 0;
    }

    /// <summary>
    /// Score all moves using pruning neural nets.
    /// Returns false if pruning isn't possible (e.g., mixed position classes).
    /// Port of FindBestMoveInEval() pruning path from eval.c.
    /// </summary>
    private bool TryScoreWithPruningNets(MoveList ml, Board board, CubeInfo? ci = null)
    {
        PositionClass evalClass = PositionClass.Over;
        Span<float> arOutput = stackalloc float[Constants.NumOutputs];

        // Create opponent-perspective cubeinfo for scoring, matching C's
        // pci->fMove = !pci->fMove flip in FindBestMoveInEval().
        // The pruning nets evaluate from opponent's perspective (Swapped board),
        // so we need opponent's gammon prices for correct UtilityME scoring.
        CubeInfo? ciOpp = ci.HasValue ? new CubeInfo
        {
            Cube = ci.Value.Cube,
            CubeOwner = ci.Value.CubeOwner,
            Move = 1 - ci.Value.Move,
            MatchTo = ci.Value.MatchTo,
            Score0 = ci.Value.Score0, Score1 = ci.Value.Score1,
            Crawford = ci.Value.Crawford,
            Jacoby = ci.Value.Jacoby,
            Beavers = ci.Value.Beavers,
            Variation = ci.Value.Variation,
        } : null;

        var pruneBoard = new Board();
        Span<float> pruneInputs = stackalloc float[Constants.NumPruningInputs];
        for (int i = 0; i < ml.Moves.Count; i++)
        {
            PositionId.FromKeySwappedInto(ml.Moves[i].Key, ref pruneBoard);
            var pc = Classifier.Classify(pruneBoard, this);

            if (i == 0)
            {
                if (pc < PositionClass.Race)
                    return false;
                evalClass = pc;
            }
            else if (pc != evalClass)
                return false;

            var key = ml.Moves[i].Key;
            uint l = _pruneCache.Lookup(key, 0, arOutput);
            if (l != EvalCache.CacheHit)
            {
                pruneInputs.Clear();
                _inputCalc.BaseInputs(pruneBoard, pruneInputs);

                var net = evalClass switch
                {
                    PositionClass.Race => _nets.PruneRace,
                    PositionClass.Crashed => _nets.PruneCrashed,
                    _ => _nets.PruneContact,
                };
                net.Evaluate(pruneInputs, arOutput);

                // Correct backgammon probabilities for race positions
                if (evalClass == PositionClass.Race)
                    EvalRaceBG(pruneBoard, arOutput);

                SanityCheck(pruneBoard, arOutput);

                _pruneCache.Add(key, 0, arOutput, l);
            }

            // Score from opponent's perspective using opponent's gammon prices,
            // negated so higher = better for us.
            // The C code (FindBestMoveInEval) flips fMove and keeps moves with the
            // LOWEST opponent utility; we negate so SelectTopMoves (highest) is correct.
            ml.Moves[i].Score = -ComputeEquity(arOutput, ciOpp);
        }

        return true;
    }

    /// <summary>
    /// Select the top N moves by score from a pruning pass.
    /// </summary>
    private static List<Move> SelectTopMoves(MoveList ml, int count)
    {
        var sorted = new List<Move>(ml.Moves);
        sorted.Sort((a, b) => b.Score.CompareTo(a.Score));
        return sorted.GetRange(0, Math.Min(count, sorted.Count));
    }

    /// <summary>
    /// Score a set of candidate moves at 0-ply and return the key for the best one.
    /// When cubeful=true, uses full cubeful evaluation matching C's ScoreMovesPruned.
    /// </summary>
    private PositionKey FindBestKeyFromCandidates(List<Move> candidates, CubeInfo? ci = null, bool cubeful = false)
    {
        if (cubeful && ci != null)
        {
            float bestScore = float.MinValue;
            int bestIdx = 0;
            for (int i = 0; i < candidates.Count; i++)
            {
                ScoreMove(candidates[i], 0, ci, true);
                if (candidates[i].Score > bestScore)
                {
                    bestScore = candidates[i].Score;
                    bestIdx = i;
                }
            }
            return candidates[bestIdx].Key;
        }

        {
            float bestScore = float.MinValue;
            int bestIdx = 0;
            Span<float> output = stackalloc float[Constants.NumOutputs];
            var scratchBoard = new Board();
            for (int i = 0; i < candidates.Count; i++)
            {
                PositionId.FromKeySwappedInto(candidates[i].Key, ref scratchBoard);
                EvaluatePosition(scratchBoard, output);
                InvertEvaluation(output);

                float score = ComputeEquity(output, ci);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestIdx = i;
                }
            }

            return candidates[bestIdx].Key;
        }
    }

    /// <summary>
    /// Score all moves at 0-ply and return the key for the best one.
    /// When cubeful=true, uses full cubeful evaluation matching C's ScoreMoves.
    /// </summary>
    private PositionKey FindBestKeyFromAll(MoveList ml, CubeInfo? ci = null, bool cubeful = false)
    {
        if (cubeful && ci != null)
        {
            float bestScore = float.MinValue;
            int bestIdx = 0;
            for (int i = 0; i < ml.Moves.Count; i++)
            {
                ScoreMove(ml.Moves[i], 0, ci, true);
                if (ml.Moves[i].Score > bestScore)
                {
                    bestScore = ml.Moves[i].Score;
                    bestIdx = i;
                }
            }
            return ml.Moves[bestIdx].Key;
        }

        {
            float bestScore = float.MinValue;
            int bestIdx = 0;
            Span<float> output = stackalloc float[Constants.NumOutputs];
            var scratchBoard = new Board();
            for (int i = 0; i < ml.Moves.Count; i++)
            {
                PositionId.FromKeySwappedInto(ml.Moves[i].Key, ref scratchBoard);
                EvaluatePosition(scratchBoard, output);
                InvertEvaluation(output);

                float score = ComputeEquity(output, ci);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestIdx = i;
                }
            }

            return ml.Moves[bestIdx].Key;
        }
    }

    /// <summary>
    /// Compute equity from output, using UtilityME for match play and MoneyEquity for money.
    /// Port of UtilityME() dispatch from eval.c.
    /// </summary>
    private float ComputeEquity(ReadOnlySpan<float> output, CubeInfo? ci)
    {
        if (ci.HasValue && ci.Value.MatchTo > 0)
        {
            var ciVal = ci.Value;
            return CubeDecision.UtilityMatch(output, ref ciVal, _met);
        }
        return MatchEquityTable.MoneyEquity(output);
    }

    private static int FloorLog2(int n)
    {
        int r = 0;
        while (n > 1) { n >>= 1; r++; }
        return r;
    }

    /// <summary>
    /// Evaluate given a known position class.
    /// Port of leaf evaluation in EvaluatePositionFull.
    /// </summary>
    public void EvaluatePositionByClass(Board board, Span<float> output, PositionClass pc)
    {
        switch (pc)
        {
            case PositionClass.Over:
                EvalOver(board, output);
                break;
            case PositionClass.Hypergammon1:
                HypergammonBearoff1!.Evaluate(board, output);
                break;
            case PositionClass.Hypergammon2:
                HypergammonBearoff2!.Evaluate(board, output);
                break;
            case PositionClass.Hypergammon3:
                HypergammonBearoff3!.Evaluate(board, output);
                break;
            case PositionClass.BearoffTwoSided:
                TwoSidedBearoff!.Evaluate(board, output);
                break;
            case PositionClass.BearoffOneSided:
                OneSidedBearoff!.Evaluate(board, output);
                break;
            case PositionClass.Race:
                EvalRace(board, output);
                break;
            case PositionClass.Crashed:
                EvalCrashed(board, output);
                break;
            case PositionClass.Contact:
                EvalContact(board, output);
                break;
            default:
                EvalContact(board, output);
                break;
        }

        if (pc > PositionClass.Over)
            SanityCheck(board, output);
    }

    private void EvalRace(Board board, Span<float> output)
    {
        Span<float> inputs = stackalloc float[Constants.NumRaceInputs];
        _inputCalc.CalculateRaceInputs(board, inputs);
        _nets.Race.Evaluate(inputs, output);

        // Special evaluation of backgammons overrides net output
        EvalRaceBG(board, output);
    }

    /// <summary>
    /// Override backgammon probabilities in race evaluation.
    /// Port of EvalRaceBG() from eval.c.
    /// board.Player (anBoard[1]) is on roll.
    /// </summary>
    private void EvalRaceBG(Board board, Span<float> output)
    {
        const int G_POSSIBLE = 0x1;
        const int BG_POSSIBLE = 0x2;
        const int OG_POSSIBLE = 0x4;
        const int OBG_POSSIBLE = 0x8;

        int totMen0 = 0; // opponent (not on roll)
        int totMen1 = 0; // player (on roll)

        for (int i = 23; i >= 0; --i)
        {
            totMen0 += (int)board.Opponent[i];
            totMen1 += (int)board.Player[i];
        }

        int any = 0;
        if (totMen1 == 15) any |= OG_POSSIBLE;
        if (totMen0 == 15) any |= G_POSSIBLE;

        if (any == 0) return;

        if ((any & OG_POSSIBLE) != 0)
        {
            for (int i = 23; i >= 18; --i)
            {
                if (board.Player[i] > 0)
                {
                    any |= OBG_POSSIBLE;
                    break;
                }
            }
        }

        if ((any & G_POSSIBLE) != 0)
        {
            for (int i = 23; i >= 18; --i)
            {
                if (board.Opponent[i] > 0)
                {
                    any |= BG_POSSIBLE;
                    break;
                }
            }
        }

        if ((any & (BG_POSSIBLE | OBG_POSSIBLE)) == 0) return;

        // side that can have the backgammon: 1 = player wins BG, 0 = opponent wins BG
        int side = (any & BG_POSSIBLE) != 0 ? 1 : 0;

        float pr = RaceBGprob(board, side);

        if (pr > 0.0f)
        {
            if (side == 1)
            {
                output[Constants.OutputWinBackgammon] = pr;
                if (output[Constants.OutputWinGammon] < output[Constants.OutputWinBackgammon])
                    output[Constants.OutputWinGammon] = output[Constants.OutputWinBackgammon];
            }
            else
            {
                output[Constants.OutputLoseBackgammon] = pr;
                if (output[Constants.OutputLoseGammon] < output[Constants.OutputLoseBackgammon])
                    output[Constants.OutputLoseGammon] = output[Constants.OutputLoseBackgammon];
            }
        }
        else
        {
            if (side == 1)
                output[Constants.OutputWinBackgammon] = 0.0f;
            else
                output[Constants.OutputLoseBackgammon] = 0.0f;
        }
    }

    /// <summary>
    /// Compute probability of backgammon for given side.
    /// Port of raceBGprob() from eval.c.
    /// Uses bearoff evaluation fallback (not the precomputed lookup tables from bearoffgammon.c).
    /// </summary>
    private float RaceBGprob(Board board, int side)
    {
        // side: 1 = player (anBoard[1]), 0 = opponent (anBoard[0])
        ReadOnlySpan<uint> sideBoard = side == 1 ? board.Player.AsReadOnlySpan() : board.Opponent.AsReadOnlySpan();
        ReadOnlySpan<uint> oppBoard = side == 1 ? board.Opponent.AsReadOnlySpan() : board.Player.AsReadOnlySpan();

        int totMenHome = 0;
        for (int i = 0; i < 6; ++i)
            totMenHome += (int)sideBoard[i];

        int totPipsOp = 0;
        for (int i = 22; i >= 18; --i)
            totPipsOp += (int)oppBoard[i] * (i - 17);

        // Quick check: can the bearing-off side finish before opponent escapes?
        if (!((totMenHome + 3) / 4 - (side == 1 ? 1 : 0) <= (totPipsOp + 2) / 3))
            return 0.0f;

        // Create dummy board: keep side's pieces, map opponent's last 6 points to first 6
        // In C: dummy[side] = anBoard[side], dummy[1-side][0..5] = anBoard[1-side][18..23], rest = 0
        var dummy = new Board();
        Span<uint> dummyPlayer = side == 1 ? dummy.Player.AsSpan() : dummy.Opponent.AsSpan();
        Span<uint> dummyOpp = side == 1 ? dummy.Opponent.AsSpan() : dummy.Player.AsSpan();

        sideBoard.CopyTo(dummyPlayer);
        for (int i = 0; i < 6; ++i)
            dummyOpp[i] = oppBoard[18 + i];

        // Evaluate using bearoff
        float result = 0.0f;
        if (TwoSidedBearoff != null)
        {
            uint pos0 = PositionId.PositionBearoff(dummy.Opponent.AsReadOnlySpan(), TwoSidedBearoff.Points, TwoSidedBearoff.Chequers);
            uint pos1 = PositionId.PositionBearoff(dummy.Player.AsReadOnlySpan(), TwoSidedBearoff.Points, TwoSidedBearoff.Chequers);
            if (pos0 < TwoSidedBearoff.NumPositions && pos1 < TwoSidedBearoff.NumPositions)
            {
                Span<float> ar = stackalloc float[Constants.NumOutputs];
                TwoSidedBearoff.Evaluate(dummy, ar);
                float p = side == 1 ? ar[Constants.OutputWin] : 1.0f - ar[Constants.OutputWin];
                result = Math.Min(p, 1.0f);
                return result;
            }
        }

        if (OneSidedBearoff != null)
        {
            Span<float> ar = stackalloc float[Constants.NumOutputs];
            OneSidedBearoff.Evaluate(dummy, ar);
            float p = side == 1 ? ar[Constants.OutputWin] : 1.0f - ar[Constants.OutputWin];
            result = Math.Min(p, 1.0f);
        }

        return result;
    }

    private void EvalContact(Board board, Span<float> output)
    {
        Span<float> inputs = stackalloc float[Constants.NumContactInputs];
        _inputCalc.CalculateContactInputs(board, inputs);
        _nets.Contact.Evaluate(inputs, output);
    }

    private void EvalCrashed(Board board, Span<float> output)
    {
        Span<float> inputs = stackalloc float[Constants.NumCrashedInputs];
        _inputCalc.CalculateCrashedInputs(board, inputs);
        _nets.Crashed.Evaluate(inputs, output);
    }

    /// <summary>
    /// Evaluate a completed game (one side has all pieces off).
    /// Port of EvalOver() from eval.c.
    /// </summary>
    private static void EvalOver(Board board, Span<float> output,
        BackgammonVariation variation = BackgammonVariation.Standard)
    {
        int n = variation switch
        {
            BackgammonVariation.Hypergammon1 => 1,
            BackgammonVariation.Hypergammon2 => 2,
            BackgammonVariation.Hypergammon3 => 3,
            _ => 15,
        };

        // Check if opponent has any pieces
        bool oppHasPieces = false;
        for (int i = 0; i < 25; i++)
            if (board.Opponent[i] > 0) { oppHasPieces = true; break; }

        if (!oppHasPieces)
        {
            // Opponent has no pieces: player has lost
            output[Constants.OutputWin] = 0.0f;
            output[Constants.OutputWinGammon] = 0.0f;
            output[Constants.OutputWinBackgammon] = 0.0f;

            int c = 0;
            for (int i = 0; i < 25; i++) c += (int)board.Player[i];

            if (c == n)
            {
                // Player still has all pieces: loses gammon
                output[Constants.OutputLoseGammon] = 1.0f;
                // Check for backgammon
                for (int i = 18; i < 25; i++)
                {
                    if (board.Player[i] > 0)
                    {
                        output[Constants.OutputLoseBackgammon] = 1.0f;
                        return;
                    }
                }
                output[Constants.OutputLoseBackgammon] = 0.0f;
                return;
            }
            output[Constants.OutputLoseGammon] = 0.0f;
            output[Constants.OutputLoseBackgammon] = 0.0f;
            return;
        }

        // Check if player has any pieces
        bool playerHasPieces = false;
        for (int i = 0; i < 25; i++)
            if (board.Player[i] > 0) { playerHasPieces = true; break; }

        if (!playerHasPieces)
        {
            // Player has no pieces: player wins
            output[Constants.OutputWin] = 1.0f;
            output[Constants.OutputLoseGammon] = 0.0f;
            output[Constants.OutputLoseBackgammon] = 0.0f;

            int c = 0;
            for (int i = 0; i < 25; i++) c += (int)board.Opponent[i];

            if (c == n)
            {
                output[Constants.OutputWinGammon] = 1.0f;
                for (int i = 18; i < 25; i++)
                {
                    if (board.Opponent[i] > 0)
                    {
                        output[Constants.OutputWinBackgammon] = 1.0f;
                        return;
                    }
                }
                output[Constants.OutputWinBackgammon] = 0.0f;
                return;
            }
            output[Constants.OutputWinGammon] = 0.0f;
            output[Constants.OutputWinBackgammon] = 0.0f;
            return;
        }

        // Both sides have pieces - shouldn't be CLASS_OVER, but handle gracefully
        output[Constants.OutputWin] = 0.5f;
        output[Constants.OutputWinGammon] = 0.0f;
        output[Constants.OutputWinBackgammon] = 0.0f;
        output[Constants.OutputLoseGammon] = 0.0f;
        output[Constants.OutputLoseBackgammon] = 0.0f;
    }

    /// <summary>
    /// Post-evaluation sanity check and normalization.
    /// Port of SanityCheck() from eval.c.
    /// </summary>
    internal void SanityCheck(Board board, Span<float> output)
    {
        // Clamp all outputs to [0, 1]
        for (int i = 0; i < Constants.NumOutputs; i++)
            output[i] = Math.Clamp(output[i], 0.0f, 1.0f);

        Span<int> ac = stackalloc int[2];
        Span<int> anBack = stackalloc int[2];
        Span<int> anCross = stackalloc int[2];
        Span<int> anGammonCross = stackalloc int[] { 1, 1 };

        for (int side = 0; side < 2; side++)
        {
            ReadOnlySpan<uint> b = side == 0 ? board.Opponent.AsReadOnlySpan() : board.Player.AsReadOnlySpan();

            for (int i = 0; i < 6; i++)
            {
                if (b[i] > 0) { anBack[side] = i; ac[side] += (int)b[i]; anCross[side] += (int)b[i]; }
            }
            int nciq = 0;
            for (int i = 6; i < 12; i++)
            {
                if (b[i] > 0) { anBack[side] = i; nciq += (int)b[i]; }
            }
            ac[side] += nciq; anCross[side] += 2 * nciq; anGammonCross[side] += nciq;

            nciq = 0;
            for (int i = 12; i < 18; i++)
            {
                if (b[i] > 0) { anBack[side] = i; nciq += (int)b[i]; }
            }
            ac[side] += nciq; anCross[side] += 3 * nciq; anGammonCross[side] += 2 * nciq;

            nciq = 0;
            for (int i = 18; i < 24; i++)
            {
                if (b[i] > 0) { anBack[side] = i; nciq += (int)b[i]; }
            }
            ac[side] += nciq; anCross[side] += 4 * nciq; anGammonCross[side] += 3 * nciq;

            if (b[24] > 0)
            {
                anBack[side] = 24;
                ac[side] += (int)b[24];
                anCross[side] += 5 * (int)b[24];
                anGammonCross[side] += 4 * (int)b[24];
            }
        }

        bool fContact = anBack[0] + anBack[1] >= 24;

        Span<int> anMaxTurns = stackalloc int[2];
        if (!fContact)
        {
            for (int i = 0; i < 2; i++)
            {
                ReadOnlySpan<uint> b = i == 0 ? board.Opponent.AsReadOnlySpan() : board.Player.AsReadOnlySpan();
                if (anBack[i] < 6 && OneSidedBearoff != null)
                    anMaxTurns[i] = MaxTurns(
                        PositionId.PositionBearoff(b, OneSidedBearoff.Points, OneSidedBearoff.Chequers));
                else
                    anMaxTurns[i] = anCross[i] * 2;
            }

            if (anMaxTurns[1] == 0)
                anMaxTurns[1] = 1;
        }

        if (!fContact && anCross[0] > 4 * (anMaxTurns[1] - 1))
            output[Constants.OutputWin] = 1.0f;

        if (ac[0] < 15)
            output[Constants.OutputWinGammon] = output[Constants.OutputWinBackgammon] = 0.0f;
        else if (!fContact)
        {
            if (anCross[1] > 8 * anGammonCross[0])
                output[Constants.OutputWinGammon] = 0.0f;
            else if (anGammonCross[0] > 4 * (anMaxTurns[1] - 1))
                output[Constants.OutputWinGammon] = 1.0f;
            if (anBack[0] < 18)
                output[Constants.OutputWinBackgammon] = 0.0f;
        }

        if (!fContact && anCross[1] > 4 * anMaxTurns[0])
            output[Constants.OutputWin] = 0.0f;

        if (ac[1] < 15)
            output[Constants.OutputLoseGammon] = output[Constants.OutputLoseBackgammon] = 0.0f;
        else if (!fContact)
        {
            if (anCross[0] > 8 * anGammonCross[1] - 4)
                output[Constants.OutputLoseGammon] = 0.0f;
            else if (anGammonCross[1] > 4 * anMaxTurns[0])
                output[Constants.OutputLoseGammon] = 1.0f;
            if (anBack[1] < 18)
                output[Constants.OutputLoseBackgammon] = 0.0f;
        }

        // Gammons must be <= wins
        if (output[Constants.OutputWinGammon] > output[Constants.OutputWin])
            output[Constants.OutputWinGammon] = output[Constants.OutputWin];

        float lose = 1.0f - output[Constants.OutputWin];
        if (output[Constants.OutputLoseGammon] > lose)
            output[Constants.OutputLoseGammon] = lose;

        // Backgammons <= gammons
        if (output[Constants.OutputWinBackgammon] > output[Constants.OutputWinGammon])
            output[Constants.OutputWinBackgammon] = output[Constants.OutputWinGammon];
        if (output[Constants.OutputLoseBackgammon] > output[Constants.OutputLoseGammon])
            output[Constants.OutputLoseBackgammon] = output[Constants.OutputLoseGammon];

        // Eliminate tiny values in contact
        if (fContact)
        {
            const float noise = 1 / 10000.0f;
            for (int i = Constants.OutputWinGammon; i < Constants.NumOutputs; ++i)
                if (output[i] < noise) output[i] = 0.0f;
        }
    }

    /// <summary>
    /// Upper bound on turns to complete bearoff from a one-sided position.
    /// Port of MaxTurns() from eval.c.
    /// </summary>
    private int MaxTurns(uint posId)
    {
        if (OneSidedBearoff == null)
            return 0;

        Span<float> probs = stackalloc float[32];
        OneSidedBearoff.GetDistribution(posId, probs);

        for (int i = 31; i >= 0; i--)
        {
            if (probs[i] > 0.0f)
                return i;
        }
        return 0;
    }

    /// <summary>
    /// Invert evaluation (swap perspective) — 5 outputs.
    /// Port of InvertEvaluation() from eval.c.
    /// </summary>
    public static void InvertEvaluation(Span<float> ar)
    {
        ar[Constants.OutputWin] = 1.0f - ar[Constants.OutputWin];
        (ar[Constants.OutputWinGammon], ar[Constants.OutputLoseGammon]) =
            (ar[Constants.OutputLoseGammon], ar[Constants.OutputWinGammon]);
        (ar[Constants.OutputWinBackgammon], ar[Constants.OutputLoseBackgammon]) =
            (ar[Constants.OutputLoseBackgammon], ar[Constants.OutputWinBackgammon]);
    }

    /// <summary>
    /// Invert evaluation including equity outputs — 7 outputs (NUM_ROLLOUT_OUTPUTS).
    /// Port of InvertEvaluationR() from eval.c.
    /// </summary>
    public static void InvertEvaluationR(Span<float> ar, bool isMatchPlay)
    {
        InvertEvaluation(ar);

        ar[Constants.OutputEquity] = -ar[Constants.OutputEquity];

        if (isMatchPlay)
            ar[Constants.OutputCubefulEquity] = 1.0f - ar[Constants.OutputCubefulEquity];
        else
            ar[Constants.OutputCubefulEquity] = -ar[Constants.OutputCubefulEquity];
    }

    /// <summary>
    /// Check if the game is over and return the result.
    /// Returns 0 = not over, 1 = normal win, 2 = gammon, 3 = backgammon.
    /// Port of GameStatus() from eval.c.
    /// </summary>
    public int GameStatus(Board board)
    {
        var pc = Classifier.Classify(board, this);
        if (pc != PositionClass.Over)
            return 0;

        Span<float> ar = stackalloc float[Constants.NumOutputs];
        EvalOver(board, ar);

        if (ar[Constants.OutputWinBackgammon] > 0.0f || ar[Constants.OutputLoseBackgammon] > 0.0f)
            return 3;
        if (ar[Constants.OutputWinGammon] > 0.0f || ar[Constants.OutputLoseGammon] > 0.0f)
            return 2;

        return 1;
    }

    /// <summary>
    /// General evaluation entry point: dispatches to cubeful or cubeless evaluation.
    /// Port of GeneralEvaluationEPlied() from eval.c.
    /// Returns 7-output result (5 probs + cubeless equity + cubeful equity).
    /// </summary>
    public void GeneralEvaluationEPlied(
        Board board, Span<float> arOutput, CubeInfo ci, EvalContext ec, int nPlies)
    {
        if (ec.Cubeful)
        {
            float rCubeful = 0;
            CubeInfo[] aciSpan = [ci];
            EvaluatePositionCubeful3(board, arOutput, ref rCubeful,
                aciSpan, 1, ci, ec, nPlies, false);

            arOutput[Constants.OutputEquity] = ci.MatchTo > 0
                ? CubeDecision.UtilityMatch(arOutput, ref ci, _met)
                : MatchEquityTable.MoneyEquity(arOutput);
            arOutput[Constants.OutputCubefulEquity] = rCubeful;
        }
        else
        {
            EvaluatePositionPlied(board, arOutput, nPlies, ec.UsePrune, ec, ci);
            arOutput[Constants.OutputEquity] = ci.MatchTo > 0
                ? CubeDecision.UtilityMatch(arOutput, ref ci, _met)
                : MatchEquityTable.MoneyEquity(arOutput);
            arOutput[Constants.OutputCubefulEquity] = 0.0f;
        }
    }

    /// <summary>
    /// Cubeful evaluation with caching.
    /// Port of EvaluatePositionCubeful3() from eval.c.
    /// </summary>
    internal void EvaluatePositionCubeful3(
        Board board, Span<float> arOutput, ref float cubeful,
        ReadOnlySpan<CubeInfo> aciCubePos, int cci, CubeInfo pciMove,
        EvalContext ec, int nPlies, bool fTop)
    {
        // For simplicity, skip cache for now and go directly to evaluation
        Span<float> arCubeful = stackalloc float[cci];
        EvaluatePositionCubeful4(board, arOutput, arCubeful,
            aciCubePos, cci, pciMove, ec, nPlies, fTop);
        cubeful = arCubeful[0];
    }

    /// <summary>
    /// Cubeful evaluation with caching (multi-cube-position version).
    /// </summary>
    internal void EvaluatePositionCubeful3Multi(
        Board board, Span<float> arOutput, Span<float> arCubeful,
        ReadOnlySpan<CubeInfo> aciCubePos, int cci, CubeInfo pciMove,
        EvalContext ec, int nPlies, bool fTop)
    {
        // Check cache for each cube position
        var key = PositionId.ToKey(board);
        bool allCached = !fTop;

        if (ec.Noise == 0.0f && allCached)
        {
            Span<float> cachedOutput = stackalloc float[Constants.NumOutputs + 1];
            for (int ici = 0; ici < cci && allCached; ici++)
            {
                if (aciCubePos[ici].Cube < 0) { arCubeful[ici] = -99999.9f; continue; }
                int evalKey = EvalCache.ComputeEvalKey(nPlies, ec.Cubeful,
                    ec.UsePrune, aciCubePos[ici], true);
                uint l = _mainCache.Lookup(key, evalKey, cachedOutput);
                if (l != EvalCache.CacheHit) { allCached = false; break; }
                for (int j = 0; j < Constants.NumOutputs; j++) arOutput[j] = cachedOutput[j];
                arCubeful[ici] = cachedOutput[Constants.NumOutputs];
            }
        }
        else
        {
            allCached = false;
        }

        if (!allCached)
        {
            EvaluatePositionCubeful4(board, arOutput, arCubeful,
                aciCubePos, cci, pciMove, ec, nPlies, fTop);

            // Add to cache
            if (!fTop && ec.Noise == 0.0f)
            {
                float[] cacheEntry = new float[Constants.NumOutputs + 1];
                for (int j = 0; j < Constants.NumOutputs; j++) cacheEntry[j] = arOutput[j];
                for (int ici = 0; ici < cci; ici++)
                {
                    if (aciCubePos[ici].Cube < 0) continue;
                    cacheEntry[Constants.NumOutputs] = arCubeful[ici];
                    int evalKey = EvalCache.ComputeEvalKey(nPlies, ec.Cubeful,
                        ec.UsePrune, aciCubePos[ici], true);
                    // Cache storage simplified — store 5 outputs + 1 cubeful
                    _mainCache.Add(key, evalKey, cacheEntry, 0);
                }
            }
        }
    }

    /// <summary>
    /// Core cubeful evaluation engine.
    /// Port of EvaluatePositionCubeful4() from eval.c.
    /// </summary>
    internal void EvaluatePositionCubeful4(
        Board board, Span<float> arOutput, Span<float> arCubeful,
        ReadOnlySpan<CubeInfo> aciCubePos, int cci, CubeInfo pciMove,
        EvalContext ec, int nPlies, bool fTop)
    {
        var pc = Classifier.Classify(board, this);
        Span<float> arCf = stackalloc float[2 * cci];
        CubeInfo[] aci = new CubeInfo[2 * cci];

        if (pc > PositionClass.BearoffTwoSided && nPlies > 0)
        {
            // Internal node: recurse over all 21 dice rolls
            Span<float> ar = stackalloc float[Constants.NumOutputs];
            Span<float> arCfTemp = stackalloc float[2 * cci];

            for (int i = 0; i < Constants.NumOutputs; i++)
                arOutput[i] = 0.0f;
            for (int i = 0; i < 2 * cci; i++)
                arCf[i] = 0.0f;

            // Build next level cube positions
            MakeCubePos(aciCubePos, cci, fTop, aci, true);

            bool usePrune = ec.UsePrune && ec.Noise == 0.0f;

            // Create opponent cube info once (same for all 21 rolls)
            var ciMoveOpp = new CubeInfo
            {
                Cube = pciMove.Cube,
                CubeOwner = pciMove.CubeOwner,
                Move = 1 - pciMove.Move,
                MatchTo = pciMove.MatchTo,
                Score0 = pciMove.Score0, Score1 = pciMove.Score1,
                Crawford = pciMove.Crawford,
                Jacoby = pciMove.Jacoby,
                Beavers = pciMove.Beavers,
                Variation = pciMove.Variation,
            };

            for (int n0 = 1; n0 <= 6; n0++)
            {
                for (int n1 = 1; n1 <= n0; n1++)
                {
                    float w = (n0 == n1) ? 1.0f : 2.0f;

                    // Find best move with cubeful scoring
                    var newBoard = FindBestMoveForRoll(board, n0, n1, usePrune, pciMove, cubeful: true);
                    newBoard.SwapSides();

                    // Evaluate recursively
                    EvaluatePositionCubeful4(newBoard, ar, arCfTemp,
                        aci, 2 * cci, ciMoveOpp, ec, nPlies - 1, false);

                    for (int i = 0; i < Constants.NumOutputs; i++)
                        arOutput[i] += w * ar[i];
                    for (int i = 0; i < 2 * cci; i++)
                        arCf[i] += w * arCfTemp[i];
                }
            }

            // Flip evaluations (divide by 36 and invert)
            arOutput[Constants.OutputWin] = 1.0f - arOutput[Constants.OutputWin] / 36.0f;
            float rTemp = arOutput[Constants.OutputWinGammon] / 36.0f;
            arOutput[Constants.OutputWinGammon] = arOutput[Constants.OutputLoseGammon] / 36.0f;
            arOutput[Constants.OutputLoseGammon] = rTemp;
            rTemp = arOutput[Constants.OutputWinBackgammon] / 36.0f;
            arOutput[Constants.OutputWinBackgammon] = arOutput[Constants.OutputLoseBackgammon] / 36.0f;
            arOutput[Constants.OutputLoseBackgammon] = rTemp;

            for (int i = 0; i < 2 * cci; i++)
            {
                if (pciMove.MatchTo > 0)
                    arCf[i] = 1.0f - arCf[i] / 36.0f;
                else
                    arCf[i] = -arCf[i] / 36.0f;
            }

            // Invert fMove on the cube positions
            for (int i = 0; i < 2 * cci; i++)
                aci[i].Move = 1 - aci[i].Move;

            // Get cubeful equities
            GetEcf3(arCubeful, cci, arCf, aci, _met);
        }
        else
        {
            // Leaf node: static evaluation
            EvaluatePositionByClass(board, arOutput, pc);

            if (ec.Noise > 0.0f && pc != PositionClass.Over)
            {
                for (int i = 0; i < Constants.NumOutputs; i++)
                {
                    arOutput[i] += Noise(ec, board, i);
                    arOutput[i] = Math.Clamp(arOutput[i], 0.0f, 1.0f);
                }
            }

            if (pc > PositionClass.Over)
                SanityCheck(board, arOutput);

            float rCubeX = CubeEfficiency.Compute(board, pc, ec.Plies);

            // Build cube positions for leaf
            MakeCubePos(aciCubePos, cci, fTop, aci, false);

            // Check for exact cubeful bearoff equities
            bool usedPerfectCubeful = false;
            Span<float> arEquity = stackalloc float[4];
            if (pc == PositionClass.BearoffTwoSided && TwoSidedBearoff is { Cubeful: true })
            {
                TwoSidedBearoff.GetCubefulEquities(board, arEquity);
                usedPerfectCubeful = true;
            }

            // Calculate cubeful equity for each cube position
            for (int ici = 0; ici < 2 * cci; ici++)
            {
                if (aci[ici].Cube <= 0)
                    continue;

                if (aci[ici].MatchTo == 0 && usedPerfectCubeful)
                {
                    // Money play with exact cubeful equities from bearoff DB
                    // arEquity: [0]=cubeless, [1]=owned, [2]=centered, [3]=opponent
                    if (aci[ici].CubeOwner == -1)
                        arCf[ici] = arEquity[2]; // centered
                    else if (aci[ici].CubeOwner == aci[ici].Move)
                        arCf[ici] = arEquity[1]; // owned
                    else
                        arCf[ici] = arEquity[3]; // opponent
                }
                else if (aci[ici].MatchTo > 0 && usedPerfectCubeful)
                {
                    // Match play bearoff: derive cube efficiency from exact money cubeful
                    // then use Cl2CfMatch with that derived efficiency
                    float rCl = arEquity[0]; // exact cubeless
                    float rCfMoney = aci[ici].CubeOwner == -1 ? arEquity[2]
                        : aci[ici].CubeOwner == aci[ici].Move ? arEquity[1]
                        : arEquity[3];
                    // Derive cube efficiency: rCubeX = (rCfMoney - rCl) / (rCf_janowski - rCl)
                    float rCfJanowski = CubeDecision.CubelessToCubefulMoney(
                        arOutput, aci[ici].CubeOwner, aci[ici].Jacoby, 1.0f, aci[ici].Move);
                    float denom = rCfJanowski - rCl;
                    float derivedCubeX = Math.Abs(denom) > 1e-6f
                        ? Math.Clamp((rCfMoney - rCl) / denom, 0.0f, 1.0f)
                        : rCubeX;
                    arCf[ici] = CubeDecision.Cl2CfMatch(arOutput, aci[ici], _met, derivedCubeX);
                }
                else if (aci[ici].MatchTo == 0)
                {
                    // Money play
                    arCf[ici] = CubeDecision.CubelessToCubefulMoney(
                        arOutput, aci[ici].CubeOwner, aci[ici].Jacoby, rCubeX, aci[ici].Move);
                }
                else
                {
                    // Match play
                    arCf[ici] = CubeDecision.Cl2CfMatch(arOutput, aci[ici], _met, rCubeX);
                }
            }

            GetEcf3(arCubeful, cci, arCf, aci, _met);
        }
    }

    /// <summary>
    /// Build cube positions for recursive cubeful evaluation.
    /// Port of MakeCubePos() from eval.c.
    /// For each input cube position, creates two output positions:
    /// [i*2] = no-double (same cube), [i*2+1] = double (2x cube, opponent owns).
    /// </summary>
    internal static void MakeCubePos(ReadOnlySpan<CubeInfo> aciCubePos, int cci, bool fTop,
        Span<CubeInfo> aci, bool fInvert)
    {
        int idx = 0;
        for (int ici = 0; ici < cci; ici++)
        {
            // No double position
            if (aciCubePos[ici].Cube > 0)
            {
                aci[idx] = new CubeInfo
                {
                    Cube = aciCubePos[ici].Cube,
                    CubeOwner = aciCubePos[ici].CubeOwner,
                    Move = fInvert ? 1 - aciCubePos[ici].Move : aciCubePos[ici].Move,
                    MatchTo = aciCubePos[ici].MatchTo,
                    Score0 = aciCubePos[ici].Score0, Score1 = aciCubePos[ici].Score1,
                    Crawford = aciCubePos[ici].Crawford,
                    Jacoby = aciCubePos[ici].Jacoby,
                    Beavers = aciCubePos[ici].Beavers,
                    Variation = aciCubePos[ici].Variation,
                };
            }
            else
            {
                aci[idx] = new CubeInfo { Cube = -1 };
            }
            idx++;

            // Double position (opponent takes the cube)
            if (!fTop && aciCubePos[ici].Cube > 0 &&
                CubeDecision.GetDPEq(aciCubePos[ici], null, out _))
            {
                aci[idx] = new CubeInfo
                {
                    Cube = 2 * aciCubePos[ici].Cube,
                    CubeOwner = 1 - aciCubePos[ici].Move,
                    Move = fInvert ? 1 - aciCubePos[ici].Move : aciCubePos[ici].Move,
                    MatchTo = aciCubePos[ici].MatchTo,
                    Score0 = aciCubePos[ici].Score0, Score1 = aciCubePos[ici].Score1,
                    Crawford = aciCubePos[ici].Crawford,
                    Jacoby = aciCubePos[ici].Jacoby,
                    Beavers = aciCubePos[ici].Beavers,
                    Variation = aciCubePos[ici].Variation,
                };
            }
            else
            {
                aci[idx] = new CubeInfo { Cube = -1 };
            }
            idx++;
        }
    }

    /// <summary>
    /// Select best cube action from recursive cubeful equity results.
    /// Port of GetECF3() from eval.c.
    /// </summary>
    internal static void GetEcf3(Span<float> arCubeful, int cci, Span<float> arCf, ReadOnlySpan<CubeInfo> aci,
        IMatchEquityTable met)
    {
        for (int ici = 0, i = 0; ici < cci; ici++, i += 2)
        {
            if (aci[i + 1].Cube > 0)
            {
                // Cube available
                float rND = arCf[i];
                float rDT = aci[0].MatchTo > 0 ? arCf[i + 1] : 2.0f * arCf[i + 1];
                CubeDecision.GetDPEq(aci[i], met, out float rDP);

                if (rDT >= rND && rDP >= rND)
                {
                    // Double
                    arCubeful[ici] = rDT >= rDP ? rDP : rDT;
                }
                else
                {
                    // No double
                    arCubeful[ici] = rND;
                }
            }
            else
            {
                // No cube available
                arCubeful[ici] = arCf[i];
            }
        }
    }

    /// <summary>
    /// Generate noise for evaluation (deterministic or random).
    /// Port of Noise() from eval.c.
    /// </summary>
    internal static float Noise(EvalContext ec, Board board, int iOutput)
    {
        float r;

        if (ec.Deterministic)
        {
            // Deterministic: derive noise from board hash
            // Interleave opponent and player point values into 50 bytes
            byte[] auchBoard = new byte[50];
            for (int i = 0; i < 25; i++)
            {
                auchBoard[i << 1] = (byte)board.Opponent[i];
                auchBoard[(i << 1) + 1] = (byte)board.Player[i];
            }
            auchBoard[0] += (byte)iOutput;

            // MD5 hash
            byte[] hash = System.Security.Cryptography.MD5.HashData(auchBoard);

            // Sum bytes → approximately normal distribution (central limit theorem)
            r = 0.0f;
            for (int i = 0; i < 16; i++)
                r += hash[i];
            r -= 2040.0f;
            r /= 295.6f;
        }
        else
        {
            // Non-deterministic: Box-Muller transform
            float x, y, rsq;
            do
            {
                x = System.Random.Shared.NextSingle() * 2.0f - 1.0f;
                y = System.Random.Shared.NextSingle() * 2.0f - 1.0f;
                rsq = x * x + y * y;
            } while (rsq > 1.0f || rsq == 0.0f);

            r = y * MathF.Sqrt(-2.0f * MathF.Log(rsq) / rsq);
        }

        r *= ec.Noise;

        // Scale by output type
        if (iOutput == Constants.OutputWinGammon || iOutput == Constants.OutputLoseGammon)
            r *= 0.25f;
        else if (iOutput == Constants.OutputWinBackgammon || iOutput == Constants.OutputLoseBackgammon)
            r *= 0.01f;

        return r;
    }

    /// <summary>
    /// Evaluate a bearoff position with perfect cubeful equities.
    /// Port of EvaluatePerfectCubeful() / PerfectCubeful() from eval.c.
    /// Returns 4 floats: [cubeless, cube-owned, cube-centered, cube-opponent].
    /// </summary>
    public bool EvaluatePerfectCubeful(Board board, Span<float> arEquity)
    {
        var pc = Classifier.Classify(board, this);

        if (pc == PositionClass.BearoffTwoSided && TwoSidedBearoff is { Cubeful: true })
        {
            TwoSidedBearoff.GetCubefulEquities(board, arEquity);
            return true;
        }

        return false;
    }

    /// <summary>Flush all evaluation caches.</summary>
    public void FlushCaches()
    {
        _mainCache.Flush();
        _pruneCache.Flush();
    }
}
