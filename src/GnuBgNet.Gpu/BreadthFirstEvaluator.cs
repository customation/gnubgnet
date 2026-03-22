// SPDX-License-Identifier: GPL-3.0-or-later
// Breadth-first n-ply evaluator with batched neural network evaluation.

using GnuBgNet.Encoding;
using GnuBgNet.Evaluation;
using GnuBgNet.MatchEquity;
using GnuBgNet.MoveGeneration;
using GnuBgNet.NeuralNet;

namespace GnuBgNet.Gpu;

/// <summary>
/// Restructures gnubg's depth-first recursive n-ply search into breadth-first
/// level-by-level expansion with batched 0-ply evaluation at each level.
///
/// At each ply, ALL candidate positions across ALL parent positions are collected
/// and batch-evaluated in a single call (one GPU kernel launch), then the best
/// move per (parent, roll) is selected. This replaces ~7000 sequential evaluations
/// (for 2-ply) with a handful of large batch calls.
///
/// Tree structure per ply (same as original gnubg):
///   Parent → 21 dice rolls → ~15 legal moves each → 0-ply eval → select best → 21 children
///
/// For 2-ply evaluation of K positions: K × (315 + 6615) ≈ 7K move-selection evals + K×441 leaf evals.
/// </summary>
public sealed class BreadthFirstEvaluator : IPositionEvaluator
{
    private const int NumRolls = 21;

    private readonly Evaluator _fallback;
    private readonly IMoveGenerator _moveGen;
    private readonly IInputCalculator _inputCalc;
    private readonly IBatchNeuralNetwork _contactNet;
    private readonly IBatchNeuralNetwork _raceNet;
    private readonly IBatchNeuralNetwork _crashedNet;

    // Dice roll table: (d0, d1, weight) for all 21 distinct outcomes
    private static readonly (int D0, int D1, float Weight)[] DiceRolls = BuildDiceRolls();

    public BreadthFirstEvaluator(
        Evaluator fallback,
        IMoveGenerator moveGen,
        IInputCalculator inputCalc,
        IBatchNeuralNetwork contactNet,
        IBatchNeuralNetwork raceNet,
        IBatchNeuralNetwork crashedNet)
    {
        _fallback = fallback;
        _moveGen = moveGen;
        _inputCalc = inputCalc;
        _contactNet = contactNet;
        _raceNet = raceNet;
        _crashedNet = crashedNet;
    }

    // --- IPositionEvaluator implementation ---

    public PositionClass ClassifyPosition(Board board) => _fallback.ClassifyPosition(board);

    public void EvaluatePosition(Board board, Span<float> output)
        => _fallback.EvaluatePosition(board, output);

    public void EvaluatePositionByClass(Board board, Span<float> output, PositionClass pc)
        => _fallback.EvaluatePositionByClass(board, output, pc);

    public void EvaluatePositionPlied(Board board, Span<float> output, int nPlies, bool usePrune = true)
    {
        if (nPlies == 0)
        {
            _fallback.EvaluatePosition(board, output);
            return;
        }

        EvaluatePositionsBfs([board], nPlies, out var results);
        results[0].AsSpan(0, Constants.NumOutputs).CopyTo(output);
    }

    public void EvaluatePositionPlied(Board board, Span<float> output, int nPlies, bool usePrune,
        EvalContext? ec, CubeInfo? ci = null)
        => EvaluatePositionPlied(board, output, nPlies, usePrune);

    public void FindnSaveBestMoves(MoveList ml, Board board, int nDice0, int nDice1,
        EvalContext ec, MoveFilter[,]? moveFilters = null)
    {
        _moveGen.GenerateMovesInto(ml, board, nDice0, nDice1);
        if (ml.Moves.Count == 0) return;

        moveFilters ??= MoveFilterPresets.Default;
        int filterRow = Math.Min(ec.Plies, MoveFilterPresets.MaxFilterPlies) - 1;
        if (filterRow < 0) filterRow = 0;

        for (int iPly = 0; iPly < ec.Plies; iPly++)
        {
            var filter = iPly < MoveFilterPresets.MaxFilterPlies
                ? moveFilters[filterRow, iPly]
                : MoveFilter.Null;

            if (filter.Accept < 0) continue;

            // Batch score all remaining moves at iPly
            BatchScoreMoves(ml, iPly);
            SortMovesByScore(ml);
            ApplyFilter(ml, filter);

            if (ml.Moves.Count == 1 && filter.Accept != 1) break;
        }

        // Final scoring at full ply depth
        BatchScoreMoves(ml, ec.Plies);
        SortMovesByScore(ml);
    }

    public void GeneralEvaluationEPlied(Board board, Span<float> arOutput,
        CubeInfo ci, EvalContext ec, int nPlies)
        => _fallback.GeneralEvaluationEPlied(board, arOutput, ci, ec, nPlies);

    public bool EvaluatePerfectCubeful(Board board, Span<float> arEquity)
        => _fallback.EvaluatePerfectCubeful(board, arEquity);

    public int GameStatus(Board board) => _fallback.GameStatus(board);
    public void FlushCaches() => _fallback.FlushCaches();

    // --- Core BFS Algorithm ---

    /// <summary>
    /// Evaluate multiple positions at n-ply using breadth-first search with batched NN eval.
    /// </summary>
    private void EvaluatePositionsBfs(Board[] roots, int nPlies, out float[][] results)
    {
        // Level 0: root positions
        var currentLevel = new BfsNode[roots.Length];
        for (int i = 0; i < roots.Length; i++)
            currentLevel[i] = new BfsNode { Board = roots[i] };

        // Expand level by level
        var levels = new List<BfsNode[]> { currentLevel };

        for (int ply = 0; ply < nPlies; ply++)
        {
            currentLevel = ExpandLevel(currentLevel);
            levels.Add(currentLevel);
        }

        // The last level's evaluations (from ExpandLevel's batch eval) are the leaf values.
        // Backpropagate from leaves to roots.
        for (int ply = levels.Count - 1; ply > 0; ply--)
            BackpropagateLevel(levels[ply], levels[ply - 1]);

        // Extract results
        results = new float[roots.Length][];
        for (int i = 0; i < roots.Length; i++)
            results[i] = levels[0][i].Outputs!;
    }

    /// <summary>
    /// Expand one BFS level: for each parent, try all 21 dice rolls,
    /// generate all legal moves, batch-evaluate all candidates at 0-ply,
    /// then select the best move per (parent, roll).
    /// Returns the next level with exactly parentCount × 21 nodes.
    /// </summary>
    private BfsNode[] ExpandLevel(BfsNode[] parents)
    {
        // Step 1: Generate all candidate boards
        var candidates = new List<CandidateMove>();

        for (int p = 0; p < parents.Length; p++)
        {
            var parent = parents[p];
            if (parent.IsTerminal) continue;

            for (int r = 0; r < NumRolls; r++)
            {
                var (d0, d1, weight) = DiceRolls[r];
                int groupKey = p * NumRolls + r;

                var ml = _moveGen.GenerateMoves(parent.Board, d0, d1);

                if (ml.Moves.Count == 0)
                {
                    // No legal moves: board stays, swap sides
                    candidates.Add(new CandidateMove
                    {
                        GroupKey = groupKey,
                        Board = parent.Board.Swapped(),
                        DiceWeight = weight,
                    });
                }
                else
                {
                    foreach (var move in ml.Moves)
                    {
                        candidates.Add(new CandidateMove
                        {
                            GroupKey = groupKey,
                            Board = PositionId.FromKey(move.Key).Swapped(),
                            DiceWeight = weight,
                        });
                    }
                }
            }
        }

        // Step 2: Batch evaluate ALL candidates at 0-ply
        BatchEvaluateBoards(candidates);

        // Step 3: Select best per (parent, roll) group
        var nextLevel = new BfsNode[parents.Length * NumRolls];

        // Initialize with negative infinity
        var bestEquity = new float[parents.Length * NumRolls];
        Array.Fill(bestEquity, float.NegativeInfinity);

        foreach (var cand in candidates)
        {
            // Compute equity from the inverted (parent-perspective) outputs
            Span<float> inverted = new float[Constants.NumOutputs];
            cand.Outputs.AsSpan(0, Constants.NumOutputs).CopyTo(inverted);
            Evaluator.InvertEvaluation(inverted);
            float eq = MatchEquityTable.MoneyEquity(inverted);

            if (eq > bestEquity[cand.GroupKey])
            {
                bestEquity[cand.GroupKey] = eq;
                nextLevel[cand.GroupKey] = new BfsNode
                {
                    Board = cand.Board,
                    Outputs = cand.Outputs,
                    DiceWeight = cand.DiceWeight,
                };
            }
        }

        // Handle terminals at parent level — propagate terminal flag
        for (int p = 0; p < parents.Length; p++)
        {
            if (!parents[p].IsTerminal) continue;
            for (int r = 0; r < NumRolls; r++)
            {
                int idx = p * NumRolls + r;
                nextLevel[idx] = new BfsNode
                {
                    Board = parents[p].Board,
                    Outputs = parents[p].Outputs,
                    DiceWeight = DiceRolls[r].Weight,
                    IsTerminal = true,
                };
            }
        }

        return nextLevel;
    }

    /// <summary>
    /// Backpropagate: for each parent, average best-move scores across 21 rolls (weighted), inverted.
    /// </summary>
    private static void BackpropagateLevel(BfsNode[] children, BfsNode[] parents)
    {
        for (int p = 0; p < parents.Length; p++)
        {
            if (parents[p].IsTerminal) continue;

            var output = new float[Constants.NumOutputs];

            for (int r = 0; r < NumRolls; r++)
            {
                int childIdx = p * NumRolls + r;
                var child = children[childIdx];
                if (child.Outputs == null) continue;

                float w = child.DiceWeight;
                for (int o = 0; o < Constants.NumOutputs; o++)
                    output[o] += w * child.Outputs[o];
            }

            // Normalize by 36 and invert to parent's perspective
            for (int o = 0; o < Constants.NumOutputs; o++)
                output[o] /= 36.0f;
            Evaluator.InvertEvaluation(output);

            parents[p].Outputs = output;
        }
    }

    /// <summary>
    /// Batch evaluate a list of boards at 0-ply. Groups by position class
    /// and uses the appropriate batched neural network.
    /// </summary>
    private void BatchEvaluateBoards(List<CandidateMove> candidates)
    {
        // Classify all positions
        for (int i = 0; i < candidates.Count; i++)
        {
            var cand = candidates[i];
            cand.PositionClass = _fallback.ClassifyPosition(cand.Board);

            // Handle non-NN classes immediately
            if (cand.PositionClass <= PositionClass.BearoffTwoSided || cand.PositionClass == PositionClass.Over)
            {
                cand.Outputs = new float[Constants.NumOutputs];
                _fallback.EvaluatePositionByClass(cand.Board, cand.Outputs, cand.PositionClass);
            }
        }

        // Batch evaluate each NN class
        BatchEvalGroup(candidates, PositionClass.Contact, _contactNet,
            (b, inp) => _inputCalc.CalculateContactInputs(b, inp), Constants.NumContactInputs);
        BatchEvalGroup(candidates, PositionClass.Race, _raceNet,
            (b, inp) => _inputCalc.CalculateRaceInputs(b, inp), Constants.NumRaceInputs);
        BatchEvalGroup(candidates, PositionClass.Crashed, _crashedNet,
            (b, inp) => _inputCalc.CalculateCrashedInputs(b, inp), Constants.NumCrashedInputs);
    }

    private void BatchEvalGroup(List<CandidateMove> candidates, PositionClass pc,
        IBatchNeuralNetwork net, Action<Board, Span<float>> computeInputs, int inputSize)
    {
        // Collect indices matching this class
        var indices = new List<int>();
        for (int i = 0; i < candidates.Count; i++)
            if (candidates[i].PositionClass == pc)
                indices.Add(i);

        if (indices.Count == 0) return;

        int batchSize = indices.Count;
        var flatInputs = new float[batchSize * inputSize];
        var flatOutputs = new float[batchSize * Constants.NumOutputs];

        // Compute inputs
        for (int b = 0; b < batchSize; b++)
            computeInputs(candidates[indices[b]].Board,
                flatInputs.AsSpan(b * inputSize, inputSize));

        // Batch evaluate
        net.EvaluateBatch(flatInputs, flatOutputs, batchSize);

        // Copy outputs back
        for (int b = 0; b < batchSize; b++)
        {
            var cand = candidates[indices[b]];
            cand.Outputs = new float[Constants.NumOutputs];
            flatOutputs.AsSpan(b * Constants.NumOutputs, Constants.NumOutputs).CopyTo(cand.Outputs);
        }
    }

    /// <summary>
    /// Score all moves in a MoveList at the given ply using BFS batch evaluation.
    /// </summary>
    private void BatchScoreMoves(MoveList ml, int nPlies)
    {
        // Collect all move-result boards
        var boards = new Board[ml.Moves.Count];
        for (int i = 0; i < ml.Moves.Count; i++)
            boards[i] = PositionId.FromKey(ml.Moves[i].Key).Swapped();

        float[][] outputs;
        if (nPlies == 0)
        {
            // Direct batch eval
            var candidates = new List<CandidateMove>(boards.Length);
            for (int i = 0; i < boards.Length; i++)
                candidates.Add(new CandidateMove { Board = boards[i] });
            BatchEvaluateBoards(candidates);
            outputs = new float[boards.Length][];
            for (int i = 0; i < boards.Length; i++)
                outputs[i] = candidates[i].Outputs!;
        }
        else
        {
            // BFS n-ply evaluation of all boards
            EvaluatePositionsBfs(boards, nPlies, out outputs);
        }

        // Score each move
        ml.BestScore = -99999.9f;
        for (int i = 0; i < ml.Moves.Count; i++)
        {
            var inverted = new float[Constants.NumOutputs];
            outputs[i].AsSpan(0, Constants.NumOutputs).CopyTo(inverted);
            Evaluator.InvertEvaluation(inverted);

            float equity = MatchEquityTable.MoneyEquity(inverted);
            ml.Moves[i].Score = equity;
            ml.Moves[i].Score2 = equity;

            Array.Copy(inverted, ml.Moves[i].EvalOutputs, Constants.NumOutputs);
            ml.Moves[i].EvalOutputs[Constants.OutputEquity] = equity;
            ml.Moves[i].EvalOutputs[Constants.OutputCubefulEquity] = equity;

            if (equity > ml.BestScore)
            {
                ml.BestIndex = i;
                ml.BestScore = equity;
            }
        }
    }

    private static void SortMovesByScore(MoveList ml)
    {
        ml.Moves.Sort((a, b) => b.Score.CompareTo(a.Score));
        ml.BestIndex = 0;
        if (ml.Moves.Count > 0)
            ml.BestScore = ml.Moves[0].Score;
    }

    private static void ApplyFilter(MoveList ml, MoveFilter filter)
    {
        int keep = Math.Min(filter.Accept, ml.Moves.Count);
        if (keep == 0) keep = 1; // Always keep at least 1
        int limit = Math.Min(ml.Moves.Count, keep + filter.Extra);

        for (int i = keep; i < limit; i++)
        {
            if (ml.Moves[i].Score < ml.Moves[0].Score - filter.Threshold)
                break;
            keep = i + 1;
        }

        if (keep < ml.Moves.Count)
            ml.Moves.RemoveRange(keep, ml.Moves.Count - keep);
    }

    // --- Data Types ---

    private class BfsNode
    {
        public Board Board = null!;
        public float[]? Outputs;
        public float DiceWeight;
        public bool IsTerminal;
    }

    private class CandidateMove
    {
        public int GroupKey;
        public Board Board = null!;
        public float DiceWeight;
        public float[]? Outputs;
        public PositionClass PositionClass;
    }

    // --- Helpers ---

    private static (int, int, float)[] BuildDiceRolls()
    {
        var rolls = new List<(int, int, float)>();
        for (int n0 = 1; n0 <= 6; n0++)
            for (int n1 = 1; n1 <= n0; n1++)
                rolls.Add((n0, n1, n0 == n1 ? 1.0f : 2.0f));
        return rolls.ToArray();
    }
}
