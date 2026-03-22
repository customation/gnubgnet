// SPDX-License-Identifier: GPL-3.0-or-later
// Turn-synchronous rollout engine with batched GPU evaluation.

using GnuBgNet.Encoding;
using GnuBgNet.Evaluation;
using GnuBgNet.MatchEquity;
using GnuBgNet.MoveGeneration;
using GnuBgNet.NeuralNet;
using GnuBgNet.Random;

namespace GnuBgNet.Gpu;

/// <summary>
/// Rollout engine where all trials advance one turn at a time in lockstep.
/// At each turn barrier:
///   1. Move generation on CPU (fast, branchy) for all active trials
///   2. Batch all candidate evaluations into one GPU call
///   3. Select best moves, advance all boards
///
/// This amortizes GPU kernel launch overhead across thousands of trials,
/// unlike the standard RolloutEngine which evaluates one position at a time.
/// </summary>
public sealed class TurnSynchronousRolloutEngine
{
    private readonly IPositionEvaluator _evaluator;
    private readonly IMoveGenerator _moveGen;
    private readonly IInputCalculator _inputCalc;
    private readonly IBatchNeuralNetwork _contactNet;
    private readonly IBatchNeuralNetwork _raceNet;
    private readonly IBatchNeuralNetwork _crashedNet;

    public TurnSynchronousRolloutEngine(
        IPositionEvaluator evaluator,
        IMoveGenerator moveGen,
        IInputCalculator inputCalc,
        IBatchNeuralNetwork contactNet,
        IBatchNeuralNetwork raceNet,
        IBatchNeuralNetwork crashedNet)
    {
        _evaluator = evaluator;
        _moveGen = moveGen;
        _inputCalc = inputCalc;
        _contactNet = contactNet;
        _raceNet = raceNet;
        _crashedNet = crashedNet;
    }

    /// <summary>
    /// Run a turn-synchronous money-game rollout.
    /// Returns averaged probabilities over all trials [5 outputs].
    /// </summary>
    public float[] RunRollout(Board startBoard, int numTrials, int maxTurns = 400, int? seed = null)
    {
        var rng = new MersenneTwister(seed.HasValue ? (uint)seed.Value : (uint)Environment.TickCount);

        // Initialize trials
        var trials = new TrialState[numTrials];
        for (int t = 0; t < numTrials; t++)
        {
            trials[t] = new TrialState
            {
                Board = startBoard.Clone(),
                Active = true,
                Turn = 0,
                Outputs = new float[Constants.NumOutputs],
            };
        }

        // Advance all trials turn by turn
        for (int turn = 0; turn < maxTurns; turn++)
        {
            int activeCount = 0;
            for (int t = 0; t < numTrials; t++)
                if (trials[t].Active) activeCount++;

            if (activeCount == 0) break;

            ProcessTurn(trials, rng, turn);
        }

        // Handle trials that didn't finish (truncation): evaluate at 0-ply
        TruncateRemainingTrials(trials);

        // Average results
        var result = new float[Constants.NumOutputs];
        for (int t = 0; t < numTrials; t++)
        {
            for (int o = 0; o < Constants.NumOutputs; o++)
                result[o] += trials[t].Outputs[o];
        }
        for (int o = 0; o < Constants.NumOutputs; o++)
            result[o] /= numTrials;

        return result;
    }

    private void ProcessTurn(TrialState[] trials, MersenneTwister rng, int turn)
    {
        int fMove = turn & 1; // 0 = player 0 on roll

        // Step 1: Check for game-over and roll dice for active trials
        var activeTrials = new List<int>();
        var diceRolls = new List<(int D0, int D1)>();

        for (int t = 0; t < trials.Length; t++)
        {
            if (!trials[t].Active) continue;

            var pc = _evaluator.ClassifyPosition(trials[t].Board);
            if (pc == PositionClass.Over)
            {
                var output = new float[Constants.NumOutputs].AsSpan();
                _evaluator.EvaluatePositionByClass(trials[t].Board, output, pc);
                if (fMove != 0)
                    Evaluator.InvertEvaluation(output);
                output.CopyTo(trials[t].Outputs);
                trials[t].Active = false;
                continue;
            }

            var roll = rng.NextDiceRoll();
            activeTrials.Add(t);
            diceRolls.Add(roll);
        }

        if (activeTrials.Count == 0) return;

        // Step 2: Generate moves for all active trials (CPU, branchy)
        var candidateGroups = new List<(int TrialIdx, List<Board> Candidates)>();
        var forcedMoves = new List<(int TrialIdx, Board Board)>();

        for (int i = 0; i < activeTrials.Count; i++)
        {
            int t = activeTrials[i];
            var (d0, d1) = diceRolls[i];
            var ml = _moveGen.GenerateMoves(trials[t].Board, d0, d1);

            if (ml.Moves.Count <= 1)
            {
                // Forced or no move — no evaluation needed
                Board newBoard;
                if (ml.Moves.Count == 0)
                    newBoard = trials[t].Board.Clone();
                else
                    newBoard = PositionId.FromKey(ml.Moves[0].Key);
                newBoard.SwapSides();
                forcedMoves.Add((t, newBoard));
            }
            else
            {
                // Multiple candidates need evaluation
                var boards = new List<Board>(ml.Moves.Count);
                foreach (var move in ml.Moves)
                    boards.Add(PositionId.FromKey(move.Key).Swapped());
                candidateGroups.Add((t, boards));
            }
        }

        // Step 3: Batch evaluate ALL candidates across ALL trials
        if (candidateGroups.Count > 0)
        {
            // Flatten all candidate boards
            var allCandidates = new List<Board>();
            var groupOffsets = new int[candidateGroups.Count];
            int offset = 0;

            for (int g = 0; g < candidateGroups.Count; g++)
            {
                groupOffsets[g] = offset;
                allCandidates.AddRange(candidateGroups[g].Candidates);
                offset += candidateGroups[g].Candidates.Count;
            }

            // Batch evaluate
            var allOutputs = BatchEvaluateAll(allCandidates);

            // Step 4: For each trial, select best move
            for (int g = 0; g < candidateGroups.Count; g++)
            {
                int trialIdx = candidateGroups[g].TrialIdx;
                int numCandidates = candidateGroups[g].Candidates.Count;
                int baseOffset = groupOffsets[g];

                float bestEquity = float.NegativeInfinity;
                int bestIdx = 0;

                for (int c = 0; c < numCandidates; c++)
                {
                    var inverted = new float[Constants.NumOutputs].AsSpan();
                    allOutputs[baseOffset + c].AsSpan(0, Constants.NumOutputs).CopyTo(inverted);
                    Evaluator.InvertEvaluation(inverted);
                    float eq = MatchEquityTable.MoneyEquity(inverted);

                    if (eq > bestEquity)
                    {
                        bestEquity = eq;
                        bestIdx = c;
                    }
                }

                trials[trialIdx].Board = candidateGroups[g].Candidates[bestIdx];
            }
        }

        // Apply forced moves
        foreach (var (t, board) in forcedMoves)
            trials[t].Board = board;

        // Advance turn counter
        foreach (int t in activeTrials)
            if (trials[t].Active)
                trials[t].Turn++;
    }

    private float[][] BatchEvaluateAll(List<Board> boards)
    {
        var outputs = new float[boards.Count][];
        var nnIndices = new List<int>();

        // Classify and handle non-NN positions
        for (int i = 0; i < boards.Count; i++)
        {
            var pc = _evaluator.ClassifyPosition(boards[i]);
            if (pc == PositionClass.Over || pc <= PositionClass.BearoffTwoSided)
            {
                outputs[i] = new float[Constants.NumOutputs];
                _evaluator.EvaluatePositionByClass(boards[i], outputs[i], pc);
            }
            else
            {
                nnIndices.Add(i);
            }
        }

        // Group by class and batch evaluate
        BatchEvalByClass(boards, outputs, nnIndices, PositionClass.Contact, _contactNet,
            (b, inp) => _inputCalc.CalculateContactInputs(b, inp), Constants.NumContactInputs);
        BatchEvalByClass(boards, outputs, nnIndices, PositionClass.Race, _raceNet,
            (b, inp) => _inputCalc.CalculateRaceInputs(b, inp), Constants.NumRaceInputs);
        BatchEvalByClass(boards, outputs, nnIndices, PositionClass.Crashed, _crashedNet,
            (b, inp) => _inputCalc.CalculateCrashedInputs(b, inp), Constants.NumCrashedInputs);

        return outputs;
    }

    private void BatchEvalByClass(List<Board> boards, float[][] outputs, List<int> nnIndices,
        PositionClass targetClass, IBatchNeuralNetwork net,
        Action<Board, Span<float>> computeInputs, int inputSize)
    {
        var matchingIndices = new List<int>();
        foreach (int i in nnIndices)
            if (_evaluator.ClassifyPosition(boards[i]) == targetClass)
                matchingIndices.Add(i);

        if (matchingIndices.Count == 0) return;

        int batchSize = matchingIndices.Count;
        var flatInputs = new float[batchSize * inputSize];
        var flatOutputs = new float[batchSize * Constants.NumOutputs];

        for (int b = 0; b < batchSize; b++)
            computeInputs(boards[matchingIndices[b]],
                flatInputs.AsSpan(b * inputSize, inputSize));

        net.EvaluateBatch(flatInputs, flatOutputs, batchSize);

        for (int b = 0; b < batchSize; b++)
        {
            outputs[matchingIndices[b]] = new float[Constants.NumOutputs];
            flatOutputs.AsSpan(b * Constants.NumOutputs, Constants.NumOutputs)
                .CopyTo(outputs[matchingIndices[b]]);
        }
    }

    private void TruncateRemainingTrials(TrialState[] trials)
    {
        for (int t = 0; t < trials.Length; t++)
        {
            if (!trials[t].Active) continue;

            var output = new float[Constants.NumOutputs].AsSpan();
            _evaluator.EvaluatePosition(trials[t].Board, output);
            if ((trials[t].Turn & 1) != 0)
                Evaluator.InvertEvaluation(output);
            output.CopyTo(trials[t].Outputs);
            trials[t].Active = false;
        }
    }

    private class TrialState
    {
        public Board Board = null!;
        public bool Active;
        public int Turn;
        public float[] Outputs = null!;
    }
}
