// SPDX-License-Identifier: GPL-3.0-or-later

using BenchmarkDotNet.Attributes;
using GnuBgNet;
using GnuBgNet.Evaluation;
using GnuBgNet.NeuralNet;
using GnuBgNet.Optimized;

namespace GnuBgNet.Benchmarks;

/// <summary>
/// End-to-end benchmark comparing original vs optimized pipeline
/// through the Evaluator with all optimized components plugged in.
/// </summary>
[MemoryDiagnoser]
public class FullEvaluationBenchmarks
{
    private Evaluator _originalEval = null!;
    private Evaluator _optimizedEval = null!;
    private Board _openingBoard;
    private Board _contactBoard;
    private Board _raceBoard;

    [GlobalSetup]
    public void Setup()
    {
        var dataDir = BenchmarkSetup.FindDataDir();
        var nets = NetworkSet.LoadBinary(Path.Combine(dataDir, "gnubg.wd"));

        // Original evaluator with default implementations
        _originalEval = new Evaluator(nets);

        // Build SIMD network set
        var simdNets = new NetworkSet
        {
            Contact = new SimdNeuralNetwork((NeuralNetwork)nets.Contact),
            Race = new SimdNeuralNetwork((NeuralNetwork)nets.Race),
            Crashed = new SimdNeuralNetwork((NeuralNetwork)nets.Crashed),
            PruneContact = new SimdNeuralNetwork((NeuralNetwork)nets.PruneContact),
            PruneCrashed = new SimdNeuralNetwork((NeuralNetwork)nets.PruneCrashed),
            PruneRace = new SimdNeuralNetwork((NeuralNetwork)nets.PruneRace),
        };

        // Optimized evaluator with all optimized components
        _optimizedEval = new Evaluator(
            simdNets,
            moveGenerator: UndoStackMoveGenerator.Instance,
            inputCalculator: SparseInputCalculator.Instance);

        _openingBoard = BenchmarkSetup.CreateOpeningBoard();
        _contactBoard = BenchmarkSetup.CreateContactBoard();
        _raceBoard = BenchmarkSetup.CreateRaceBoard();
    }

    [Benchmark(Baseline = true)]
    public void Original_0Ply_Opening()
    {
        Span<float> output = stackalloc float[Constants.NumOutputs];
        _originalEval.EvaluatePosition(_openingBoard, output);
    }

    [Benchmark]
    public void Optimized_0Ply_Opening()
    {
        Span<float> output = stackalloc float[Constants.NumOutputs];
        _optimizedEval.EvaluatePosition(_openingBoard, output);
    }

    [Benchmark]
    public void Original_0Ply_Contact()
    {
        Span<float> output = stackalloc float[Constants.NumOutputs];
        _originalEval.EvaluatePosition(_contactBoard, output);
    }

    [Benchmark]
    public void Optimized_0Ply_Contact()
    {
        Span<float> output = stackalloc float[Constants.NumOutputs];
        _optimizedEval.EvaluatePosition(_contactBoard, output);
    }

    [Benchmark]
    public void Original_0Ply_Race()
    {
        Span<float> output = stackalloc float[Constants.NumOutputs];
        _originalEval.EvaluatePosition(_raceBoard, output);
    }

    [Benchmark]
    public void Optimized_0Ply_Race()
    {
        Span<float> output = stackalloc float[Constants.NumOutputs];
        _optimizedEval.EvaluatePosition(_raceBoard, output);
    }

    [Benchmark]
    public void Original_FindBestMoves_31()
    {
        var ec = EvalContext.ZeroPly();
        var ml = new MoveList();
        _originalEval.FindnSaveBestMoves(ml, _openingBoard, 3, 1, ec);
    }

    [Benchmark]
    public void Optimized_FindBestMoves_31()
    {
        var ec = EvalContext.ZeroPly();
        var ml = new MoveList();
        _optimizedEval.FindnSaveBestMoves(ml, _openingBoard, 3, 1, ec);
    }
}
