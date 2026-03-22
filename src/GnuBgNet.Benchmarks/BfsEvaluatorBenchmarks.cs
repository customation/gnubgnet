// SPDX-License-Identifier: GPL-3.0-or-later

using BenchmarkDotNet.Attributes;
using GnuBgNet;
using GnuBgNet.Evaluation;
using GnuBgNet.Gpu;
using GnuBgNet.NeuralNet;
using GnuBgNet.Optimized;

namespace GnuBgNet.Benchmarks;

/// <summary>
/// Benchmarks comparing original depth-first n-ply evaluation (Evaluator)
/// vs breadth-first batched evaluation (BreadthFirstEvaluator).
/// Tests FindnSaveBestMoves at 0-ply and 2-ply, plus direct position evaluation.
/// </summary>
[MemoryDiagnoser]
public class BfsEvaluatorBenchmarks
{
    private Evaluator _original = null!;
    private BreadthFirstEvaluator _bfs = null!;
    private Board _contactBoard;
    private Board _openingBoard;

    [GlobalSetup]
    public void Setup()
    {
        var dataDir = BenchmarkSetup.FindDataDir();
        var nets = NetworkSet.LoadBinary(Path.Combine(dataDir, "gnubg.wd"));

        // Original evaluator
        _original = new Evaluator(nets);

        // BFS evaluator with SIMD batch networks (CPU batched — fair comparison)
        var contactNet = new SimdBatchNeuralNetwork((NeuralNetwork)nets.Contact);
        var raceNet = new SimdBatchNeuralNetwork((NeuralNetwork)nets.Race);
        var crashedNet = new SimdBatchNeuralNetwork((NeuralNetwork)nets.Crashed);

        _bfs = new BreadthFirstEvaluator(
            _original,
            UndoStackMoveGenerator.Instance,
            SparseInputCalculator.Instance,
            contactNet, raceNet, crashedNet);

        _openingBoard = BenchmarkSetup.CreateOpeningBoard();
        _contactBoard = BenchmarkSetup.CreateContactBoard();
    }

    [Benchmark(Baseline = true)]
    public void Original_FindBest_0Ply_Opening()
    {
        var ec = EvalContext.ZeroPly();
        var ml = new MoveList();
        _original.FindnSaveBestMoves(ml, _openingBoard, 3, 1, ec);
    }

    [Benchmark]
    public void Bfs_FindBest_0Ply_Opening()
    {
        var ec = EvalContext.ZeroPly();
        var ml = new MoveList();
        _bfs.FindnSaveBestMoves(ml, _openingBoard, 3, 1, ec);
    }

    [Benchmark]
    public void Original_FindBest_2Ply_Opening()
    {
        var ec = EvalContext.WorldClass();
        var ml = new MoveList();
        _original.FindnSaveBestMoves(ml, _openingBoard, 3, 1, ec);
    }

    [Benchmark]
    public void Bfs_FindBest_2Ply_Opening()
    {
        var ec = EvalContext.WorldClass();
        var ml = new MoveList();
        _bfs.FindnSaveBestMoves(ml, _openingBoard, 3, 1, ec);
    }

    [Benchmark]
    public void Original_Eval_2Ply_Contact()
    {
        Span<float> output = stackalloc float[Constants.NumOutputs];
        _original.EvaluatePositionPlied(_contactBoard, output, 2);
    }

    [Benchmark]
    public void Bfs_Eval_2Ply_Contact()
    {
        Span<float> output = stackalloc float[Constants.NumOutputs];
        _bfs.EvaluatePositionPlied(_contactBoard, output, 2);
    }
}
