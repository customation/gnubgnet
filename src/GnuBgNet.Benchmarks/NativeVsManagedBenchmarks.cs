// SPDX-License-Identifier: GPL-3.0-or-later

using BenchmarkDotNet.Attributes;
using GnuBgNet;
using GnuBgNet.Encoding;
using GnuBgNet.MoveGeneration;
using GnuBgNet.NeuralNet;
using GnuBgNet.Optimized;
using GammonBase.Gnubg;

namespace GnuBgNet.Benchmarks;

/// <summary>
/// Performance comparison between native gnubg C library (via P/Invoke)
/// and pure managed C# GnuBgNet — both original and optimized pipelines.
/// Uses multiple distinct positions to avoid cache distortion.
/// Both native and managed have internal eval caches; the comparison is
/// fair because real usage also benefits from caching.
/// </summary>
[MemoryDiagnoser]
public class NativeVsManagedBenchmarks
{
    private Engine _managed = null!;
    private Engine _managedOptimized = null!;
    private GnubgApiContext _native = null!;

    // Multiple distinct position IDs to reduce cache hits
    private string[] _positionIds = null!;

    [GlobalSetup]
    public void Setup()
    {
        var dataDir = BenchmarkSetup.FindDataDir();

        // Original managed engine
        _managed = Engine.Create(dataDir);

        // Optimized managed engine (SIMD nets + UndoStack + Sparse inputs)
        var nets = NetworkSet.LoadBinary(Path.Combine(dataDir, "gnubg.wd"));
        var simdNets = new NetworkSet
        {
            Contact = new SimdNeuralNetwork((NeuralNetwork)nets.Contact),
            Race = new SimdNeuralNetwork((NeuralNetwork)nets.Race),
            Crashed = new SimdNeuralNetwork((NeuralNetwork)nets.Crashed),
            PruneContact = new SimdNeuralNetwork((NeuralNetwork)nets.PruneContact),
            PruneCrashed = new SimdNeuralNetwork((NeuralNetwork)nets.PruneCrashed),
            PruneRace = new SimdNeuralNetwork((NeuralNetwork)nets.PruneRace),
        };
        _managedOptimized = Engine.Create(dataDir, simdNets,
            moveGenerator: UndoStackMoveGenerator.Instance,
            inputCalculator: SparseInputCalculator.Instance);

        // Native C engine
        var weightsPath = Path.Combine(dataDir, "gnubg.weights");
        var weightsBinPath = Path.Combine(dataDir, "gnubg.wd");
        _native = GnubgApiContext.Create();
        _native.Init(weightsPath, weightsBinPath, dataDir, noBearoff: false);

        // Generate distinct positions by playing out from opening with varied dice
        _positionIds = GenerateDistinctPositions(20);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _managed?.Dispose();
        _managedOptimized?.Dispose();
        if (_native is not null && !_native.IsInvalid)
        {
            _native.Shutdown();
            _native.Dispose();
        }
    }

    // No IterationSetup — both sides benefit from their internal eval caches.
    // With 20 distinct positions, the warmup fills both caches equally.
    // This mirrors real-world usage where cache hits are the norm.

    // --- 0-ply: evaluate 20 distinct positions ---

    [Benchmark]
    public void Native_Eval_0Ply_20Pos()
    {
        for (int i = 0; i < _positionIds.Length; i++)
            _native.EvaluatePosition(_positionIds[i]);
    }

    [Benchmark(Baseline = true)]
    public void Managed_Eval_0Ply_20Pos()
    {
        for (int i = 0; i < _positionIds.Length; i++)
            _managed.EvaluatePosition(_positionIds[i]);
    }

    [Benchmark]
    public void Optimized_Eval_0Ply_20Pos()
    {
        for (int i = 0; i < _positionIds.Length; i++)
            _managedOptimized.EvaluatePosition(_positionIds[i]);
    }

    // --- 2-ply: evaluate 5 distinct positions (expensive) ---

    [Benchmark]
    public void Native_Eval_2Ply_5Pos()
    {
        for (int i = 0; i < 5; i++)
            _native.EvaluatePositionPlied(_positionIds[i], 2);
    }

    [Benchmark]
    public void Managed_Eval_2Ply_5Pos()
    {
        for (int i = 0; i < 5; i++)
            _managed.EvaluatePositionPlied(_positionIds[i], 2);
    }

    [Benchmark]
    public void Optimized_Eval_2Ply_5Pos()
    {
        for (int i = 0; i < 5; i++)
            _managedOptimized.EvaluatePositionPlied(_positionIds[i], 2);
    }

    private string[] GenerateDistinctPositions(int count)
    {
        var positions = new List<string>();
        var rng = new System.Random(12345);

        // Start from opening and make random valid moves with varied dice
        for (int i = 0; i < count; i++)
        {
            var board = Board.Opening();
            // Make 1-3 random moves to get varied mid-game positions
            int numMoves = rng.Next(1, 4);
            for (int m = 0; m < numMoves; m++)
            {
                int d0 = rng.Next(1, 7);
                int d1 = rng.Next(1, 7);
                var ml = MoveGenerator.GenerateMoves(board, d0, d1);
                if (ml.Moves.Count > 0)
                {
                    var move = ml.Moves[rng.Next(ml.Moves.Count)];
                    board = PositionId.FromKey(move.Key);
                    board.SwapSides();
                }
            }
            positions.Add(PositionId.Encode(board));
        }

        return positions.ToArray();
    }
}
