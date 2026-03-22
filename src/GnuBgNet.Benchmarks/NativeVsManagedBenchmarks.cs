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
///
/// Each iteration uses freshly generated unique positions so neither
/// side's eval cache provides an advantage. This measures raw computation
/// speed without cache effects.
/// </summary>
[MemoryDiagnoser]
public class NativeVsManagedBenchmarks
{
    private Engine _managed = null!;
    private Engine _managedOptimized = null!;
    private GnubgApiContext _native = null!;
    private string _dataDir = null!;

    // Fresh positions generated each iteration
    private string[] _positions0Ply = null!;
    private string _position2Ply = null!;
    private int _iterationSeed;

    [GlobalSetup]
    public void Setup()
    {
        _dataDir = BenchmarkSetup.FindDataDir();

        // Original managed engine
        _managed = Engine.Create(_dataDir);

        // Optimized managed engine (SIMD nets + UndoStack + Sparse inputs)
        var nets = NetworkSet.LoadBinary(Path.Combine(_dataDir, "gnubg.wd"));
        var simdNets = new NetworkSet
        {
            Contact = new SimdNeuralNetwork((NeuralNetwork)nets.Contact),
            Race = new SimdNeuralNetwork((NeuralNetwork)nets.Race),
            Crashed = new SimdNeuralNetwork((NeuralNetwork)nets.Crashed),
            PruneContact = new SimdNeuralNetwork((NeuralNetwork)nets.PruneContact),
            PruneCrashed = new SimdNeuralNetwork((NeuralNetwork)nets.PruneCrashed),
            PruneRace = new SimdNeuralNetwork((NeuralNetwork)nets.PruneRace),
        };
        _managedOptimized = Engine.Create(_dataDir, simdNets,
            moveGenerator: UndoStackMoveGenerator.Instance,
            inputCalculator: SparseInputCalculator.Instance);

        // Native C engine
        var weightsPath = Path.Combine(_dataDir, "gnubg.weights");
        var weightsBinPath = Path.Combine(_dataDir, "gnubg.wd");
        _native = GnubgApiContext.Create();
        _native.Init(weightsPath, weightsBinPath, _dataDir, noBearoff: false);
    }

    [IterationSetup]
    public void IterationSetup()
    {
        // Generate unique positions each iteration so caches never help.
        // Incrementing seed ensures every iteration gets different positions.
        _iterationSeed++;
        _positions0Ply = GenerateDistinctPositions(20, _iterationSeed * 1000);
        _position2Ply = GenerateDistinctPositions(1, _iterationSeed * 2000)[0];
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

    // --- 0-ply: evaluate 20 unique positions (no cache hits) ---

    [Benchmark]
    public void Native_Eval_0Ply_20Pos()
    {
        for (int i = 0; i < _positions0Ply.Length; i++)
            _native.EvaluatePosition(_positions0Ply[i]);
    }

    [Benchmark(Baseline = true)]
    public void Managed_Eval_0Ply_20Pos()
    {
        for (int i = 0; i < _positions0Ply.Length; i++)
            _managed.EvaluatePosition(_positions0Ply[i]);
    }

    [Benchmark]
    public void Optimized_Eval_0Ply_20Pos()
    {
        for (int i = 0; i < _positions0Ply.Length; i++)
            _managedOptimized.EvaluatePosition(_positions0Ply[i]);
    }

    // --- 2-ply: evaluate 1 unique position (no cache hits) ---

    [Benchmark]
    public GnubgEvaluationResult Native_Eval_2Ply()
        => _native.EvaluatePositionPlied(_position2Ply, 2);

    [Benchmark]
    public EvaluationResult Managed_Eval_2Ply()
        => _managed.EvaluatePositionPlied(_position2Ply, 2);

    [Benchmark]
    public EvaluationResult Optimized_Eval_2Ply()
        => _managedOptimized.EvaluatePositionPlied(_position2Ply, 2);

    /// <summary>
    /// Generate distinct mid-game positions by playing random moves from opening.
    /// Each call with a different seed produces entirely different positions.
    /// </summary>
    private string[] GenerateDistinctPositions(int count, int seed)
    {
        var positions = new List<string>(count);
        var rng = new System.Random(seed);

        for (int i = 0; i < count; i++)
        {
            var board = Board.Opening();
            // Play 2-6 random half-moves to reach varied mid-game contacts
            int numMoves = rng.Next(2, 7);
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
