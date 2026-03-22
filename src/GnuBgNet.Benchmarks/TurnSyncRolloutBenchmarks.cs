// SPDX-License-Identifier: GPL-3.0-or-later

using BenchmarkDotNet.Attributes;
using GnuBgNet;
using GnuBgNet.Evaluation;
using GnuBgNet.Gpu;
using GnuBgNet.NeuralNet;
using GnuBgNet.Optimized;
using GnuBgNet.Rollout;

namespace GnuBgNet.Benchmarks;

/// <summary>
/// Benchmarks comparing original sequential RolloutEngine vs TurnSynchronousRolloutEngine
/// which batches all evaluations across trials at each turn barrier.
/// Uses small trial counts to keep benchmark runtime reasonable.
/// </summary>
[MemoryDiagnoser]
public class TurnSyncRolloutBenchmarks
{
    private Evaluator _evaluator = null!;
    private RolloutEngine _originalRollout = null!;
    private TurnSynchronousRolloutEngine _turnSync = null!;
    private Board _contactBoard = null!;

    [Params(36, 144)]
    public int NumTrials { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var dataDir = BenchmarkSetup.FindDataDir();
        var nets = NetworkSet.LoadBinary(Path.Combine(dataDir, "gnubg.wd"));

        _evaluator = new Evaluator(nets);
        _originalRollout = new RolloutEngine(_evaluator);

        // TurnSync with SIMD batch networks
        var contactNet = new SimdBatchNeuralNetwork((NeuralNetwork)nets.Contact);
        var raceNet = new SimdBatchNeuralNetwork((NeuralNetwork)nets.Race);
        var crashedNet = new SimdBatchNeuralNetwork((NeuralNetwork)nets.Crashed);

        _turnSync = new TurnSynchronousRolloutEngine(
            _evaluator,
            UndoStackMoveGenerator.Instance,
            SparseInputCalculator.Instance,
            contactNet, raceNet, crashedNet);

        _contactBoard = BenchmarkSetup.CreateContactBoard();
    }

    [Benchmark(Baseline = true)]
    public RolloutResult Original_Rollout()
    {
        var settings = new RolloutSettings
        {
            Trials = (uint)NumTrials,
            ChequerPlies = 0,
            Truncate = true,
            TruncatePlies = 10,
            Rotate = false,
            Seed = 42,
        };
        return _originalRollout.Rollout(_contactBoard, settings);
    }

    [Benchmark]
    public float[] TurnSync_Rollout()
    {
        return _turnSync.RunRollout(_contactBoard, NumTrials, maxTurns: 400, seed: 42);
    }
}
