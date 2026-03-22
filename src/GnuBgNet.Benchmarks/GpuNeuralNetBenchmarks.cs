// SPDX-License-Identifier: GPL-3.0-or-later

using BenchmarkDotNet.Attributes;
using GnuBgNet;
using GnuBgNet.Gpu;
using GnuBgNet.NeuralNet;
using GnuBgNet.Optimized;

namespace GnuBgNet.Benchmarks;

/// <summary>
/// Benchmarks comparing CPU (original + SIMD) vs GPU (ONNX+DirectML) neural network evaluation.
/// Tests single eval and batched eval at various batch sizes to find the GPU crossover point.
/// </summary>
[MemoryDiagnoser]
public class GpuNeuralNetBenchmarks
{
    private NeuralNetwork _cpuOriginal = null!;
    private SimdNeuralNetwork _cpuSimd = null!;
    private GpuNeuralNetwork _gpu = null!;
    private float[] _singleInput = null!;
    private float[][] _batchInputs = null!;
    private float[] _flatBatchInput = null!;

    [Params(1, 10, 50, 200)]
    public int BatchSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var nets = BenchmarkSetup.LoadNetworks();
        _cpuOriginal = (NeuralNetwork)nets.Contact;
        _cpuSimd = new SimdNeuralNetwork(_cpuOriginal);
        _gpu = new GpuNeuralNetwork(_cpuOriginal);

        // Build varied inputs from different board positions
        var rng = new System.Random(42);
        int inputSize = _cpuOriginal.InputCount;
        _singleInput = new float[inputSize];
        InputCalculator.CalculateContactInputs(BenchmarkSetup.CreateOpeningBoard(), _singleInput);

        _batchInputs = new float[BatchSize][];
        _flatBatchInput = new float[BatchSize * inputSize];

        for (int b = 0; b < BatchSize; b++)
        {
            _batchInputs[b] = new float[inputSize];
            // Create varied positions by perturbing the opening
            var board = BenchmarkSetup.CreateOpeningBoard();
            // Slight random variation
            int from = rng.Next(0, 24);
            int to = rng.Next(0, 24);
            if (board.Player[from] > 0 && from != to)
            {
                board.Player[from]--;
                board.Player[to]++;
            }
            InputCalculator.CalculateContactInputs(board, _batchInputs[b]);
            Array.Copy(_batchInputs[b], 0, _flatBatchInput, b * inputSize, inputSize);
        }
    }

    [GlobalCleanup]
    public void Cleanup() => _gpu?.Dispose();

    [Benchmark(Baseline = true)]
    public void Cpu_Original_Batch()
    {
        Span<float> output = stackalloc float[Constants.NumOutputs];
        for (int b = 0; b < BatchSize; b++)
            _cpuOriginal.Evaluate(_batchInputs[b], output);
    }

    [Benchmark]
    public void Cpu_Simd_Batch()
    {
        Span<float> output = stackalloc float[Constants.NumOutputs];
        for (int b = 0; b < BatchSize; b++)
            _cpuSimd.Evaluate(_batchInputs[b], output);
    }

    [Benchmark]
    public void Gpu_Sequential_Batch()
    {
        Span<float> output = stackalloc float[Constants.NumOutputs];
        for (int b = 0; b < BatchSize; b++)
            _gpu.Evaluate(_batchInputs[b], output);
    }

    [Benchmark]
    public void Gpu_Batched()
    {
        Span<float> output = new float[BatchSize * Constants.NumOutputs];
        _gpu.EvaluateBatch(_flatBatchInput, output, BatchSize);
    }
}
