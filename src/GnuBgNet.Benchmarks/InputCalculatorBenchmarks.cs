// SPDX-License-Identifier: GPL-3.0-or-later

using BenchmarkDotNet.Attributes;
using GnuBgNet;
using GnuBgNet.NeuralNet;
using GnuBgNet.Optimized;

namespace GnuBgNet.Benchmarks;

[MemoryDiagnoser]
public class InputCalculatorBenchmarks
{
    private Board _openingBoard = null!;
    private Board _contactBoard = null!;
    private Board _sparseBoard = null!;

    [GlobalSetup]
    public void Setup()
    {
        _openingBoard = BenchmarkSetup.CreateOpeningBoard();
        _contactBoard = BenchmarkSetup.CreateContactBoard();

        // A board with very few occupied points (best case for sparse)
        _sparseBoard = new Board();
        _sparseBoard.Player[0] = 10;
        _sparseBoard.Player[5] = 5;
        _sparseBoard.Opponent[0] = 10;
        _sparseBoard.Opponent[5] = 5;
    }

    [Benchmark(Baseline = true)]
    public void Original_Opening()
    {
        Span<float> input = stackalloc float[200];
        InputCalculator.BaseInputs(_openingBoard, input);
    }

    [Benchmark]
    public void Sparse_Opening()
    {
        Span<float> input = stackalloc float[200];
        SparseInputCalculator.Instance.BaseInputs(_openingBoard, input);
    }

    [Benchmark]
    public void Original_Contact()
    {
        Span<float> input = stackalloc float[200];
        InputCalculator.BaseInputs(_contactBoard, input);
    }

    [Benchmark]
    public void Sparse_Contact()
    {
        Span<float> input = stackalloc float[200];
        SparseInputCalculator.Instance.BaseInputs(_contactBoard, input);
    }

    [Benchmark]
    public void Original_Sparse()
    {
        Span<float> input = stackalloc float[200];
        InputCalculator.BaseInputs(_sparseBoard, input);
    }

    [Benchmark]
    public void Sparse_Sparse()
    {
        Span<float> input = stackalloc float[200];
        SparseInputCalculator.Instance.BaseInputs(_sparseBoard, input);
    }
}
