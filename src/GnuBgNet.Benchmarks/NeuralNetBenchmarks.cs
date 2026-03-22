// SPDX-License-Identifier: GPL-3.0-or-later

using BenchmarkDotNet.Attributes;
using GnuBgNet;
using GnuBgNet.NeuralNet;
using GnuBgNet.Optimized;

namespace GnuBgNet.Benchmarks;

[MemoryDiagnoser]
public class NeuralNetBenchmarks
{
    private NeuralNetwork _originalContact = null!;
    private SimdNeuralNetwork _simdContact = null!;
    private NeuralNetwork _originalRace = null!;
    private SimdNeuralNetwork _simdRace = null!;
    private float[] _contactInput = null!;
    private float[] _raceInput = null!;

    [GlobalSetup]
    public void Setup()
    {
        var nets = BenchmarkSetup.LoadNetworks();

        _originalContact = (NeuralNetwork)nets.Contact;
        _simdContact = new SimdNeuralNetwork(_originalContact);
        _originalRace = (NeuralNetwork)nets.Race;
        _simdRace = new SimdNeuralNetwork(_originalRace);

        _contactInput = new float[Constants.NumContactInputs];
        var board = BenchmarkSetup.CreateOpeningBoard();
        InputCalculator.CalculateContactInputs(board, _contactInput);

        _raceInput = new float[Constants.NumRaceInputs];
        var raceBoard = BenchmarkSetup.CreateRaceBoard();
        InputCalculator.CalculateRaceInputs(raceBoard, _raceInput);
    }

    [Benchmark(Baseline = true)]
    public void Original_Contact()
    {
        Span<float> output = stackalloc float[Constants.NumOutputs];
        _originalContact.Evaluate(_contactInput, output);
    }

    [Benchmark]
    public void Simd_Contact()
    {
        Span<float> output = stackalloc float[Constants.NumOutputs];
        _simdContact.Evaluate(_contactInput, output);
    }

    [Benchmark]
    public void Original_Race()
    {
        Span<float> output = stackalloc float[Constants.NumOutputs];
        _originalRace.Evaluate(_raceInput, output);
    }

    [Benchmark]
    public void Simd_Race()
    {
        Span<float> output = stackalloc float[Constants.NumOutputs];
        _simdRace.Evaluate(_raceInput, output);
    }
}
