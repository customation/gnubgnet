// SPDX-License-Identifier: GPL-3.0-or-later

using Xunit;
using GnuBgNet;
using GnuBgNet.NeuralNet;
using SystemRandom = System.Random;

namespace GnuBgNet.Tests;

public class SigmoidTests
{
    [Fact]
    public void Sigmoid_Zero_ReturnsHalf()
    {
        // sigmoid(0) = 1 / (1 + e^0) = 0.5
        float result = Sigmoid.Evaluate(0.0f);
        Assert.InRange(result, 0.499f, 0.501f);
    }

    [Fact]
    public void Sigmoid_LargePositive_ReturnsNearZero()
    {
        float result = Sigmoid.Evaluate(20.0f);
        Assert.InRange(result, 0.0f, 0.001f);
    }

    [Fact]
    public void Sigmoid_LargeNegative_ReturnsNearOne()
    {
        float result = Sigmoid.Evaluate(-20.0f);
        Assert.InRange(result, 0.999f, 1.0f);
    }

    [Fact]
    public void Sigmoid_MidRange_MatchesMathSigmoid()
    {
        // Test several values in the common range
        float[] testValues = [0.5f, 1.0f, 2.0f, 3.0f, 5.0f, -0.5f, -1.0f, -3.0f, -5.0f];
        foreach (float x in testValues)
        {
            float expected = 1.0f / (1.0f + MathF.Exp(x));
            float actual = Sigmoid.Evaluate(x);
            // Lookup table approximation should be within 1% for common values
            Assert.InRange(actual, expected - 0.01f, expected + 0.01f);
        }
    }
}

public class NeuralNetworkTests
{
    [Fact]
    public void Evaluate_TrivialNetwork_ProducesOutput()
    {
        // Create a tiny network with known weights
        int cInput = 4, cHidden = 2, cOutput = 1;
        var nn = new NeuralNetwork(
            cInput, cHidden, cOutput,
            Constants.BetaHidden, Constants.BetaOutput,
            trained: true,
            hiddenWeight: new float[cInput * cHidden], // all zeros
            outputWeight: new float[cHidden * cOutput], // all zeros
            hiddenThreshold: new float[cHidden],
            outputThreshold: new float[cOutput]);

        float[] input = [1.0f, 0.0f, 0.0f, 0.0f];
        float[] output = new float[cOutput];

        nn.Evaluate(input, output);

        // With all-zero weights and thresholds:
        // hidden[i] = sigmoid(-beta * 0) = sigmoid(0) ≈ 0.5
        // output = sigmoid(-beta * (0.5 * 0 + 0.5 * 0 + 0)) = sigmoid(0) ≈ 0.5
        Assert.InRange(output[0], 0.45f, 0.55f);
    }

    [Fact]
    public void Evaluate_Deterministic()
    {
        // Same inputs should always produce same outputs
        int cInput = 10, cHidden = 4, cOutput = 2;
        var rng = new SystemRandom(42);
        var hiddenW = Enumerable.Range(0, cInput * cHidden).Select(_ => (float)(rng.NextDouble() - 0.5)).ToArray();
        var outputW = Enumerable.Range(0, cHidden * cOutput).Select(_ => (float)(rng.NextDouble() - 0.5)).ToArray();
        var hiddenT = Enumerable.Range(0, cHidden).Select(_ => (float)(rng.NextDouble() - 0.5)).ToArray();
        var outputT = Enumerable.Range(0, cOutput).Select(_ => (float)(rng.NextDouble() - 0.5)).ToArray();

        var nn = new NeuralNetwork(cInput, cHidden, cOutput, 0.1f, 1.0f, true,
            hiddenW, outputW, hiddenT, outputT);

        float[] input = [1f, 0f, 1f, 0f, 0.5f, 0f, 0f, 1f, 0f, 0.25f];
        float[] output1 = new float[cOutput];
        float[] output2 = new float[cOutput];

        nn.Evaluate(input, output1);
        nn.Evaluate(input, output2);

        for (int i = 0; i < cOutput; i++)
            Assert.Equal(output1[i], output2[i]);
    }

    [Fact]
    public void Evaluate_OutputsInRange01()
    {
        // Sigmoid outputs are always in (0, 1)
        int cInput = 10, cHidden = 8, cOutput = 5;
        var rng = new SystemRandom(123);
        var hiddenW = Enumerable.Range(0, cInput * cHidden).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        var outputW = Enumerable.Range(0, cHidden * cOutput).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        var hiddenT = Enumerable.Range(0, cHidden).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        var outputT = Enumerable.Range(0, cOutput).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();

        var nn = new NeuralNetwork(cInput, cHidden, cOutput, 0.1f, 1.0f, true,
            hiddenW, outputW, hiddenT, outputT);

        float[] input = Enumerable.Range(0, cInput).Select(_ => (float)rng.NextDouble()).ToArray();
        float[] output = new float[cOutput];

        nn.Evaluate(input, output);

        for (int i = 0; i < cOutput; i++)
        {
            Assert.True(output[i] > 0.0f, $"output[{i}] = {output[i]} should be > 0");
            Assert.True(output[i] < 1.0f, $"output[{i}] = {output[i]} should be < 1");
        }
    }
}

public class WeightLoaderTests
{
    private static string? FindWeightsFile()
    {
        // Look for gnubg.wd in several likely locations
        string[] candidates =
        [
            Path.Combine(Environment.GetEnvironmentVariable("GNUBG_DATA_DIR") ?? "", "gnubg.wd"),
            @"C:\git\github\customation\gnubg\gnubg.wd",
            @"C:\git\github\customation\gnubgnet\data\gnubg.wd",
        ];

        return candidates.FirstOrDefault(File.Exists);
    }

    [Fact]
    public void LoadBinary_ValidFile_LoadsAllNetworks()
    {
        var path = FindWeightsFile();
        if (path == null)
        {
            // Skip if weights file not available
            return;
        }

        var nets = NetworkSet.LoadBinary(path);

        // Validate Contact network
        Assert.Equal(Constants.NumContactInputs, nets.Contact.InputCount); // 250
        Assert.Equal(Constants.NumOutputs, nets.Contact.OutputCount);       // 5
        Assert.True(nets.Contact.HiddenCount > 0);
        Assert.True(nets.Contact.Trained);

        // Validate Race network
        Assert.Equal(Constants.NumRaceInputs, nets.Race.InputCount); // 214
        Assert.Equal(Constants.NumOutputs, nets.Race.OutputCount);

        // Validate Crashed network
        Assert.Equal(Constants.NumCrashedInputs, nets.Crashed.InputCount); // 250
        Assert.Equal(Constants.NumOutputs, nets.Crashed.OutputCount);

        // Validate pruning networks
        Assert.Equal(Constants.NumPruningInputs, nets.PruneContact.InputCount); // 200
        Assert.Equal(Constants.NumPruningInputs, nets.PruneCrashed.InputCount);
        Assert.Equal(Constants.NumPruningInputs, nets.PruneRace.InputCount);
    }

    [Fact]
    public void LoadBinary_ContactNet_ProducesReasonableOutput()
    {
        var path = FindWeightsFile();
        if (path == null) return;

        var nets = NetworkSet.LoadBinary(path);

        // Create base inputs for the opening position
        var board = Board.Opening();
        float[] inputs = new float[Constants.NumContactInputs];

        // Just use base inputs (200 floats) + zeroes for contact features
        // This won't be a perfect evaluation but should produce values in [0, 1]
        InputCalculator.BaseInputs(board, inputs);

        float[] output = new float[Constants.NumOutputs];
        nets.Contact.Evaluate(inputs, output);

        // All outputs should be valid probabilities
        for (int i = 0; i < Constants.NumOutputs; i++)
        {
            Assert.True(output[i] >= 0.0f && output[i] <= 1.0f,
                $"Contact output[{i}] = {output[i]} out of range");
        }

        // Win probability should be a valid value (contact features are zeroed,
        // so the output may be extreme but should still be in (0, 1))
        Assert.True(output[Constants.OutputWin] > 0.0f && output[Constants.OutputWin] < 1.0f,
            $"Win probability {output[Constants.OutputWin]} should be in (0, 1)");
    }
}

public class InputCalculatorTests
{
    [Fact]
    public void BaseInputs_OpeningPosition_Produces200Floats()
    {
        var board = Board.Opening();
        float[] inputs = new float[200];

        InputCalculator.BaseInputs(board, inputs);

        // Opening: Opponent[5]=5, so arInput[5*4+0] = inpvec[5][0] = 0
        // inpvec[5] = {0, 0, 1, 1.0}
        Assert.Equal(0f, inputs[5 * 4 + 0]);
        Assert.Equal(0f, inputs[5 * 4 + 1]);
        Assert.Equal(1f, inputs[5 * 4 + 2]);
        Assert.Equal(1.0f, inputs[5 * 4 + 3]);

        // Point with 3 checkers: Opponent[7]=3, inpvec[3] = {0, 0, 1, 0}
        Assert.Equal(0f, inputs[7 * 4 + 0]);
        Assert.Equal(0f, inputs[7 * 4 + 1]);
        Assert.Equal(1f, inputs[7 * 4 + 2]);
        Assert.Equal(0f, inputs[7 * 4 + 3]);

        // Point with 2 checkers: Opponent[23]=2, inpvec[2] = {0, 1, 0, 0}
        Assert.Equal(0f, inputs[23 * 4 + 0]);
        Assert.Equal(1f, inputs[23 * 4 + 1]);
        Assert.Equal(0f, inputs[23 * 4 + 2]);
        Assert.Equal(0f, inputs[23 * 4 + 3]);

        // Empty point: Opponent[0]=0, inpvec[0] = {0, 0, 0, 0}
        Assert.Equal(0f, inputs[0]);
        Assert.Equal(0f, inputs[1]);
        Assert.Equal(0f, inputs[2]);
        Assert.Equal(0f, inputs[3]);
    }

    [Fact]
    public void CalculateRaceInputs_AllOnFirst_HasMenOff()
    {
        // All 15 checkers on point 0 (1-point): menOff = 0
        var board = new Board();
        board.Player[0] = 15;
        board.Opponent[0] = 15;

        float[] inputs = new float[Constants.NumRaceInputs];
        InputCalculator.CalculateRaceInputs(board, inputs);

        // For opponent (side 0): menOff = 15 - 15 = 0
        // All men-off indicators should be 0
        for (int k = 0; k < 14; k++)
            Assert.Equal(0f, inputs[92 + k]);
    }

    [Fact]
    public void CalculateRaceInputs_SomeOff_CorrectIndicator()
    {
        // 12 checkers on board, 3 off
        var board = new Board();
        board.Player[0] = 12;
        board.Opponent[0] = 12;

        float[] inputs = new float[Constants.NumRaceInputs];
        InputCalculator.CalculateRaceInputs(board, inputs);

        // Opponent side (0): menOff = 15 - 12 = 3, so RI_OFF + 2 should be 1.0
        Assert.Equal(0f, inputs[92 + 0]); // menOff == 1?
        Assert.Equal(0f, inputs[92 + 1]); // menOff == 2?
        Assert.Equal(1f, inputs[92 + 2]); // menOff == 3? YES
        Assert.Equal(0f, inputs[92 + 3]); // menOff == 4?
    }
}
