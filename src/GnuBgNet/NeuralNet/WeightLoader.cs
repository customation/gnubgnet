// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2019 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Ported from neuralnet.c (NeuralNetLoadBinary / NeuralNetLoad)

namespace GnuBgNet.NeuralNet;

/// <summary>
/// The set of 6 neural networks loaded from a gnubg weight file.
/// </summary>
public sealed class NetworkSet
{
    public required NeuralNetwork Contact { get; init; }
    public required NeuralNetwork Race { get; init; }
    public required NeuralNetwork Crashed { get; init; }
    public required NeuralNetwork PruneContact { get; init; }
    public required NeuralNetwork PruneCrashed { get; init; }
    public required NeuralNetwork PruneRace { get; init; }

    /// <summary>
    /// Load all 6 networks from a binary .wd file.
    /// File format: magic(4) + version(4) + 6 sequential networks.
    /// </summary>
    public static NetworkSet LoadBinary(string path)
    {
        using var stream = File.OpenRead(path);
        return LoadBinary(stream);
    }

    /// <summary>
    /// Load all 6 networks from a binary stream (.wd format).
    /// </summary>
    public static NetworkSet LoadBinary(Stream stream)
    {
        using var reader = new BinaryReader(stream);

        // Read and validate magic number
        float magic = reader.ReadSingle();
        if (MathF.Abs(magic - Constants.WeightsMagicBinary) > 0.001f)
            throw new InvalidDataException($"Invalid weights file: magic {magic}, expected {Constants.WeightsMagicBinary}");

        float version = reader.ReadSingle();
        if (MathF.Abs(version - Constants.WeightsVersionBinary) > 0.001f)
            throw new InvalidDataException($"Invalid weights version: {version}, expected {Constants.WeightsVersionBinary}");

        // Load 6 networks in order: Contact, Race, Crashed, PruneContact, PruneCrashed, PruneRace
        var contact = LoadOneNetwork(reader);
        var race = LoadOneNetwork(reader);
        var crashed = LoadOneNetwork(reader);
        var pruneContact = LoadOneNetwork(reader);
        var pruneCrashed = LoadOneNetwork(reader);
        var pruneRace = LoadOneNetwork(reader);

        // Validate dimensions
        ValidateNetwork(contact, Constants.NumContactInputs, Constants.NumOutputs, "Contact");
        ValidateNetwork(crashed, Constants.NumCrashedInputs, Constants.NumOutputs, "Crashed");
        ValidateNetwork(race, Constants.NumRaceInputs, Constants.NumOutputs, "Race");
        ValidateNetwork(pruneContact, Constants.NumPruningInputs, Constants.NumOutputs, "PruneContact");
        ValidateNetwork(pruneCrashed, Constants.NumPruningInputs, Constants.NumOutputs, "PruneCrashed");
        ValidateNetwork(pruneRace, Constants.NumPruningInputs, Constants.NumOutputs, "PruneRace");

        return new NetworkSet
        {
            Contact = contact,
            Race = race,
            Crashed = crashed,
            PruneContact = pruneContact,
            PruneCrashed = pruneCrashed,
            PruneRace = pruneRace,
        };
    }

    private static NeuralNetwork LoadOneNetwork(BinaryReader reader)
    {
        int cInput = reader.ReadInt32();
        int cHidden = reader.ReadInt32();
        int cOutput = reader.ReadInt32();
        int nTrained = reader.ReadInt32();
        float betaHidden = reader.ReadSingle();
        float betaOutput = reader.ReadSingle();

        if (cInput < 1 || cHidden < 1 || cOutput < 1 || betaHidden <= 0f || betaOutput <= 0f)
            throw new InvalidDataException($"Invalid network dimensions: {cInput}×{cHidden}×{cOutput}");

        var hiddenWeight = ReadFloatArray(reader, cInput * cHidden);
        var outputWeight = ReadFloatArray(reader, cHidden * cOutput);
        var hiddenThreshold = ReadFloatArray(reader, cHidden);
        var outputThreshold = ReadFloatArray(reader, cOutput);

        return new NeuralNetwork(
            cInput, cHidden, cOutput,
            betaHidden, betaOutput,
            nTrained != 0,
            hiddenWeight, outputWeight,
            hiddenThreshold, outputThreshold);
    }

    private static float[] ReadFloatArray(BinaryReader reader, int count)
    {
        var arr = new float[count];
        var bytes = reader.ReadBytes(count * sizeof(float));
        if (bytes.Length < count * sizeof(float))
            throw new EndOfStreamException("Unexpected end of weights file");
        Buffer.BlockCopy(bytes, 0, arr, 0, bytes.Length);
        return arr;
    }

    private static void ValidateNetwork(NeuralNetwork nn, int expectedInputs, int expectedOutputs, string name)
    {
        if (nn.InputCount != expectedInputs)
            throw new InvalidDataException($"{name} network has {nn.InputCount} inputs, expected {expectedInputs}");
        if (nn.OutputCount != expectedOutputs)
            throw new InvalidDataException($"{name} network has {nn.OutputCount} outputs, expected {expectedOutputs}");
    }
}
