// SPDX-License-Identifier: GPL-3.0-or-later

using BenchmarkDotNet.Attributes;
using GnuBgNet;
using GnuBgNet.MoveGeneration;
using GnuBgNet.Optimized;

namespace GnuBgNet.Benchmarks;

[MemoryDiagnoser]
public class MoveGeneratorBenchmarks
{
    private Board _openingBoard;
    private Board _contactBoard;
    private readonly DefaultMoveGenerator _original = DefaultMoveGenerator.Instance;
    private readonly UndoStackMoveGenerator _undoStack = UndoStackMoveGenerator.Instance;

    [GlobalSetup]
    public void Setup()
    {
        _openingBoard = BenchmarkSetup.CreateOpeningBoard();
        _contactBoard = BenchmarkSetup.CreateContactBoard();
    }

    [Benchmark(Baseline = true)]
    public MoveList Original_Opening_31()
        => _original.GenerateMoves(_openingBoard, 3, 1);

    [Benchmark]
    public MoveList UndoStack_Opening_31()
        => _undoStack.GenerateMoves(_openingBoard, 3, 1);

    [Benchmark]
    public MoveList Original_Opening_Doubles()
        => _original.GenerateMoves(_openingBoard, 4, 4);

    [Benchmark]
    public MoveList UndoStack_Opening_Doubles()
        => _undoStack.GenerateMoves(_openingBoard, 4, 4);

    [Benchmark]
    public MoveList Original_Contact_52()
        => _original.GenerateMoves(_contactBoard, 5, 2);

    [Benchmark]
    public MoveList UndoStack_Contact_52()
        => _undoStack.GenerateMoves(_contactBoard, 5, 2);
}
