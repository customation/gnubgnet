// SPDX-License-Identifier: GPL-3.0-or-later

using GnuBgNet.MatchEquity;

namespace GnuBgNet.Evaluation;

/// <summary>
/// Interface for position evaluation.
/// Allows plugging in a completely different evaluation engine
/// (e.g. ONNX-based, Monte Carlo only, tabular, etc.).
/// </summary>
public interface IPositionEvaluator
{
    void EvaluatePosition(Board board, Span<float> output);

    void EvaluatePositionPlied(Board board, Span<float> output, int nPlies, bool usePrune = true);

    void EvaluatePositionPlied(Board board, Span<float> output, int nPlies, bool usePrune,
        EvalContext? ec, CubeInfo? ci = null);

    /// <summary>
    /// Classify a board position for evaluation dispatch.
    /// </summary>
    PositionClass ClassifyPosition(Board board);

    /// <summary>
    /// Evaluate a position with a known position class (avoids re-classification).
    /// </summary>
    void EvaluatePositionByClass(Board board, Span<float> output, PositionClass pc);

    void FindnSaveBestMoves(MoveList ml, Board board, int nDice0, int nDice1,
        EvalContext ec, MoveFilter[,]? moveFilters = null);

    void GeneralEvaluationEPlied(Board board, Span<float> arOutput,
        CubeInfo ci, EvalContext ec, int nPlies);

    bool EvaluatePerfectCubeful(Board board, Span<float> arEquity);

    int GameStatus(Board board);

    void FlushCaches();
}
