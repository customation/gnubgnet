// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2019 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Ported from eval.c (EvalEfficiency)

namespace GnuBgNet.Evaluation;

/// <summary>
/// Position-class-specific cube efficiency computation.
/// Port of EvalEfficiency() and its coefficients from eval.c.
/// </summary>
public static class CubeEfficiency
{
    // Coefficients indexed by [ply >= 2 ? 0 : 1]
    private static readonly float _tsCubeX = 0.6f;
    private static readonly float _osCubeX = 0.6f;
    private static readonly float[] _raceFactorX = [0.00125f, 0.00250f];
    private static readonly float[] _raceCoefficientX = [0.55f, 0.60f];
    private static readonly float[] _raceMax = [0.7f, 0.8f];
    private static readonly float[] _raceMin = [0.6f, 0.6f];
    private static readonly float[] _crashedX = [0.68f, 0.76f];
    private static readonly float[] _contactX = [0.68f, 0.76f];

    /// <summary>
    /// Compute cube efficiency for a position.
    /// Port of EvalEfficiency() from eval.c.
    /// </summary>
    public static float Compute(Board board, PositionClass pc, int ply)
    {
        int i = ply >= 2 ? 0 : 1;

        switch (pc)
        {
            case PositionClass.Over:
                return 0.0f;

            case PositionClass.Hypergammon1:
            case PositionClass.Hypergammon2:
            case PositionClass.Hypergammon3:
                return 0.60f;

            case PositionClass.Bearoff1:
            case PositionClass.BearoffOneSided:
                return _osCubeX;

            case PositionClass.Race:
                var (playerPips, _) = PipCount.Count(board);
                float eff = playerPips * _raceFactorX[i] + _raceCoefficientX[i];
                return Math.Clamp(eff, _raceMin[i], _raceMax[i]);

            case PositionClass.Contact:
                return _contactX[i];

            case PositionClass.Crashed:
                return _crashedX[i];

            case PositionClass.Bearoff2:
            case PositionClass.BearoffTwoSided:
                return _tsCubeX;

            default:
                return 0.68f;
        }
    }
}
