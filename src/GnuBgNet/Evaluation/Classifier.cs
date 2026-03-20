// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2019 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Ported from eval.c (ClassifyPosition)

namespace GnuBgNet.Evaluation;

/// <summary>
/// Classifies a board position into one of the evaluation categories.
/// Port of ClassifyPosition() from eval.c.
/// </summary>
public static class Classifier
{
    /// <summary>
    /// Determine the position class for evaluation dispatch.
    /// </summary>
    public static PositionClass Classify(Board board, Evaluator? evaluator = null)
    {
        int nOppBack, nBack;

        // Find back checker for opponent (anBoard[0])
        for (nOppBack = 24; nOppBack >= 0; --nOppBack)
            if (board.Opponent[nOppBack] > 0) break;

        // Find back checker for player (anBoard[1])
        for (nBack = 24; nBack >= 0; --nBack)
            if (board.Player[nBack] > 0) break;

        if (nBack < 0 || nOppBack < 0)
            return PositionClass.Over;

        // Standard backgammon only (no hypergammon variants)
        if (nBack + nOppBack > 22)
        {
            // Contact position — check if crashed
            const uint N = 6;
            for (int side = 0; side < 2; ++side)
            {
                uint[] b = side == 0 ? board.Opponent : board.Player;
                uint tot = 0;
                for (int i = 0; i < 25; ++i)
                    tot += b[i];

                if (tot <= N)
                    return PositionClass.Crashed;

                if (b[0] > 1)
                {
                    if (tot <= N + b[0])
                        return PositionClass.Crashed;
                    if (1 + tot - (b[0] + b[1]) <= N && b[1] > 1)
                        return PositionClass.Crashed;
                }
                else
                {
                    if (tot <= N + (b[1] - 1))
                        return PositionClass.Crashed;
                }
            }
            return PositionClass.Contact;
        }

        // Race / bearoff — check bearoff databases
        if (evaluator != null)
        {
            if (evaluator.HasTwoSidedBearoff && IsBearoff(board, evaluator.TwoSidedBearoff!))
                return PositionClass.BearoffTwoSided;

            if (evaluator.HasOneSidedBearoff && IsBearoff(board, evaluator.OneSidedBearoff!))
                return PositionClass.BearoffOneSided;
        }

        return PositionClass.Race;
    }

    private static bool IsBearoff(Board board, Bearoff.BearoffDatabase db)
    {
        // Check both sides have all checkers within the bearoff range
        for (int side = 0; side < 2; side++)
        {
            uint[] b = side == 0 ? board.Opponent : board.Player;
            // Checkers must be on points 0..(nPoints-1) only
            for (int i = db.Points; i < 25; i++)
                if (b[i] > 0) return false;
        }
        return true;
    }
}
