// Copyright (C) 1999-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2002-2017 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later

namespace GnuBgNet.NeuralNet;

public enum NNEvalType
{
    None,
    Save,
    FromBase,
}

public enum NNStateType
{
    None = -1,
    Incremental = 0,
    Done = 1,
}

/// <summary>
/// Incremental evaluation state for reusing hidden layer across moves.
/// Port of NNState from neuralnet.h.
/// </summary>
public sealed class NNState
{
    public NNStateType State { get; set; } = NNStateType.None;
    public float[]? SavedBase { get; set; }
    public float[]? SavedIBase { get; set; }
    public int SavedIBaseCount { get; set; }

    public NNState(int maxHidden, int maxInput)
    {
        SavedBase = new float[maxHidden];
        SavedIBase = new float[maxInput];
    }

    public NNEvalType GetAction()
    {
        switch (State)
        {
            case NNStateType.None:
                return NNEvalType.None;
            case NNStateType.Incremental:
                State = NNStateType.Done;
                return NNEvalType.Save;
            case NNStateType.Done:
                return NNEvalType.FromBase;
            default:
                return NNEvalType.None;
        }
    }
}
