// SPDX-License-Identifier: GPL-3.0-or-later

using Xunit;
using GnuBgNet;
using GnuBgNet.Encoding;

namespace GnuBgNet.Tests;

public class MatchIdTests
{
    [Fact]
    public void Encode_Decode_RoundTrip_MoneyGame()
    {
        // Money game: cube 1, centered, no dice, playing
        var encoded = MatchId.Encode(
            die1: 0, die2: 0,
            turn: 0, resigned: 0, doubled: false,
            move: 0, cubeOwner: -1, crawford: false,
            matchTo: 0, score0: 0, score1: 0,
            cube: 1, jacoby: false, gs: GameState.Playing);

        var decoded = MatchId.Decode(encoded);
        Assert.NotNull(decoded);

        Assert.Equal(1, decoded.Cube);
        Assert.Equal(-1, decoded.CubeOwner);
        Assert.Equal(0, decoded.Move);
        Assert.False(decoded.Crawford);
        Assert.Equal(GameState.Playing, decoded.GameState);
        Assert.Equal(0, decoded.Turn);
        Assert.False(decoded.Doubled);
        Assert.Equal(0, decoded.Resigned);
        Assert.Equal(0, decoded.Die1);
        Assert.Equal(0, decoded.Die2);
        Assert.Equal(0, decoded.MatchTo);
        Assert.Equal(0, decoded.Score0);
        Assert.Equal(0, decoded.Score1);
        Assert.False(decoded.Jacoby);
    }

    [Fact]
    public void Encode_Decode_RoundTrip_MatchPlay()
    {
        // 5-point match, 2-1 with dice 3-1, cube at 2 owned by player 0
        var encoded = MatchId.Encode(
            die1: 3, die2: 1,
            turn: 1, resigned: 0, doubled: false,
            move: 1, cubeOwner: 0, crawford: false,
            matchTo: 5, score0: 2, score1: 1,
            cube: 2, jacoby: false, gs: GameState.Playing);

        var decoded = MatchId.Decode(encoded);
        Assert.NotNull(decoded);

        Assert.Equal(2, decoded.Cube);
        Assert.Equal(0, decoded.CubeOwner);
        Assert.Equal(1, decoded.Move);
        Assert.False(decoded.Crawford);
        Assert.Equal(GameState.Playing, decoded.GameState);
        Assert.Equal(1, decoded.Turn);
        Assert.False(decoded.Doubled);
        Assert.Equal(0, decoded.Resigned);
        // Dice are stored high-first, so 3 and 1
        Assert.True(
            (decoded.Die1 == 3 && decoded.Die2 == 1) ||
            (decoded.Die1 == 1 && decoded.Die2 == 3));
        Assert.Equal(5, decoded.MatchTo);
        Assert.Equal(2, decoded.Score0);
        Assert.Equal(1, decoded.Score1);
    }

    [Fact]
    public void Encode_MatchIdLength_Is12()
    {
        var encoded = MatchId.Encode(
            die1: 0, die2: 0,
            turn: 0, resigned: 0, doubled: false,
            move: 0, cubeOwner: -1, crawford: false,
            matchTo: 0, score0: 0, score1: 0,
            cube: 1, jacoby: false, gs: GameState.None);

        Assert.Equal(12, encoded.Length);
    }

    [Fact]
    public void Decode_KnownMatchId_cAkAAAAAAAAA()
    {
        // "cAkAAAAAAAAA" is a known match ID from gnubg
        // This represents a specific match state that should decode without error
        var decoded = MatchId.Decode("cAkAAAAAAAAA");
        Assert.NotNull(decoded);
    }

    [Fact]
    public void Encode_Decode_RoundTrip_CrawfordGame()
    {
        var encoded = MatchId.Encode(
            die1: 6, die2: 5,
            turn: 0, resigned: 0, doubled: false,
            move: 0, cubeOwner: -1, crawford: true,
            matchTo: 7, score0: 6, score1: 4,
            cube: 1, jacoby: false, gs: GameState.Playing);

        var decoded = MatchId.Decode(encoded);
        Assert.NotNull(decoded);
        Assert.True(decoded.Crawford);
        Assert.Equal(7, decoded.MatchTo);
        Assert.Equal(6, decoded.Score0);
        Assert.Equal(4, decoded.Score1);
    }

    [Fact]
    public void Encode_Decode_RoundTrip_Jacoby()
    {
        var encoded = MatchId.Encode(
            die1: 4, die2: 2,
            turn: 1, resigned: 0, doubled: false,
            move: 1, cubeOwner: -1, crawford: false,
            matchTo: 0, score0: 0, score1: 0,
            cube: 1, jacoby: true, gs: GameState.Playing);

        var decoded = MatchId.Decode(encoded);
        Assert.NotNull(decoded);
        Assert.True(decoded.Jacoby);
        Assert.Equal(0, decoded.MatchTo); // Money game
    }

    [Fact]
    public void Encode_Decode_RoundTrip_LargeCube()
    {
        var encoded = MatchId.Encode(
            die1: 5, die2: 3,
            turn: 0, resigned: 0, doubled: false,
            move: 0, cubeOwner: 1, crawford: false,
            matchTo: 0, score0: 0, score1: 0,
            cube: 64, jacoby: false, gs: GameState.Playing);

        var decoded = MatchId.Decode(encoded);
        Assert.NotNull(decoded);
        Assert.Equal(64, decoded.Cube);
        Assert.Equal(1, decoded.CubeOwner);
    }

    [Fact]
    public void Decode_EmptyString_ReturnsNull()
    {
        Assert.Null(MatchId.Decode(""));
        Assert.Null(MatchId.Decode(null!));
    }

    [Fact]
    public void LogCube_Powers()
    {
        Assert.Equal(0, MatchId.LogCube(1));
        Assert.Equal(1, MatchId.LogCube(2));
        Assert.Equal(2, MatchId.LogCube(4));
        Assert.Equal(3, MatchId.LogCube(8));
        Assert.Equal(4, MatchId.LogCube(16));
        Assert.Equal(5, MatchId.LogCube(32));
        Assert.Equal(6, MatchId.LogCube(64));
    }

    [Fact]
    public void Encode_Decode_RoundTrip_Doubled()
    {
        var encoded = MatchId.Encode(
            die1: 0, die2: 0,
            turn: 0, resigned: 0, doubled: true,
            move: 1, cubeOwner: -1, crawford: false,
            matchTo: 0, score0: 0, score1: 0,
            cube: 1, jacoby: false, gs: GameState.Playing);

        var decoded = MatchId.Decode(encoded);
        Assert.NotNull(decoded);
        Assert.True(decoded.Doubled);
    }

    [Fact]
    public void Encode_Decode_RoundTrip_Resigned()
    {
        var encoded = MatchId.Encode(
            die1: 0, die2: 0,
            turn: 0, resigned: 2, doubled: false,
            move: 0, cubeOwner: -1, crawford: false,
            matchTo: 3, score0: 0, score1: 0,
            cube: 1, jacoby: false, gs: GameState.Resigned);

        var decoded = MatchId.Decode(encoded);
        Assert.NotNull(decoded);
        Assert.Equal(2, decoded.Resigned);
        Assert.Equal(GameState.Resigned, decoded.GameState);
    }
}
