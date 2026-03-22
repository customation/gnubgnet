using Xunit;
using Xunit.Abstractions;
using GnuBgNet;
using GnuBgNet.Encoding;
using GammonBase.Gnubg;

namespace GnuBgNet.Tests;

public class RolloutQuickTest : IClassFixture<ParityFixture>
{
    private readonly ParityFixture _fixture;
    private readonly ITestOutputHelper _output;

    public RolloutQuickTest(ParityFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    [Theory]
    [InlineData("AAAA/xgAAAAAAMA", "Strong bearoff")]
    [InlineData("4HPwATDgc/ABMA", "Opening")]
    [InlineData("sG2wATDgc/ABMA", "After 31")]
    public void Rollout_CubelessProbabilities(string positionId, string label)
    {
        if (_fixture.SkipReason != null) { Assert.Fail(_fixture.SkipReason); return; }

        var managed = _fixture.ManagedEngine!;
        var native = _fixture.NativeContext!;

        var nativeSettings = new GammonBase.Gnubg.RolloutSettings
        {
            Trials = 1296,
            Cubeful = false,
            VarianceReduction = true,
            ChequerPlies = 0,
            CubePlies = 0,
            Seed = 42,
            Truncate = true,
            TruncatePlies = 10,
        };
        var managedSettings = new GnuBgNet.RolloutSettings
        {
            Trials = 1296,
            Cubeful = false,
            VarianceReduction = true,
            ChequerPlies = 0,
            CubePlies = 0,
            Seed = 42,
            Truncate = true,
            TruncatePlies = 10,
            Rotate = false,
        };

        var nativeResult = native.RolloutPosition(positionId, settings: nativeSettings);
        var managedResult = managed.RolloutPosition(positionId, managedSettings);

        _output.WriteLine($"[{label}] Native  Win: {nativeResult.WinProbability:F4}  WG: {nativeResult.WinGammonProbability:F4}  LG: {nativeResult.LoseGammonProbability:F4}  Eq: {nativeResult.CubelessEquity:F4}");
        _output.WriteLine($"[{label}] Managed Win: {managedResult.WinProbability:F4}  WG: {managedResult.WinGammonProbability:F4}  LG: {managedResult.LoseGammonProbability:F4}  Eq: {managedResult.CubelessEquity:F4}");

        Assert.Equal(nativeResult.WinProbability, managedResult.WinProbability, 0.03);
        Assert.Equal(nativeResult.CubelessEquity, managedResult.CubelessEquity, 0.05);
    }

    [Theory]
    [InlineData("AAAA/xgAAAAAAMA", "Strong bearoff")]
    [InlineData("4HPwATDgc/ABMA", "Opening")]
    [InlineData("sG2wATDgc/ABMA", "After 31")]
    public void Rollout_CubefulEquity(string positionId, string label)
    {
        if (_fixture.SkipReason != null) { Assert.Fail(_fixture.SkipReason); return; }

        var managed = _fixture.ManagedEngine!;
        var native = _fixture.NativeContext!;

        var nativeSettings = new GammonBase.Gnubg.RolloutSettings
        {
            Trials = 1296,
            Cubeful = true,
            VarianceReduction = true,
            ChequerPlies = 0,
            CubePlies = 2,
            Seed = 42,
            Truncate = true,
            TruncatePlies = 10,
        };
        var managedSettings = new GnuBgNet.RolloutSettings
        {
            Trials = 1296,
            Cubeful = true,
            VarianceReduction = true,
            ChequerPlies = 0,
            CubePlies = 2,
            Seed = 42,
            Truncate = true,
            TruncatePlies = 10,
            Rotate = false,
        };

        var nativeResult = native.RolloutPosition(positionId, settings: nativeSettings);
        var managedResult = managed.RolloutPosition(positionId, managedSettings);

        _output.WriteLine($"[{label}] Native  CfEq: {nativeResult.CubefulEquity:F4}  ClEq: {nativeResult.CubelessEquity:F4}");
        _output.WriteLine($"[{label}] Managed CfEq: {managedResult.CubefulEquity:F4}  ClEq: {managedResult.CubelessEquity:F4}");

        Assert.Equal(nativeResult.CubefulEquity, managedResult.CubefulEquity, 0.05);
        Assert.Equal(nativeResult.CubelessEquity, managedResult.CubelessEquity, 0.05);
    }
}
