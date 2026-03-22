// SPDX-License-Identifier: GPL-3.0-or-later

using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;

var config = DefaultConfig.Instance
    .WithOptions(ConfigOptions.DisableOptimizationsValidator)
    .AddJob(Job.ShortRun);

BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args, config);
