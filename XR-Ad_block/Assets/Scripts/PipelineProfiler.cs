// Pipeline Profiler — Centralized timing for the ad-block pipeline.
// Prints one summary table every few seconds via Debug.Log.
//
// VIEW IN TERMINAL:
//   adb logcat -s Unity | grep PIPELINE
//
// Or use the included view_profiler.sh script.
//
// USAGE IN OTHER SCRIPTS:
//   PipelineProfiler.Begin("StepName");
//   // ... work ...
//   PipelineProfiler.End("StepName");
//   PipelineProfiler.Set("Detections", count);

using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using UnityEngine;
using UnityEngine.Profiling;

using Debug = UnityEngine.Debug;

public static class PipelineProfiler
{
    private const float PrintIntervalSeconds = 3.0f;
    private const string Tag = "[PIPELINE]";

    // Active stopwatches (currently running steps)
    private static readonly Dictionary<string, Stopwatch> running = new Dictionary<string, Stopwatch>();

    // Completed step timings in ms (latest value per step)
    private static readonly Dictionary<string, double> completed = new Dictionary<string, double>();

    // Extra values (detection count, etc.)
    private static readonly Dictionary<string, string> values = new Dictionary<string, string>();

    // Ordered list of step names (preserves insertion order for table display)
    private static readonly List<string> stepOrder = new List<string>();

    private static float lastPrintTime = -PrintIntervalSeconds;

    /// <summary>
    /// Start timing a named step. Call End() with the same name when done.
    /// Stopwatch survives across coroutine yields (measures wall-clock time).
    /// </summary>
    public static void Begin(string stepName)
    {
        if (running.TryGetValue(stepName, out Stopwatch sw))
        {
            sw.Restart();
        }
        else
        {
            var newSw = Stopwatch.StartNew();
            running[stepName] = newSw;
        }

        // Track step order for table display
        if (!stepOrder.Contains(stepName))
        {
            stepOrder.Add(stepName);
        }
    }

    /// <summary>
    /// Stop timing a named step and record the elapsed ms.
    /// Also triggers a table print if enough time has passed.
    /// </summary>
    public static void End(string stepName)
    {
        if (running.TryGetValue(stepName, out Stopwatch sw))
        {
            sw.Stop();
            completed[stepName] = sw.Elapsed.TotalMilliseconds;
        }

        TryPrint();
    }

    /// <summary>
    /// Record an arbitrary key-value pair (e.g., detection count).
    /// </summary>
    public static void Set(string key, object val)
    {
        values[key] = val.ToString();
    }

    private static void TryPrint()
    {
        if (Time.time - lastPrintTime < PrintIntervalSeconds)
        {
            return;
        }

        lastPrintTime = Time.time;

        if (completed.Count == 0 && values.Count == 0)
        {
            return;
        }

        var sb = new StringBuilder();
        float fps = 1f / Time.unscaledDeltaTime;
        sb.AppendLine($"{Tag} ====== FPS: {fps:F0} ======");

        // Print timing steps in insertion order
        foreach (var step in stepOrder)
        {
            if (completed.TryGetValue(step, out double ms))
            {
                sb.AppendLine($"{Tag}  {step,-22}: {ms,8:F1} ms");
            }
        }

        sb.AppendLine($"{Tag} --------------------------------");

        // Print extra values
        foreach (var kv in values)
        {
            sb.AppendLine($"{Tag}  {kv.Key,-22}: {kv.Value,8}");
        }

        // Memory stats
        long monoMB = Profiler.GetMonoUsedSizeLong() / (1024 * 1024);
        long totalMB = Profiler.GetTotalAllocatedMemoryLong() / (1024 * 1024);
        sb.AppendLine($"{Tag}  {"Mono Heap",-22}: {monoMB,5} MB");
        sb.AppendLine($"{Tag}  {"Total Alloc",-22}: {totalMB,5} MB");
        sb.AppendLine($"{Tag} ================================");

        Debug.Log(sb.ToString());
    }
}
