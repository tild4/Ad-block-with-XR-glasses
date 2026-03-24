/*
    PipelineProfiler

    PURPOSE:
    Centralized static profiler that collects per-step timing across the
    ad-block pipeline and prints one summary table every N seconds via
    Debug.Log. Designed for viewing through ADB logcat on Meta Quest 3.

    ARCHITECTURE:
    - Static class: No MonoBehaviour, no GameObject, no scene setup needed.
    - Begin/End API: Each pipeline script calls Begin("StepName") and
      End("StepName") around work. Stopwatch measures wall-clock time,
      surviving coroutine yields across frames.
    - Rate-limited printing: tryPrint() fires at most once per
      PrintIntervalSeconds to avoid log spam.
    - Set API: For non-timing values (e.g. detection count).

    USAGE:
      PipelineProfiler.begin("StepName");
      // ... work ...
      PipelineProfiler.end("StepName");
      PipelineProfiler.set("Detections", count);

    VIEW IN TERMINAL:
      adb logcat -s Unity | grep PIPELINE
      Or use: ./view_profiler.sh
*/

using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using UnityEngine;
using UnityEngine.Profiling;
// Alias to avoid conflict with System.Diagnostics.Debug
using Debug = UnityEngine.Debug;

public static class PipelineProfiler
{
    // How often the summary table is printed (seconds)
    private const float PrintIntervalSeconds = 3.0f;

    // Prefix tag for all log lines, used for filtering in logcat
    private const string Tag = "[PIPELINE]";

    // Currently running stopwatches keyed by step name
    private static readonly Dictionary<string, Stopwatch> running =
        new Dictionary<string, Stopwatch>();

    // Most recently completed timing per step name (ms)
    private static readonly Dictionary<string, double> completed = new Dictionary<string, double>();

    // Arbitrary key-value pairs (e.g. detection count)
    private static readonly Dictionary<string, string> values = new Dictionary<string, string>();

    // Preserves insertion order so the table rows appear in pipeline order
    private static readonly List<string> stepOrder = new List<string>();

    // Tracks when the last table was printed
    private static float lastPrintTime = -PrintIntervalSeconds;

    /*
        Start timing a named step. Call end() with the same name when done.
        Stopwatch survives across coroutine yields (measures wall-clock time).
        If the same step is started again before ending, the stopwatch restarts.
    */
    public static void begin(string stepName)
    {
        // Restart existing stopwatch or create a new one
        if (running.TryGetValue(stepName, out Stopwatch sw))
        {
            sw.Restart();
        }
        else
        {
            running[stepName] = Stopwatch.StartNew();
        }

        // Track step order for consistent table display
        if (!stepOrder.Contains(stepName))
        {
            stepOrder.Add(stepName);
        }
    }

    /*
        Stop timing a named step and record the elapsed milliseconds.
        Also triggers a table print if enough time has passed since the last one.
    */
    public static void end(string stepName)
    {
        if (running.TryGetValue(stepName, out Stopwatch sw))
        {
            sw.Stop();
            completed[stepName] = sw.Elapsed.TotalMilliseconds;
        }

        tryPrint();
    }

    /*
        Record an arbitrary key-value pair for display in the summary table.
    */
    public static void set(string key, object val)
    {
        values[key] = val.ToString();
    }

    /*
        Prints the summary table if PrintIntervalSeconds has elapsed.
        Called automatically from end() — no Update() loop needed.
    */
    private static void tryPrint()
    {
        // Rate-limit: skip if printed recently
        if (Time.time - lastPrintTime < PrintIntervalSeconds)
        {
            return;
        }

        lastPrintTime = Time.time;

        // Nothing to print yet
        if (completed.Count == 0 && values.Count == 0)
        {
            return;
        }

        var sb = new StringBuilder();

        // Header with current FPS
        float fps = 1f / Time.unscaledDeltaTime;
        sb.AppendLine($"{Tag} ====== FPS: {fps:F0} ======");

        // Timing rows in pipeline order
        foreach (string step in stepOrder)
        {
            if (completed.TryGetValue(step, out double ms))
            {
                sb.AppendLine($"{Tag}  {step, -22}: {ms, 8:F1} ms");
            }
        }

        sb.AppendLine($"{Tag} --------------------------------");

        // Extra values (detection count, etc.)
        foreach (var kv in values)
        {
            sb.AppendLine($"{Tag}  {kv.Key, -22}: {kv.Value, 8}");
        }

        // Memory stats from Unity Profiler API
        long monoMB = Profiler.GetMonoUsedSizeLong() / (1024 * 1024);
        long totalMB = Profiler.GetTotalAllocatedMemoryLong() / (1024 * 1024);
        sb.AppendLine($"{Tag}  {"Mono Heap", -22}: {monoMB, 5} MB");
        sb.AppendLine($"{Tag}  {"Total Alloc", -22}: {totalMB, 5} MB");

        sb.AppendLine($"{Tag} ================================");

        Debug.Log(sb.ToString());
    }
}
