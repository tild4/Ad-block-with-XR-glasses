/*
    Summary:
    Collects timing and status values from the ad-blocking pipeline and
    periodically prints a compact logcat table.

    Pipeline:
    Pipeline components -> PipelineProfiler -> Unity Debug.Log
*/

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

    private static readonly Dictionary<string, Stopwatch> running =
        new Dictionary<string, Stopwatch>();

    private static readonly Dictionary<string, double> completed = new Dictionary<string, double>();

    private static readonly Dictionary<string, string> values = new Dictionary<string, string>();

    private static readonly List<string> stepOrder = new List<string>();

    private static float lastPrintTime = -PrintIntervalSeconds;

    public static void begin(string stepName)
    {
        if (running.TryGetValue(stepName, out Stopwatch sw))
        {
            sw.Restart();
        }
        else
        {
            running[stepName] = Stopwatch.StartNew();
        }

        if (!stepOrder.Contains(stepName))
        {
            stepOrder.Add(stepName);
        }
    }

    public static void end(string stepName)
    {
        if (running.TryGetValue(stepName, out Stopwatch sw))
        {
            sw.Stop();
            completed[stepName] = sw.Elapsed.TotalMilliseconds;
        }

        tryPrint();
    }

    public static void set(string key, object val)
    {
        values[key] = val.ToString();
    }

    private static void tryPrint()
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

        foreach (string step in stepOrder)
        {
            if (completed.TryGetValue(step, out double ms))
            {
                string label = step.PadRight(22);
                string value = ms.ToString("F1").PadLeft(8);
                sb.AppendLine($"{Tag}  {label}: {value} ms");
            }
        }

        sb.AppendLine($"{Tag} --------------------------------");

        foreach (var kv in values)
        {
            string label = kv.Key.PadRight(22);
            string value = kv.Value.PadLeft(8);
            sb.AppendLine($"{Tag}  {label}: {value}");
        }

        long monoMB = Profiler.GetMonoUsedSizeLong() / (1024 * 1024);
        long totalMB = Profiler.GetTotalAllocatedMemoryLong() / (1024 * 1024);
        sb.AppendLine($"{Tag}  {"Mono Heap".PadRight(22)}: {monoMB.ToString().PadLeft(5)} MB");
        sb.AppendLine($"{Tag}  {"Total Alloc".PadRight(22)}: {totalMB.ToString().PadLeft(5)} MB");

        sb.AppendLine($"{Tag} ================================");

        Debug.Log(sb.ToString());
    }
}
