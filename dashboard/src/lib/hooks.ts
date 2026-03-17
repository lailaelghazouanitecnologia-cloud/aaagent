import { useState, useEffect, useCallback, useRef } from "react";
import { fetchRuns, fetchMetrics, connectWS, fetchEvalRuns, fetchEvals } from "./api";

export function useRuns() {
  const [runs, setRuns] = useState<string[]>([]);
  useEffect(() => {
    fetchRuns().then(setRuns).catch(() => {});
  }, []);
  return runs;
}

export function useMetrics(run: string | null) {
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    if (!run) return;
    setLoading(true);
    try {
      const metrics = await fetchMetrics(run);
      setData(metrics);
    } finally {
      setLoading(false);
    }
  }, [run]);

  useEffect(() => {
    load();
  }, [load]);

  return { data, loading, reload: load };
}

export function useRealtimeMetrics(run: string | null) {
  const [data, setData] = useState<any[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!run) return;

    // Load initial data
    fetchMetrics(run).then(setData).catch(() => {});

    // Connect WebSocket for live updates
    wsRef.current = connectWS((msg) => {
      if (msg.type === "metric" && msg.data.run === run) {
        setData((prev) => [...prev, msg.data]);
      }
    });

    return () => {
      wsRef.current?.close();
    };
  }, [run]);

  return data;
}

export function useEvalRuns() {
  const [runs, setRuns] = useState<string[]>([]);
  useEffect(() => {
    fetchEvalRuns().then(setRuns).catch(() => {});
  }, []);
  return runs;
}

export function useEvals(run: string | null) {
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    if (!run) return;
    setLoading(true);
    try {
      const evals = await fetchEvals(run);
      setData(evals);
    } finally {
      setLoading(false);
    }
  }, [run]);

  useEffect(() => {
    load();
  }, [load]);

  return { data, loading, reload: load };
}
