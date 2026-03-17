const BASE = "";

export async function fetchRuns(): Promise<string[]> {
  const res = await fetch(`${BASE}/api/runs`);
  return res.json();
}

export async function fetchRunSummary(run: string) {
  const res = await fetch(`${BASE}/api/runs/${encodeURIComponent(run)}/summary`);
  return res.json();
}

export async function fetchMetrics(run: string, from = 0) {
  const res = await fetch(
    `${BASE}/api/runs/${encodeURIComponent(run)}/metrics?from=${from}`,
  );
  return res.json();
}

export async function fetchLatest(run: string) {
  const res = await fetch(`${BASE}/api/runs/${encodeURIComponent(run)}/latest`);
  return res.json();
}

export async function fetchComparison(runs: string[]) {
  const res = await fetch(`${BASE}/api/compare?runs=${runs.join(",")}`);
  return res.json();
}

// WebSocket for real-time updates
export function connectWS(onMessage: (data: any) => void): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch {
      // ignore parse errors
    }
  };

  ws.onclose = () => {
    // Reconnect after 2s
    setTimeout(() => connectWS(onMessage), 2000);
  };

  return ws;
}
