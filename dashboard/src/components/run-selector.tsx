import * as Select from "@radix-ui/react-select";

interface RunSelectorProps {
  runs: string[];
  value: string | null;
  onChange: (run: string) => void;
}

export function RunSelector({ runs, value, onChange }: RunSelectorProps) {
  if (runs.length === 0) {
    return (
      <p className="text-sm text-zinc-500 italic">
        No runs yet. Start training to see metrics.
      </p>
    );
  }

  return (
    <Select.Root value={value ?? undefined} onValueChange={onChange}>
      <Select.Trigger className="inline-flex items-center gap-2 rounded-md border border-zinc-700 bg-surface-raised px-3 py-1.5 text-sm text-zinc-200 hover:border-zinc-600 focus:outline-none focus:ring-2 focus:ring-brand/50">
        <Select.Value placeholder="Select a run..." />
        <Select.Icon className="text-zinc-500">▾</Select.Icon>
      </Select.Trigger>
      <Select.Portal>
        <Select.Content className="rounded-md border border-zinc-700 bg-surface-raised shadow-xl z-50">
          <Select.Viewport className="p-1">
            {runs.map((run) => (
              <Select.Item
                key={run}
                value={run}
                className="flex items-center rounded px-3 py-1.5 text-sm text-zinc-300 cursor-pointer outline-none data-[highlighted]:bg-brand/20 data-[highlighted]:text-brand-light"
              >
                <Select.ItemText>{run}</Select.ItemText>
              </Select.Item>
            ))}
          </Select.Viewport>
        </Select.Content>
      </Select.Portal>
    </Select.Root>
  );
}
