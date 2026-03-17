import { NavLink, Outlet } from "react-router-dom";
import { Separator } from "@radix-ui/react-separator";
import clsx from "clsx";

const navItems = [
  { to: "/", label: "Overview", icon: "◉" },
  { to: "/losses", label: "Losses", icon: "📉" },
  { to: "/clusters", label: "Clusters", icon: "◎" },
  { to: "/gate", label: "Gate", icon: "⊞" },
  { to: "/hierarchy", label: "Hierarchy", icon: "⊟" },
  { to: "/generation", label: "Generation", icon: "✦" },
  { to: "/evals", label: "Evals", icon: "⬡" },
  { to: "/ablations", label: "Ablations", icon: "⊕" },
];

export function Layout() {
  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <nav className="w-56 shrink-0 border-r border-zinc-800 bg-zinc-950 flex flex-col">
        <div className="p-4">
          <h1 className="text-lg font-bold tracking-tight">
            <span className="text-brand-light">z86</span>
            <span className="text-text-dim">.dev</span>
          </h1>
          <p className="text-xs text-text-dim">HCLM-D Training Dashboard</p>
        </div>
        <Separator className="h-px bg-zinc-800" />
        <div className="flex-1 py-2 space-y-0.5 px-2">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                clsx(
                  "flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors",
                  isActive
                    ? "bg-brand/15 text-brand-light font-medium"
                    : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50",
                )
              }
            >
              <span className="text-base">{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
        </div>
        <div className="p-3 text-[10px] text-text-dim">
          z86.dev — Bun + Hono + Radix
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-1 overflow-auto bg-zinc-950">
        <div className="max-w-7xl mx-auto p-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
