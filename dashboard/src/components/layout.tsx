import { NavLink, Outlet } from "react-router-dom";
import clsx from "clsx";

const navItems = [
  { to: "/", label: "Overview" },
  { to: "/losses", label: "Losses" },
  { to: "/clusters", label: "Clusters" },
  { to: "/gate", label: "Gate" },
  { to: "/hierarchy", label: "Hierarchy" },
  { to: "/generation", label: "Generation" },
  { to: "/evals", label: "Evals" },
  { to: "/ablations", label: "Ablations" },
];

export function Layout() {
  return (
    <div className="flex flex-col h-screen overflow-hidden">
      {/* Top bar */}
      <header
        className="h-8 flex items-center justify-between px-3 shrink-0"
        style={{ background: "var(--color-s1)", borderBottom: "1px solid var(--color-border)" }}
      >
        <div className="flex items-center gap-2">
          <span className="font-[var(--font-mono)] text-[10px] font-semibold tracking-wider" style={{ color: "var(--color-accent-red-hi)" }}>
            Z86.DEV
          </span>
          <span className="text-[10px]" style={{ color: "var(--color-text-sub)" }}>/</span>
          <span className="font-[var(--font-mono)] text-[10px]" style={{ color: "var(--color-t3)" }}>
            HCLM-D · train
          </span>
        </div>

        {/* Navigation tabs */}
        <nav className="flex items-center gap-0.5">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                clsx(
                  "terminal-tab",
                  isActive && "active",
                )
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1">
            <span className="status-dot" />
            <span className="font-[var(--font-mono)] text-[9px]" style={{ color: "var(--color-t3)" }}>
              ready
            </span>
          </div>
          <button className="top-btn">checkpoint</button>
          <button className="top-btn">export</button>
          <button className="top-btn danger">stop</button>
        </div>
      </header>

      {/* Main content — pages render full-bleed grids */}
      <main className="flex-1 overflow-auto" style={{ background: "var(--color-bg)" }}>
        <Outlet />
      </main>
    </div>
  );
}
