# FailureAtlas Interface System

## Direction

An operational ML lab console: compact, high-signal, and calm. The interface behaves
like an instrument panel for experiments rather than a generic business dashboard.
The central visual is the model's robustness frontier, with repair evidence adjacent
to the controls that produced it.

## Mode

Operate. Users inspect evidence, compare runs, and start local jobs.

## Palette

- Canvas: `#090c10`
- Sidebar: `#0d1117`
- Primary surface: `#111720`
- Raised surface: `#171e28`
- Rule: `#26303d`
- Primary text: `#f2f5f7`
- Secondary text: `#9aa8b8`
- Signal green: `#c6f66f` for repaired/current/positive states and primary actions
- Baseline blue: `#7197c9`
- Warning amber: `#f3b35a`
- Critical coral: `#f07467`

## Typography

Use Inter when available, falling back to Segoe UI and system sans. Use tabular
numerals for measurements. Monospace is reserved for severity values, file paths,
and logs.

## Geometry

Panels use 14px corners, thin rules, and no decorative shadows. Buttons and compact
status chips may be pill-shaped. Charts and tables align to a 4px spacing base with
24–32px section separation.

## Interaction

The selected perturbation controls the comparison chart, boundary summary, and
diagnostic copy together. Transitions are 180ms and communicate state changes only.
Keyboard focus is always visible. Long-running actions expose queued/running/success/error states.

## Responsive Behavior

At tablet width the sidebar becomes a top navigation strip. Below 720px, chart and
diagnosis panels stack, comparison tables become horizontally scrollable, and job
actions remain reachable without hiding core evidence.
