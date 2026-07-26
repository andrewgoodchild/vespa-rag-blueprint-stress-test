# Design direction — approved

Three directions were built as 2-page showcases and shown to the user
(screenshots in `shots/`, live HTML in `design-demos/`):
- `terminal.html` — IR-lab dark, warm near-black + amber signal + teal, JetBrains Mono + Space Grotesk
- `editorial.html` — research editorial, warm paper + oxblood serif (Fraunces)
- `dataviz.html` — Swiss/Pentagram, bold cobalt poster

**User choice (verbatim): "1. terminal"**

→ Build the full ~8-slide deck in the **terminal / IR-lab** direction, matching
`design-demos/terminal.html`'s design system exactly (colors, type, header/footer
chrome, cost-tier bar coloring). Fix the minor overflow issues seen in the other
two directions by keeping each slide within 1920×1080.

---

## Companion teaching deck (2026-07-26, since merged)

User asked for a second deck to teach the concepts the study spans
(descriptions of everything tried + tradeoffs + nuances). Treated as an
**iteration within the already-approved terminal direction** (same project,
companion to deck.html) per the "iteration after a direction is chosen"
exemption — built in the identical terminal/IR-lab visual system for a matched
pair. User told the option to run fresh directions instead. Content in
`factsheet-learn.md`; output `deck-learn.html`.

**2026-07-28:** the two decks were consolidated into a single 27-slide
`deck.html` — frame, then one slide per technique carrying its own measured
result, then the evidence sections and a glossary. `deck-learn.html` and
`factsheet-learn.md` were retired; their copy lives on in `factsheet.md`.
