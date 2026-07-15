# Plan: Twitch local-pricing price elasticity study

**Question.** Did cheaper subs sell more subs? Twitch cut Tier-1 subscription prices country-by-country through 2021 (Turkey/Mexico May 20; rest of LatAm by Jul 27; APAC from Aug 5; Europe/MEA through Q3–Q4), calibrated to local purchasing power. US prices never changed. The staggered rollout with country-varying price cuts (~80% Turkey, ~50% Mexico, smaller in Western Europe) is a natural experiment with dose variation — enough to estimate a demand elasticity for channel subscriptions.

**Design.** Channel-month panel. Prices changed per *viewer* country but outcomes are per *channel*, so each channel is assigned a treatment cohort via its dominant audience country, proxied by broadcast language + streamer country (Turkish → Turkey May-2021 cohort; pt-BR → Brazil; Spanish split by streamer country; …). Never-treated control: channels with US/English-dominant audiences. Primary estimator: Callaway & Sant'Anna (2021) group-time ATTs, doubly robust, with event-study aggregation and an anticipation allowance (rollout pre-announced ~Apr 2021). Elasticity from dose-response across cohorts: ε = Δlog(subs)/Δlog(price). Outcome is subscription *counts*, not revenue — Twitch's 12-month revenue guarantee for creators muddies revenue but not counts.

**Why not plain TWFE.** Staggered timing + heterogeneous, dynamic effects make the two-way fixed-effects coefficient a weighted mix of clean and forbidden comparisons (Goodman-Bacon 2021). The repo demonstrates this: on synthetic data with known truth, TWFE is measurably biased while CS recovers the planted effect (`tests/test_estimator_validation.py`); the real-data writeup includes the decomposition.

## Data sources (reconnaissance verified 2026-07-15)

| Source | What | Access reality |
|---|---|---|
| TwitchTracker per-channel `/subscribers` | Monthly subs by tier incl. **Prime & Gifted split** — primary outcome | Robots-allowed HTML, but data renders client-side behind Cloudflare → headed real-browser scraping (Playwright), ≥2s jittered delays, on-disk cache, never `/api/*` |
| TwitchTracker `/subscribers` leaderboard | Top-channel monthly sub counts | Server-rendered; **Wayback has ~monthly snapshots 2020–2022** → independent validation series |
| SullyGnome | Language, streamer country, followers | Same Cloudflare situation; browser scraping |
| Wayback Machine | Archived leaderboards, press, price lists | Open; help-page snapshots are JS-rendered skeletons (no price table in HTML) |
| Twitch Helix API | Current broadcaster language (enrichment only) | Free client credentials |
| Price table | country → rollout month, new price, %Δ price (USD at rollout-month FX) | Hand-curated CSV with per-row source URL + date (`data/reference/price_table.csv`); anchors: Turkey 9.90 TRY, Mexico 48 MXN, both 2021-05-20 |
| Leaked 2021 payout dataset | — | **Excluded.** Pluggable interface exists but is hard-disabled; requires explicit owner opt-in |

## Milestones (all complete)

- **M0** scaffold (uv, ruff, mypy, pytest, CI, Makefile) ✅
- **M1** synthetic panel with known elasticity + estimator validation ✅ (CS recovers planted ε=−0.6; TWFE misses by 5.7×; placebos null)
- **M2** cohort/price reference table with provenance + anchor tests ✅ (incl. corrected timeline: Europe launched Aug 5, 2021)
- **M3** scraper infrastructure ✅ (fetcher enforces no-circumvention; Cloudflare blocks automation on data pages → pivot to real user session)
- **M4** pilot via the project owner's real browser session ✅ — **finding: treated side hard-capped at ~29 channels tracked pre-May-2021**; focused scale-up chosen at the decision gate, treated expansion self-limited, EN controls added
- **M5** real panel + R `did` cross-check ✅ (Python ≡ R to machine precision, enforced in CI; anticipation=1 spec for real data)
- **M6** estimation ✅ — **ε = −0.90, cluster-bootstrap 95% CI [−1.36, −0.34]**; overall ATT +0.56±0.21; TWFE 0.84 with the Goodman-Bacon decomposition explaining the gap
- **M7** diagnostics ✅ — Prime placebo null (−0.07±0.22); permutation p=0.005; leads test p=0.35; linear-trend sensitivity halves the magnitude; effect concentrates below median channel size (+0.90 vs +0.02)
- **M8** README (hiring-manager narrative) + RESULTS.md (stakeholder summary) + executed notebook ✅

**Deviations from the original plan, for the record:** SullyGnome follower data was replaced by Prime subs as the falsification outcome (blocked pages; Prime is arguably sharper — same product, price-insensitive). Full Rambachan–Roth relative-magnitudes honest-DiD remains future work; the transparent linear-trend special case is implemented. Spanish-channel country labels are curated and flagged provisional. The 1–5k channel target was impossible from public data: TwitchTracker's tracking history, not scraping effort, is the binding constraint.

## Known threats (tracked from day one, discussed honestly in README)

Contaminated controls (English channels have treated international viewers → attenuation); TRY collapse late 2021 (dose in USD at rollout-date FX; sensitivity checks); COVID-era platform growth (calendar-time effects absorbed by design, but trend breaks discussed); gift & Prime subs (mitigated: tier table separates them — paid non-gift subs is the primary outcome); tracker measurement error and selection into tracking (top-channel bias documented); anticipation (pre-announced ~Apr 2021 — CS anticipation parameter); rate limits & robots respected throughout — if a source blocks polite browser access, we stop and descope rather than circumvent.
