# Did cheaper subs sell more subs?

**Price elasticity of Twitch channel subscriptions from the 2021 local-pricing rollout** — a staggered difference-in-differences study (Callaway & Sant'Anna 2021) using the country-by-country price cuts as a natural experiment with dose variation.

> Status: in progress. Synthetic-data pipeline and estimator validation are live
> (`make synth && make estimate && make test`); real-data acquisition is next.
> See [PLAN.md](PLAN.md) for design, data reconnaissance, and milestones.

## Quick start

```bash
uv sync --group dev
make all    # synthetic panel → CS event study, TWFE contrast, elasticity, figures
make test   # includes estimator-recovers-known-truth validation
```

*(Full README — identification strategy, results, and threats to validity — lands at milestone M8.)*
