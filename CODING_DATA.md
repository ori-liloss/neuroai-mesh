# Coding specialist: recommended data mix

For a future build_coding_mix.py. Shares are row counts after dedup and the
`build_all_mix.reject` filters. Real pool sizes in `neuro-ai-data/training`
were not readable here; check them before fixing a total.

## What the existing builders can produce today

| Source | Gives | Correctness signal |
|---|---|---|
| `build_long_mix.py` code_rows | one SQL template family (LEFT JOIN count, ~600 variants) and a few Python fix/filter families | executed on fixtures |
| `verified_code.py` | 4 Python, 3 JavaScript, 2 SQL gold rows | unit tests / sqlite results |
| `build_long_mix.py` luau_rows, `build_ring_mix.py` | Luau families, English and Hebrew prompts | executed with luau |
| `build_long_mix.py` math_rows, general_rows | computed math; Hebrew/English drills (counting capped at 2500) | computed |
| `build_all_mix.py` | every older chat file, no per-domain cap | none beyond filters |

There is no source of TypeScript or HTML/CSS rows, and almost none for
JavaScript. Those shares have to come from mesh rejection sampling:
`jobs add generate-verify` with `"samples": 4` and `verify_js`, `verify_ts`,
`verify_html_css` or `verify_sql`, then `jobs export` (one verified answer per
prompt). `prepare.py` knows only the domains general/code/math/luau, so put the
language in `meta.lang` and let the mix builder balance on it.

## Target shares (SFT)

| Domain | Share | Notes |
|---|---|---|
| code | 50% | split below |
| general | 25% | Hebrew-first: at least 60% of these rows in Hebrew |
| luau | 13% | executed rows only; keep `luau_holdout` reserved |
| math | 12% | computed answers; keep both math holdouts reserved |

Code, as a share of the whole mix:

| Language | Share | Verifier |
|---|---|---|
| JavaScript (Node + browser) | 13% | verify_js (parse) |
| TypeScript | 9% | verify_ts (parse, erasable syntax only) |
| Python | 11% | gold tests where they exist, else verify_tests (runs the answer's own asserts), else verify_python_syntax |
| HTML/CSS | 8% | verify_html_css (structure) |
| SQL (SQLite) | 6% | executed drills plus verify_sql (compile) |
| Review, debugging, explanation | 3% | answer must contain a block that passes its verifier |

Code tasks: ~45% write-from-spec, 25% bug fix, 15% refactor/port, 15% explain/review.

## Hebrew and rules

- About 15% of all rows are Hebrew chat (60% of general). About 20% of coding
  prompts are Hebrew with English code, like the first `verified_code.py` row.
- Cap every template family: no single family above 1500 rows. The SQL
  LEFT JOIN family alone could fill the whole SQL share.
- `seeds/coding-v2-tested.jsonl` (Python and JavaScript) is checked by running the answer's own
  asserts (`verify_tests`) plus a cross-check against the other samples' asserts; prefer
  `jobs export --require-agreement` rows over parse-verified ones.
- Parse-only verifiers do not prove behaviour. When a gold-tested row and a
  parse-verified mesh row exist for the same task, keep the gold one.
- Ask for SQLite in SQL prompts. verify_sql fails other dialects.
- Pairs from `jobs export --pairs` are for a later DPO/KTO stage. They must not
  go into an SFT folder; the export refuses paths under `neuro-ai-data/training`.
- Validation 10% per domain. First build ~24k rows for the 4070 QLoRA run
  (packing off); grow it from mesh exports at the same shares.
