# Known Issues — wikipicture

Concrete defects and gaps found while writing this repository's documentation in
August 2026. **Nothing here was changed** — each one needs a code, configuration, or
licensing decision rather than a documentation one.

Ordered by severity. See [`docs/roadmap.md`](../roadmap.md) for the narrative version,
which also covers deliberate non-goals.

**3 open:** 1 medium, 2 low.

## 1. A location with existing Commons photos is reported as having none when their dates cannot be parsed

**Severity:** Medium
**Where:** `src/wikipicture/scorer.py` -> `_score_freshness`; `src/wikipicture/commons_checker.py` -> `check_freshness`

**What:** `_score_freshness` returns `(15.0, "No existing photos at this location")` for two different conditions: `commons.nearby_image_count == 0`, and `commons.newest_image_year is None`. The second means Commons found images but no usable dates. `check_freshness` has the same shape -- its final `return "No existing photos"` is reached whenever either year is `None`, after the `nearby_image_count == 0` check has already passed.

**Why it matters:** The report states, in words, that a place has no photographic coverage while the same row's Commons count says otherwise -- and awards the full 15 freshness points on top, pushing the photo toward 'recommended'. A user acting on that uploads to a location already well covered, which is precisely the outcome the saturation and freshness signals exist to prevent. The two conditions share a message that is true for only one of them, so nothing looks wrong when reading the code either.

**Suggested fix:** Separate the branches. When images exist but dates are unknown, say so ('coverage exists, age unknown') and score it in the middle rather than at the maximum -- the honest position is uncertainty, not absence.

## 2. The "outdated" freshness message quotes the oldest year while the decision used the newest

**Severity:** Low
**Where:** `src/wikipicture/commons_checker.py` -> `check_freshness`

**What:** The branch reads: `if result.newest_image_year < current_year - 5: return f"Photos are outdated (oldest from {result.oldest_image_year})"`. The comparison is on `newest_image_year`; the message reports `oldest_image_year`.

**Why it matters:** A location whose Commons photos span 1995 to 2015 is described as 'Photos are outdated (oldest from 1995)'. The reader concludes the coverage is thirty years old when it is ten -- and the number they need in order to judge whether their own photo adds anything is the one not shown. The report's value is its explanations, so an explanation quoting the wrong variable undermines the part that matters.

**Suggested fix:** Report `newest_image_year` in that message, matching the condition. `oldest_image_year` is still worth showing, but as a range rather than as the basis for the verdict.

## 3. The implied scan command only works when the path is the first argument

**Severity:** Low
**Where:** `src/wikipicture/cli.py` -> `_DefaultGroup.parse_args`

**What:** `if args and args[0] not in self.commands and not args[0].startswith("-"): args = ["scan"] + args`. The `startswith("-")` guard exists so a bare `--help` or `--version` still reaches the group, but it also means any invocation beginning with an option skips the injection.

**Why it matters:** `wikipicture --output report.html ./photos` fails with `no such option: --output`, which names the option rather than the missing subcommand and sends the user looking at their flags. Options-before-arguments is a normal habit, and the README's own examples put the path first without saying it is required -- so the failure looks like the flag does not exist.

**Suggested fix:** Inject `scan` whenever the first non-option argument is not a known command, rather than keying on `args[0]` alone. Special-casing `--help` and `--version` explicitly keeps those working.

---

## Also, across every repository

**`.bandit` is present on disk but untracked in git.** Verified in PyWorkout, treklogger,
skyscanner-cli, booking-cli, piggy, and aibot — the config file exists locally in each but
`git ls-files` does not know about it, so none of it reached GitHub.

The August 2026 security sweep therefore looks complete locally and landed nowhere. Worth
checking across all 44 repositories it covered.
