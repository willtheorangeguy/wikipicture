# WikiPicture — Roadmap

Direction, not a schedule. Defects are tracked in
[`internal/known-issues.md`](./internal/known-issues.md); this page is about what the tool is
*for*.

## Where it is

The pipeline runs end to end: EXIF, clustering, geocoding, Wikipedia, Commons, quality, scoring,
report. Each stage is a separate module with its own tests and fully mocked HTTP.

## Considered

**Fixing the freshness reporting.** A location with Commons photos whose dates cannot be parsed
is described as having none, and scores full freshness marks. It is the one place the report says
something untrue.

**Cache expiry for API results.** Photo entries invalidate on mtime; Wikipedia and Commons
results never expire, so an article that gained images stays cached as needing one.

**Accepting options before the path.** The implied `scan` only fires when the first argument is
not an option.

**Configurable search radii.** The 10 km Wikipedia geosearch and the Commons radius are
constants; a dense city and a national park want different values.

**Resuming an interrupted scan.** The cache means a re-run is cheap, but there is no explicit
resume.

**Suggesting Commons categories.** `commons_checker.get_upload_categories` already exists;
surfacing it in the report would shorten the actual upload.

## Non-goals

**Uploading to Commons.** Deliberate. Commons has licensing, scope, and naming rules that a
person should read before contributing, and an automated uploader would produce exactly the sort
of contribution that gets reverted. The pipeline ends at a recommendation.

**Editing Wikipedia articles.** Same reasoning, more so — adding an image to an article is an
editorial act.

**Uploading or transmitting photos.** Only coordinates leave the machine, and clustering keeps
even those coarse.

**Working around rate limits.** Nominatim allows one request per second and this respects it.
Anything that circumvents that gets shared IP ranges blocked from a service other people rely on.

**Guessing at locations.** Photos without GPS are skipped rather than inferred. A wrong location
produces a wrong recommendation, and a photo uploaded to the wrong article is worse than one not
uploaded.

## Contributing

Issues and pull requests welcome — see the
[Contributing Guide](https://github.com/willtheorangeguy/.github/blob/main/CONTRIBUTING.md).
The freshness reporting is the highest-value fix and is contained in two functions.
