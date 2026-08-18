# WikiPicture — FAQ

### Are my photos uploaded anywhere?

No. They are read locally for EXIF and for the blur check. What goes out is the **coordinates of
each cluster** — sent to Nominatim, Wikipedia, and Commons to ask what is there.

Because photos are clustered first, that is one lookup per place rather than one per photo.

### Do I need an API key?

No. All three services are public and unauthenticated.

### Does it cost anything?

No. It does cost *someone* — these are volunteer-funded services, which is why the tool rate
limits itself to one geocoding request per second and caches everything.

### Why is the first run so slow?

The one-request-per-second geocoding limit. Time is roughly one second per cluster, not per
photo, so `--cluster-distance 1000` speeds things up considerably at the cost of coarser matches.

Subsequent runs hit the cache and are fast.

### My photos were skipped.

Almost always missing GPS. The tool needs EXIF coordinates and cannot infer them. Check:

- Was location tagging on when they were taken?
- Did an editor or an upload strip EXIF? Many do, silently.
- Are they JPEG or HEIC/HEIF? Other formats are not scanned.

### It found no Wikipedia articles.

Either there is nothing notable within the search radius, or what is there is not on Wikipedia.
Remote places genuinely have no articles — and articles that do not exist cannot need photos.

### What do the scores mean?

Out of 100: article need (40), Commons saturation (30), photo quality (15), freshness (15). 70+
is "highly recommended". The report shows the breakdown for every photo — read that rather than
the total.

### It says "No existing photos at this location" but Commons has some.

A known bug: when the dates of nearby Commons images cannot be determined, the freshness check
reports no photos and awards full marks. Check the Commons count in the same row. Recorded in
[`internal/known-issues.md`](./internal/known-issues.md).

### Why does the "outdated" message quote an older year than I expect?

It quotes the oldest photo's year while the decision was made on the newest. Same known-issues
file.

### Does it upload to Commons for me?

No, and that is deliberate. Commons has licensing rules, scope rules, and naming conventions
worth understanding before contributing — see
[Commons: Licensing](https://commons.wikimedia.org/wiki/Commons:Licensing). This tool tells you
where a photo would help; the judgement is yours.

### Can I run it on my whole library at once?

Yes, but start with `--limit 50` to see the shape of the results. A large first run is a long
wait for output you may want to tune the flags for.

### Does `--skip-quality-check` make my photos score worse?

No — it makes them score *less specifically*. Unassessed photos get 10 of the 15 quality points
rather than 15 or 0.

### Where is the cache, and is it safe to delete?

`~/.wikipicture/cache.db`. Safe to delete; the next run rebuilds it, slowly.

### Does it work with HEIC from an iPhone?

Yes, via `pillow-heif`, which is a required dependency.

### Why does `wikipicture --output r.html ./photos` fail?

The implied `scan` only fires when the first argument is a path. Put the directory first, or type
`scan` explicitly. See [Configuration](./configuration.md).
