# Backlog

## Podcast Audio

- **Switch podcast audio from MP3 to M4A (AAC 32k)** — the `techradar` Pages site is ~2.2 GB against GitHub's 1 GB limit and nearly all of it is mp3; 90-day retention holds it there but can't shrink it. AAC at 32 kbps halves the size at close to current quality (32k MP3 was audibly worse in A/B). Needs: selectable `audio_format` config, ffmpeg AAC encode with `+faststart`, embedded chapters via FFMETADATA instead of ID3 (mutagen can't write MP4 chapters), `covr` cover art, mixed `.mp3`+`.m4a` catalog support in `podcast_rss.py` and `retention.py`. **New episodes only — never re-encode the back catalog** (`podcast-app` keys episodes on the audio URL). m4a alone lands at ~1.1 GB, so pair it with shorter audio retention (~60 days). `podcast-app` must ship its Cast MIME fix first. See plan: `docs/plans/2026-09-20-m4a-switch.md`.
