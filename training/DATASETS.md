# Training data sources and licences

The weights we ship are meant to be MIT. Training on data does not automatically make the weights a
derivative work of that data, but the law is unsettled. So the **default dataset only uses sources
whose terms clearly allow reuse, modification and commercial use**. Anything with research-only,
non-commercial or no-derivatives terms is marked `tier: research` in
[`data/sources.yaml`](data/sources.yaml). It only runs with `--include-research`, and weights
trained on it should not be shipped under MIT.

Licences were checked against the files served next to the media in November 2026. Re-check before
a release, because hosting sites change.

## Default (permissive) set

| id(s) in sources.yaml | What | Format we read | Licence / terms | Attribution |
|---|---|---|---|---|
| `tears_of_steel`, `tears_of_steel_val` | *Tears of Steel* (Blender Foundation, 2012): live action (Sony F65) with VFX | Lossless 1080p PNG master, `media.xiph.org/tearsofsteel/tearsofsteel-1080-png/` (17,620 frames; only sampled frames are downloaded) | **CC-BY 3.0** (`media.xiph.org/tearsofsteel/README.txt`). The soundtrack is CC-BY-ND, but we don't use it | "(CC) Blender Foundation \| mango.blender.org" |
| `netflix_elfuente_boat`, `netflix_elfuente_foodmarket` | Netflix *El Fuente* (2013 documentary) | 4096x2160 10-bit 4:2:0 BT.709 y4m in Netflix's Open Content bucket, read with HTTP range requests | **CC-BY 4.0**: `ElFuente/creative-commons-attribution-4-intl-public-license.txt` in the bucket, and the opencontent.netflix.com page says all Open Content is CC-BY 4.0 | "Netflix Open Content, CC BY 4.0" |
| `netflix_sparks_sdr` | Netflix *Sparks* (2017) | SDR UHD ProRes 422 BT.709 master (`sparks/sdr/…mov`, 46 GB), read remotely by ffmpeg seeks | **CC-BY 4.0** (`sparks/creative-commons-attribution-4-intl-public-license.txt`) | same |
| `netflix_nocturne_grain_sdr` | Netflix *Nocturne* (2018), with synthetic Vision3 film grain | SDR UHD ProRes (ACES 2.0 → 709), read remotely | **CC-BY 4.0** (`Nocturne/creative-commons-attribution-4-intl-public-license.txt`) | same |
| `tum_*` (blue_sky, riverbed, rush_hour, station2, sunflower, tractor; `tum_pedestrian_area` = val) | TU München 1080p25 test sequences, Sony HDW-F900 | 8-bit 4:2:0 y4m on the Xiph *derf* mirror | `ReadMe_1080p.txt`: "**Restrictions of use: No restrictions. Copyright: No Copyright**" | courtesy: "TU München, LDV" |
| `svt_*` (crowd_run, ducks_take_off, in_to_tree, park_joy; `svt_old_town_cross` = val) | SVT MultiFormat (Sveriges Television, 2001), 65 mm film scans | 1080p50 y4m on the Xiph *derf* mirror | SVT readme on the Xiph mirror: "**Restrictions of use: No restrictions. Copyright: No copyright**". Xiph links a separate SVT readme/PDF for the 1080p set too; worth re-reading before a release | courtesy: "SVT Sveriges Television AB" |
| `custom_stills` (disabled) | Your own footage or photos | any image folder | yours | — |

Ship a NOTICE with the weights that lists the CC-BY sources and their attribution lines. CC-BY asks
for attribution "reasonable to the medium", and a NOTICE next to the weights is the conservative
way to give it.

### Notes and caveats

- **The same Netflix clip can carry different terms depending on where you download it.** The
  *Chimera* and *El Fuente* y4m files on the Xiph mirror come with `Netflix_*_Copyright.txt`
  files that say **CC BY-NC-ND 4.0**. The copies in Netflix's own Open Content bucket sit next to a
  **CC BY 4.0** licence file. We only read the Netflix-bucket copies in the default set, and the
  Xiph copies are research-tier.
- Netflix *Meridian* and *Chimera* are also CC-BY 4.0 in the Netflix bucket, but they are
  published as HDR PQ / P3 masters. Using them needs an HDR → SDR tone map (e.g. ffmpeg `zscale` +
  `tonemap`), and a wrong tone map teaches the network wrong contrast, so they are not wired in.
  They would make good extra sources once a tested tone-map step exists.
- *Sintel*, *Spring*, *Cosmos Laundromat* and *Big Buck Bunny* are CC-BY but fully CG-animated. They
  are not live action, so they are deliberately left out (the anime/CG problem is the one this
  project is *not* solving).
- Tears of Steel has VFX/CG shots. That is fine (streaming content does too), but it is not pure
  camera footage.

## Research-only (opt in with `--include-research`)

| id(s) | What | Terms | Why not default |
|---|---|---|---|
| `ntia_*` | NTIA/ITS HDTV 1080p sources (aspen, red_kayak, speed_bag, touchdown_pass, west_wind_easy) | `HDTV_Readme.txt`: "may be used for research purposes, only", with acknowledgment kept with derived works | research only |
| `xiph_netflix_*` | Netflix Chimera clips as mirrored on Xiph | CC BY-NC-ND 4.0 per their copyright files | non-commercial, no derivatives |
| `div2k` | DIV2K stills (download yourself, point `path` at `DIV2K_train_HR`) | academic research only (NTIRE challenge terms) | research only |
| (not wired) Flickr2K | Flickr stills | per-image Flickr licences, distributed for research | research only |
| (not wired) HDGreetings clips on Xiph (dinner, factory, life) | 1080p30 | separate copyright notice at hdgreetings.com | unclear terms |
| (not wired) LIVE (UT Austin) VQA sets, BVI-DVC / BVI-HD (Bristol) | video-quality datasets | access on request, research/non-commercial terms | research only |
| (not wired) UVG (Tampere) | 4K test sequences | CC BY-NC 3.0 | non-commercial |

## What gets stored

`data/fetch.py` samples `positions` places per source, reads `clip_len` (default 6) consecutive
frames, optionally area-downscales them (`hr_scale`, to clean source noise and get 1080p-like HR
from 4K masters) and keeps `crops_per_position` 512x512 crops. Flat crops (luma std < 0.035) and
near-duplicates (dHash Hamming distance ≤ 5 within a source) are dropped. Crops are written as
16-bit RGB PNG, so 10-bit sources keep their precision. Val sources are whole held-out scenes or
clips (ToS frames 13000–14100, TUM pedestrian_area, SVT old_town_cross), so no scene leaks between
train and val.
