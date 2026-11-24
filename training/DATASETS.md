# Training data sources and licences

The weights we ship are meant to be MIT. Training on data does not automatically make the weights a
derivative work of that data, but the law is unsettled. So the **default dataset only uses sources
whose terms clearly allow reuse, modification and commercial use and do not forbid ML training**.
[`data/sources.yaml`](data/sources.yaml) has three tiers:

| tier | runs with | meaning |
|---|---|---|
| `permissive` | always (default) | public domain / no restrictions, CC BY, MIT-style |
| `sharealike` | `--include-sharealike` (or `--include-research`) | CC BY-SA: commercial use and derivatives are allowed, but if trained weights count as "adapted material" they would have to be BY-SA too. Decide before shipping weights trained on it |
| `research` | `--include-research` | research-only, non-commercial or no-derivatives terms. Do not ship weights trained on it under MIT |

Licences were checked against the files served next to the media, and against the publishers'
pages, in November 2026 (the check dates are in the tables). Re-check before a release, because
hosting sites change.

## Default (permissive) set

| id(s) in sources.yaml | What | Format we read | Licence / terms | Attribution |
|---|---|---|---|---|
| `tears_of_steel`, `tears_of_steel_val` | *Tears of Steel* (Blender Foundation, 2012): live action (Sony F65) with VFX | Lossless 1080p PNG master, `media.xiph.org/tearsofsteel/tearsofsteel-1080-png/` (17,620 frames; only sampled frames are downloaded) | **CC-BY 3.0** (`media.xiph.org/tearsofsteel/README.txt`). The soundtrack is CC-BY-ND, but we don't use it | "(CC) Blender Foundation \| mango.blender.org" |
| `netflix_elfuente_boat`, `netflix_elfuente_foodmarket` | Netflix *El Fuente* (2013 documentary) | 4096x2160 10-bit 4:2:0 BT.709 y4m in Netflix's Open Content bucket, read with HTTP range requests | **CC-BY 4.0**: `ElFuente/creative-commons-attribution-4-intl-public-license.txt` in the bucket, and the opencontent.netflix.com page says all Open Content is CC-BY 4.0 | "Netflix Open Content, CC BY 4.0" |
| `netflix_sparks_sdr` | Netflix *Sparks* (2017) | SDR UHD ProRes 422 BT.709 master (`sparks/sdr/…mov`, 46 GB), read remotely by ffmpeg seeks | **CC-BY 4.0** (`sparks/creative-commons-attribution-4-intl-public-license.txt`) | same |
| `netflix_nocturne_grain_sdr` | Netflix *Nocturne* (2018), with synthetic Vision3 film grain | SDR UHD ProRes (ACES 2.0 → 709), read remotely | **CC-BY 4.0** (`Nocturne/creative-commons-attribution-4-intl-public-license.txt`) | same |
| `tum_*` (blue_sky, riverbed, rush_hour, station2, sunflower, tractor; `tum_pedestrian_area` = val) | TU München 1080p25 test sequences, Sony HDW-F900 | 8-bit 4:2:0 y4m on the Xiph *derf* mirror | `ReadMe_1080p.txt`: "**Restrictions of use: No restrictions. Copyright: No Copyright**" | courtesy: "TU München, LDV" |
| `svt_*` (crowd_run, ducks_take_off, in_to_tree, park_joy; `svt_old_town_cross` = val) | SVT MultiFormat (Sveriges Television, 2001), 65 mm film scans | 1080p50 y4m on the Xiph *derf* mirror | SVT readme on the Xiph mirror: "**Restrictions of use: No restrictions. Copyright: No copyright**". Xiph links a separate SVT readme/PDF for the 1080p set too; worth re-reading before a release. Note that the BVI-AOM readme quotes a stricter SVT notice ("only … for developing, testing and presenting technology standards") for an SVT clip it redistributes | courtesy: "SVT Sveriges Television AB" |
| `custom_stills` (disabled) | Your own footage or photos | any image folder | yours | — |

### Added in November 2026 (checked 2026-11-24)

| id(s) in sources.yaml | What | Format we read | Licence / terms | Attribution |
|---|---|---|---|---|
| `netflix_chimera_*` (aerial, barscene1/2, dinnerscene, drivingpov, freeway, rollercoaster, toddlerfountain, windandnature; `netflix_chimera_pierseaside` = val) | Netflix *Chimera* (2014): dinner party, bar, driving, freeway, aerial, roller coaster, fountain, foliage. Faces, low light, water, motion | Uncompressed 3840x2160 10-bit 4:2:0 BT.709 `.yuv` in `Netflix_test_conditions/` (210–600 frames each). New `raw_yuv` reader with `band_rows: 1344`: only a 1344-row band of each sampled frame is fetched (~93 MB per position) | **CC-BY 4.0**: `Netflix_test_conditions/creative-commons-attribution-4-intl-public-license.txt` (2021-05-27) and opencontent.netflix.com (updated 2026-06-01). See "Netflix licence history" below for the older NC-ND notices | "Netflix Open Content, CC BY 4.0" |
| `netflix_elfuente_*` (boxingpractice, crosswalk, foodmarket2, narrator, ritualdance, tango, tunnelflag) | The other seven *El Fuente* scenes: boxing gym, street crossing, market, narrator close-up, dance, tango, flag in a tunnel | same `.yuv` layout, same folder | same | same |
| `netflix_meridian_sdr_*` (7 shots) | Netflix *Meridian* (2016, 1940s noir): faces, smoke, interiors at night, period costume, signage, cars | The official SDR version of seven shots (made from the Dolby Vision trim pass, "10-bit BT.709" per `aom_test_materials/README_SDR.txt`): 3840x2160 10-bit 4:2:0 JPEG 2000 frames with the reversible 5/3 wavelet (lossless coding), ~7.2 MB each, in one zip per shot (`aom_test_materials/SDR/meridian/`, 2.3–3.3 GB). New `zip_sequence` reader: the central directory is read with range requests, then only the sampled members (~43 MB per position) | **CC-BY 4.0** (`aom_test_materials/creative-commons-attribution-4-intl-public-license.txt`, 2021-05-27; `Meridian/creative-commons-…txt`; opencontent.netflix.com) | same |
| `netflix_meridian_hdr_tonemapped` | *Meridian*, the whole film (frames 240–34,500: no opening logo, no end credits) | 16-bit RGB TIFF master, HDR PQ / P3-D65 (`Meridian/tiffs/`, 43,093 frames × 50 MB), tone-mapped to SDR BT.709 (`tonemap: pq_p3d65`, see below). ~300 MB per position, the most expensive source per clip | **CC-BY 4.0** (`Meridian/creative-commons-…txt`, 2021-05-27) | same |
| `netflix_nocturne_sdr_*` (8 shots) | *Nocturne* without the synthetic grain (the existing `netflix_nocturne_grain_sdr` has grain baked in) | Official SDR JPEG 2000 shots in zips (`aom_test_materials/SDR/nocturne/`), ~3.6 MB per frame | **CC-BY 4.0** (as Meridian) | same |
| `uvg_*` (highway_view, floorball_train, floorball_game, spot_robot, volleyball_game, winter_drive, highway_drive, traffic_lights, office_walk, boat_reverse, job_fair, car_park, parking_garage, karting_time, under_park, auditorium_walk; `uvg_campus_view` = val) | **UVG-VCM** (Ultra Video Group, Tampere University): sports (floorball, volleyball), traffic, driving in snow, campus, offices, a parking garage with signage, karting | Uncompressed 3840x2160 60 fps **16-bit 4:4:4 full-range** `.yuv` (300–1,140 frames, 14–53 GB each), `band_rows: 1280` (~177 MB per position); the two stereo scenes are 1080p 8-bit 4:2:2 (left camera only). Read as BT.709. At 1:1, flat areas show texture from the camera's own codec, so it is area-downscaled to 0.4–0.55x | **CC-BY 4.0**: tie-ultravideo.rd.tuni.fi/UVG-VCM/ ("20 uncompressed and annotated video sequences released under the CC BY 4.0 license", page modified 2026-11-23). The three `Synthetic*` sequences are CG and are not used | "UVG-VCM, Ultra Video Group, Tampere University, CC BY 4.0" + the paper they ask to cite: T. Partanen et al., "UVG-VCM: Benchmarking dataset for machine-oriented visual data compression", QoMEX 2026 |

Ship a NOTICE with the weights that lists the CC-BY sources and their attribution lines. CC-BY asks
for attribution "reasonable to the medium", and a NOTICE next to the weights is the conservative
way to give it.

### Netflix licence history (why the bucket copies count as CC BY 4.0)

The same Netflix footage has carried different terms over the years, so this is the part to
re-check first before a release:

- **2015–2020: CC BY-NC-ND 4.0.** The `*_Copyright.txt` files next to the Chimera / El Fuente clips
  on the Xiph mirror (`media.xiph.org/video/derf/`), the AOM CTC copies
  (`media.xiph.org/video/aomctc/test_set/*/…_Copyright.txt`, "Copyright (C) NETFLIX INC. 2020"),
  `Netflix_test_conditions/Chimera_copyright.txt` and `ElFuente_copyright.txt` (2019-10-17) and the
  table in `aom_test_materials/README.txt` (2020-06-15, lists Sparks, Meridian and Nocturne as
  BY-NC-ND) all say BY-NC-ND. The per-title NC-ND licence files that README links
  (`TechblogAssets/Sparks/sparks_license.txt`, `Nocturne/nocturne_license.txt`,
  `SolLevante/sollevante-cclicense.txt`) **no longer exist** (HTTP 404, checked 2026-11-24).
- **Since 2021-05-27: CC BY 4.0.** Netflix put an identical
  `creative-commons-attribution-4-intl-public-license.txt` ("Our open source content is available
  under the Creative Commons Attribution 4.0 International Public License") into every folder of
  the bucket, including `Netflix_test_conditions/`, `aom_test_materials/`, `Meridian/`, `Chimera/`,
  `ElFuente/`, `sparks/`, `Nocturne/` and `av2ctc/` (same MD5 everywhere, S3 Last-Modified
  2021-05-27). opencontent.netflix.com (updated 2026-06-01) says: "Our open source content is
  available under the Creative Commons Attribution 4.0 International Public License."

A licensor can always grant broader terms than an earlier licence. The newer grant sits in the
same folders as the files we read, and Netflix's own page confirms it, so we treat the **bucket
copies as CC BY 4.0** (as this file already did for El Fuente, Sparks and Nocturne). The older files
were never withdrawn, though, so the exact files we read still have stale NC-ND notices next to
them. If you want zero ambiguity before a release, ask Netflix Open Content for a one-line
confirmation. Until then, keep reading the bucket copies only, never the Xiph / AOM-CTC mirrors.

### HDR → SDR tone map (`tonemap: pq_p3d65`)

Meridian's full-length master and Chimera's TIFFs are HDR (SMPTE ST 2084 PQ, P3-D65). The network
learns SDR BT.709 video, so a source with `tonemap: pq_p3d65` goes through
`lasr.ffmpeg.tonemap_pq_to_sdr` before cropping:

```
zscale  PQ -> linear light, P3-D65 RGB, 203 cd/m2 (BT.2408 HDR reference white) = 1.0
zscale  gamut P3-D65 -> BT.709 (linear)
tonemap mobius, desat=0
zscale  BT.709 OETF, full-range RGB
```

**Validation.** Netflix also publishes official SDR versions (from the Dolby Vision trim pass) of
seven Meridian shots, and their frame numbers map one to one onto the TIFF master. Four frames
from four different shots, tone-mapped by the pipeline's own code and compared with the official
SDR frame:

| TIFF frame (shot) | PSNR-Y | PSNR-RGB | corr(Y) | median Y ours / official | p99 Y ours / official |
|---|---|---|---|---|---|
| 1982 (1782–2163, office, smoke) | 34.04 dB | 31.70 dB | 0.9980 | 0.187 / 0.184 | 0.958 / 0.956 |
| 21088 (20988–21412, dark) | 36.14 dB | 34.14 dB | 0.9962 | 0.127 / 0.114 | 0.385 / 0.401 |
| 12564 (12264–12745, car interior) | 34.56 dB | 32.52 dB | 0.9964 | 0.108 / 0.092 | 0.816 / 0.778 |
| 24108 (24058–24550) | 34.61 dB | 32.79 dB | 0.9968 | 0.128 / 0.115 | 0.585 / 0.567 |

The alternatives scored worse on the same frames: Hable 20–26 dB, Reinhard 32–35 dB, and (tested on
the first frame only) 100 cd/m² as reference white 19–28 dB. A side-by-side of frame 12564 is visually indistinguishable
from the official grade. The remaining difference is mostly a slightly lifted black floor (p1 ≈
0.015 vs 0.0), which the trim pass crushes. `tests/test_sources.py` pins the curve (grey ramp:
100 cd/m² → 0.72, 203 → 0.84, monotonic, neutral stays neutral). Only `pq_p3d65` is enabled,
because only it was validated. HLG or BT.2020 masters would need the same kind of check first.

Chimera is **not** read through the tone map: its SDR versions (the `Netflix_test_conditions/`
`.yuv` files) are used instead. Official SDR is always better than any tone map.

## Share-alike set (opt in: `--include-sharealike`)

| id(s) | What | Format | Terms (per-clip `*_Copyright.txt` next to each file, checked 2026-11-24) |
|---|---|---|---|
| `aom_neon1224` | Neon signs at night, 4K | 3840x2160 10-bit y4m, `media.xiph.org/video/aomctc/test_set/a1_4k/` | CC BY-SA 4.0, © James Leslie Skinner |
| `aom_worldcup`, `aom_worldcup_far`, `aom_skater227`, `aom_vertical_carnaby`, `aom_vertical_bees` | Football crowd, skater, Carnaby Street (vertical), bees (vertical) | 1080p y4m (`a2_2k/`) | CC BY-SA 4.0, © James Leslie Skinner. WorldCup's file says BY-SA but links the BY legal code, so treat it as BY-SA |
| `aom_walkinginstreet` | Street walk (vertical) | 1080x1920 y4m | CC BY-SA 4.0, © Victor Cherepanov |
| `aom_motorcycle`, `aom_mountainbike` | Motorcycle, mountain bike (fast motion) | 1080p 8-bit y4m | CC BY-SA 4.0, © Neil Birkbeck |
| `aom_treesandgrass` | Foliage | 1080p 8-bit y4m | CC BY-SA 4.0, © Hangyu Kuang |

They are uncompressed y4m as uploaded, but the camera originals were compressed (8-bit, phone and
action-camera looks), and each clip is only 130 frames long. Attribution: "<title> © <author>, CC
BY-SA 4.0, via the AOM common test conditions".

## Research-only (opt in with `--include-research`)

| id(s) | What | Terms | Why not default |
|---|---|---|---|
| `ntia_*` | NTIA/ITS HDTV 1080p sources (aspen, red_kayak, speed_bag, touchdown_pass, west_wind_easy) | `HDTV_Readme.txt`: "may be used for research purposes, only", with acknowledgment kept with derived works | research only |
| `xiph_netflix_*` | Netflix Chimera clips as mirrored on Xiph | CC BY-NC-ND 4.0 per their copyright files | non-commercial, no derivatives (the same footage from the Netflix bucket is in the default set, see above) |
| `div2k` | DIV2K stills (download yourself, point `path` at `DIV2K_train_HR`) | academic research only (NTIRE challenge terms) | research only |

## Rejected (checked 2026-11-24)

Not wired at all, with the reason. "Licence OK" rows were rejected for quality or access reasons
only and could be revisited.

| Source | Terms found (primary source) | Reason |
|---|---|---|
| **Pexels** (photos, video) | pexels.com/terms-of-service (2026-10-20) §8: "Data mining, extraction, scraping and the use of programs or robots for automatic data collection … is strictly prohibited for all unauthorised purposes, including without limitation for machine learning purposes"; "Bulk, large-scale or systematic copying of Content is strictly prohibited" | ML use and bulk copying forbidden |
| **Pixabay** | pixabay.com/service/terms (2026-10-20) §8: same wording as Pexels. API docs: "Systematic mass downloads are not allowed" | ML use and bulk copying forbidden |
| **Unsplash** | unsplash.com/terms §8: may not "Use the Images in connection with any machine learning and/or artificial intelligence datasets"; no bots/scrapers outside the API | ML datasets forbidden |
| **Unsplash Lite / Full datasets** | github.com/unsplash/datasets TERMS.md: training only "for your internal business purposes", no redistribution; Full is non-commercial | publicly shipped weights are not internal use |
| **Mixkit** | mixkit.co/terms (2025-10-02) §9: no "scripts or bots to mass download Items", no scraping the library, no building "a similar or competitive product" | scripted downloading forbidden |
| **Videvo** | videvo.net licence pages now redirect to Freepik/Magnific, whose terms (§8.1) forbid use "for any machine learning and/or artificial intelligence purposes" and inclusion "in a database" | ML use forbidden |
| **Coverr** | coverr.co/license (2026-11-04): "Nothing from Coverr may be used to train, fine-tune, distill or evaluate AI models, or go into a dataset for that purpose" | ML use explicitly forbidden |
| **UVG (original 16 sequences)** | ultravideo.fi/dataset.html: "available under a non-commercial Creative Commons BY-NC license" (CC BY-NC 3.0) | non-commercial (the newer UVG-VCM set is CC BY 4.0 and is in the default set) |
| **BVI-DVC** (Bristol) | data.bris.ac.uk: "Non-Commercial Government Licence for public sector information" | non-commercial |
| **BVI-AOM** (Bristol, `bvi_aom_dataset/` in the Netflix bucket) | `README_bviaom.txt`: "The material shall only be used for developing future video coding standards, for training or performance evaluation of test models in JVET and Alliance for Open Media…"; mixes Videvo, CableLabs (CC BY-NC-ND 3.0), ASC StEM2 and other sources. Also distributed only as 6–23 GB tar.gz | standards work only; whole archives only |
| **Inter4K** | licence.txt in the GitHub repo: CC BY-NC-SA 4.0 | non-commercial |
| **LSDIR** | ofsoundof.github.io/lsdir-data: "made available for academic research purpose only… copyright belongs to the original owners" | research only |
| **DIV8K** | no licence or official download page found; same team as DIV2K (academic research only) | unclear, assume research only |
| **Flickr2K** | no licence; "collected by ourselves using Flickr API"; every photo keeps its own Flickr copyright | unclear per-image rights |
| **SJTU 4K** | medialab.sjtu.edu.cn: "available for free and academic research only, no commercial use" | research only |
| **Vimeo-90K** | no licence on toflow.csail.mit.edu; frames scraped from Vimeo videos; 448x256 | unclear, and far below 1080p |
| **DAVIS** | the repo's BSD licence covers the code only; no terms for the frames | unclear |
| **YouTube-8M** | CC BY 4.0, but it ships features and video IDs, no pixels (the videos fall under YouTube's terms) | no image data |
| **MCL-JCV** | MIT-style permission notice (mcl.usc.edu); download behind a shared login | licence OK, but every clip is H.264-encoded (QP 1–51): no clean HR; login-gated |
| **REDS** | seungjunnah.github.io/Datasets/reds.html: "released under CC BY 4.0 license" | licence OK, but the HR frames are 1280x720 (downscaled GoPro footage) and come only as multi-GB zips on Google Drive / Hugging Face. Below the ≥1080p HR target. Could be added later with `zip_sequence` if the zips are reachable by range requests |
| **AOM CTC Netflix clips** (Xiph `aomctc/test_set`) | per-clip copyright files, "Copyright (C) NETFLIX INC. 2020", CC BY-NC-ND 4.0 | NC-ND notice. We read the same footage from the Netflix bucket instead |
| **AOM CTC RushFieldCuts** | NTIA: "may be used for research purposes, only" | research only |
| **AOM CTC Tencent stills** (`f1_still_HiRes/Tencent`) | "limited right to display or publish the photographs for the purpose of developing, testing and promulgating technology standards" | standards work only |
| **AOM CTC Apple stills** (`f1_still_HiRes/Apple`) | "CC-BY-SA-4.0 … © 2020 Apple Inc." (links the BY legal code) | BY-SA, and they are processed iPhone photos stored as single-frame y4m, which no reader handles yet. Candidate for the sharealike tier later |
| **EBU UHD-1 test set** | EBU test-sequence policy PDF (2015): the public test set is CC BY-NC-ND, the full set is licensed for a fee without redistribution. *Read through search snippets only, not re-fetched* | non-commercial / paid |
| **JVET / ITU-T CTC sequences** (Campfire, Tango2, …) | distributed to standards participants; no public licence; JVET calls forbid training on the test set | standards work only |
| Netflix **Sol Levante**, **Cosmos Laundromat**, Blender **Sintel**, **Spring**, **Big Buck Bunny** | CC BY (Sol Levante's old README row says NC-ND) | animation / CG, not live action (the anime/CG problem is the one this project is *not* solving) |
| **HDGreetings** clips on Xiph (dinner, factory, life) | separate copyright notice at hdgreetings.com | unclear terms |
| **LIVE (UT Austin) VQA sets, BVI-HD** | access on request, research / non-commercial terms | research only |

Vendor sample footage (ARRI, RED, Blackmagic, Sony), NASA / US-government public-domain video,
Wikimedia Commons and Internet Archive public-domain scans are covered in
[Public-domain and vendor footage](#public-domain-and-vendor-footage) below.

## Public-domain and vendor footage

Checked 2026-11-24. None of these is wired. The reasons are licence or quality, as noted.

| Source | Terms (primary source) | Format | Verdict |
|---|---|---|---|
| **NASA SVS** (svs.gsfc.nasa.gov) | "All of our content is in the public domain (unless otherwise noted)". NASA insignia/logos are not public domain; identifiable people may have publicity rights | mostly .mp4/.webm, some frame sequences (.exr/.tif) | Licence OK, but almost all of it is CG visualisation, not live action. A per-item curated live-action list would be a later addition |
| **NASA images.nasa.gov** | NASA media "generally are not subject to copyright in the United States"; third-party material and identifiable persons need a per-item check | lossy MP4 delivery copies | needs per-item screening; lossy |
| **Wikimedia Commons** video | per file (CC0 / PD / CC BY / CC BY-SA); bulk clients must send an identifying User-Agent with contact details | only free lossy codecs (WebM VP9/AV1, Theora, MPEG-1/2); originals are never mezzanine | lossy everywhere, per-file licence curation. Not worth it for clean HR |
| **Internet Archive / Prelinger** | "Many files are available under the Creative Commons Public Domain Dedication … many are not"; IA gives no written licences | free copies are ~12 Mbit/s H.264 MP4 or SD MPEG-2; the 4K/5K scans are sold separately | lossy and mostly sub-1080p; rights vary per item |
| **ARRI** sample footage (arri.com/samplefootage) | "created for workflow evaluation purposes"; no licence grant | ARRIRAW / ProRes 4444 | evaluation only |
| **RED** sample R3D | testing only, RED keeps all copyrights (from search snippets of the official page; the page itself redirected) | R3D 6K–8K | forbidden beyond testing |
| **Blackmagic** BRAW samples | site-wide "All items on this website are copyright Blackmagic Design … all rights reserved"; no clip licence | .braw | no grant |
| **Sony** VENICE 2 X-OCN samples | behind registration (page returned 403); no clip licence found | X-OCN / ProRes 8K | unclear, assume evaluation only; registration-gated |
| **Blender** open movies | — | — | Tears of Steel is still the only live-action film (already in the default set) |

## What gets stored

`data/fetch.py` samples `positions` places per source, reads `clip_len` (default 6) consecutive
frames, optionally area-downscales them (`hr_scale`, to clean source noise and get 1080p-like HR
from 4K masters) and keeps `crops_per_position` 512x512 crops. Flat crops (luma std < 0.035) and
near-duplicates (dHash Hamming distance ≤ 5 within a source) are dropped. Crops are written as
16-bit RGB PNG, so 10-bit sources keep their precision. Raw sources with `band_rows` only fetch a
horizontal band of each frame (the crops come from inside the band), and HDR sources with
`tonemap` are tone-mapped before scaling. Val sources are whole held-out scenes or clips (ToS
frames 13000–14100, TUM pedestrian_area, SVT old_town_cross, Chimera PierSeaside, UVG-VCM
CampusView), so no scene leaks between train and val. `tests/test_sources.py` checks that no media
file is read by both a train and a val source.

Size of the default set (positions × crops, before the flat/duplicate filter): ~6,350 train + ~180
val clips (before November 2026: ~2,940 + 140). Download: roughly 100–115 GB, ~77 GB of it for
the new sources (the per-position costs are in the comments of `sources.yaml`). The fetch is
bandwidth-bound: the new sources take about 77 GB / downstream bandwidth (~11 min at 1 Gbit/s,
~1.7 h at 100 Mbit/s) plus a few minutes of JPEG 2000 decoding and tone mapping spread over the
threads. Measured from a European desktop line: ~12–15 MB/s per connection to the Netflix bucket
(us-east S3) and ~14 MB/s to UVG-VCM, so use `--threads 16`.
