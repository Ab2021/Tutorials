# Whole-Slide Imaging Pipeline Infrastructure

> **Module 10 — Medical Imaging and Pathology AI**
> Topic 11 of 13
> Estimated study time: 4 hours
> Prerequisites: Topics 01, 02, 03, and 05
> Version 2026.3

## 1. Why this topic exists

A pathology laboratory produces glass slides. A scanner turns each slide into a file that is
larger than any image most data scientists have handled: a 15 mm by 15 mm tissue section scanned
at 40x is 60,000 by 60,000 pixels, 3.6 gigapixels, 10.8 GB of raw red-green-blue (RGB) bytes
before compression. A hospital that scans 600 slides per day writes roughly 180 TB of image data
per year, at 250 working days and 1.2 GB per slide. Nothing in a standard computer-vision
tutorial prepares a team for that.

The industry problem is not that pathology models are hard to train. It is that the pixels never
arrive in the shape the model expects, and the failures are quiet. A pipeline that maps tile
coordinates one level off will produce embeddings for the wrong tissue, train a model that
converges, report a plausible area under the receiver operating characteristic curve (AUROC),
and fail in deployment. A pipeline that reads one tile per web request will work on ten slides
and collapse on six hundred. A pipeline that stores every extracted patch as a separate image file will run out
of budget before it runs out of disk.

Four decisions are blocked until someone can do this arithmetic:

| Decision | What pipeline knowledge provides |
|---|---|
| Scanner and format choice | Whether the output can be read at the analysis magnification without conversion |
| Storage and retention budget | Bytes per slide, pyramid overhead, and the cost of keeping patches versus recomputing them |
| Serving architecture | Whether inference streams tiles from object storage or reads a materialised patch store |
| Failure diagnosis | Whether a bad result is a model problem, a coordinate problem, or a data problem |

The topic is deliberately infrastructural. Section 5 walks a pipeline from scanner output to a
manifest of patch coordinates. Section 6 carries one laboratory's numbers end to end and shows
that the compute is usually the cheap part while request latency and coordinate correctness are
the expensive parts. Section 7 reproduces every number in standard-library Python.

### 1.1 Why radiology patterns do not transfer

Radiology pipelines assume a study is a few hundred megabytes, a voxel grid is regular, and the
whole volume fits in memory after resampling. Whole-slide images (WSIs) break all three
assumptions at once:

1. **Memory.** A single slide at full resolution exceeds the memory of any accelerator. End-to-end
   backpropagation over a slide is not a tuning problem; it is not possible.
2. **Sparsity.** The diagnostic object may be a cluster of 200 pixels on a 3.6-gigapixel canvas.
   Most of the slide is glass, and the fraction that is tissue is itself mostly uninformative
   stroma. The pipeline exists to discard pixels without discarding the diagnosis.
3. **Heterogeneity.** Scanner make, staining protocol, section thickness, and coverslip mounting
   all change the pixel values. Two images of the same tissue from two hospitals can differ more
   than two different tissues from one hospital.

A radiology pipeline that is 90 percent correct still produces a usable image. A WSI pipeline that
is 99 percent correct can silently drop the one region that contains the metastasis, because the
tissue mask excluded it or the tile grid stepped over it.

### 1.2 The cost of getting it wrong

The failures are not hypothetical. Three patterns recur in production reviews:

- **The mask that learned the scanner.** A threshold tuned on one scanner's white balance
  excludes tissue on a second scanner with a darker glass background. Patch counts fall by
  30 percent on that site, sensitivity falls with them, and every slide still logs "success".
- **The level mismatch.** A model trained on 20x patches is deployed on 40x level-0 tiles, so it
  sees cells at twice the expected scale. Accuracy degrades gradually and the team spends weeks
  tuning the model instead of the loader.
- **The retry that duplicated work.** A transient storage error triggers a retry that re-embeds a
  slide already written. Duplicate rows enter the training set, the same patient appears on both
  sides of a split, and the reported result is inflated.

Each of these is cheap to prevent with an explicit contract and expensive to discover after a
validation study.

## 2. Learning outcomes

By the end of this topic you can:

1. **Compute** the pixel matrix, pyramid geometry, tile count, compressed size, and patch count
   for a stated scanner output, showing units at every step.
2. **Distinguish** the container formats SVS, NDPI, MRXS, and Digital Imaging and Communications in
   Medicine (DICOM) whole-slide imaging by their structure, metadata, and trade-offs.
3. **Design** a tissue-masking and artefact-filtering stage, and justify its thresholds against a
   measured rather than assumed tissue fraction.
4. **Choose** between streaming tiles at inference and materialising a patch store, using a
   working-set and re-use calculation rather than preference.
5. **Diagnose** a throughput problem as bandwidth-bound, latency-bound, compute-bound, or
   lock-bound from measurements the pipeline should already emit.
6. **Specify** an orchestration contract with idempotency, retries, and provenance so that a
   re-run cannot duplicate or silently skip a slide.
7. **Audit** a pipeline for coordinate and metadata defects before a model is trained on its
   output.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

A whole-slide image is a pyramid of tiles stored inside a container, and a pipeline is a sequence
of filters that removes pixels before any model sees them. Hold two ideas at once: the slide is
a **file**, and the slide is a **coordinate system**. Most production bugs are a confusion
between the two.

```text
glass slide (15 x 15 mm)
        |
        v
   scanner  ------------------------------> container file
   (40x, 0.25 um/px)                        SVS | NDPI | MRXS | DICOM
        |                                        |
        |                                        +--> label image  (text; PHI risk)
        |                                        +--> macro image  (overview)
        |                                        +--> pyramid L0 .. Ln
        |                                                   |
        |                                                   v
        |                                          tile grid, 256 x 256 px
        |                                                   |
        |                                        tissue mask (drop glass)
        |                                                   |
        |                                        artefact mask (pen, folds, blur)
        |                                                   |
        |                                                   v
        |                                          patch manifest (x, y, level)
        |                                                   |
        |                              +--------------------+--------------------+
        |                              v                                         v
        |                         tile reader                              QC metrics
        |                              |                                         |
        |                              v                                         v
        |                     encoder (ViT / CNN)                        quarantine or pass
        |                              |
        |                              v
        |                     embeddings  N x D  (float16)
        |                              |
        +------------------------------+--> slide aggregator --> score
```

### 3.1 The information funnel

The pipeline's job is to reduce 3.6 gigapixels to something a model can consume without losing
the signal. For the 15 mm by 15 mm slide above:

| Stage | Count | Typical unit size | Data volume |
|---|---:|---|---:|
| Level-0 pixels | 3.6e9 | 3 bytes | 10.8 GB raw |
| Level-0 tiles | 55,225 | about 16 KB compressed | 0.90 GB |
| Analysis-level tiles (20x) | 13,924 | about 16 KB | 0.23 GB |
| Tiles passing tissue mask | 4,873 | about 16 KB | 79 MB read |
| Embeddings | 4,873 | 2,112 bytes | 10.3 MB |

The reduction from 10.8 GB to 10.3 MB is roughly a thousandfold. Every step in that funnel is a
decision that can be wrong, and a wrong decision at the top cannot be repaired at the bottom.

### 3.2 Two clocks and one coordinate system

A pipeline is bound by whichever is slower: the **data clock** (how fast bytes arrive) or the
**compute clock** (how fast the encoder consumes them). In the worked example the compute clock
dominates at 10.8 s per slide against 7.6 s of request latency, but the two are close enough that
naive serialisation doubles the wall-clock time. The first design question is therefore not "how
fast is the accelerator" but "which clock is the constraint at this scale".

The coordinate system is a separate axis and it is not a performance question. A tile has a
**level** index, which fixes the microns per pixel; a **column and row** within that level's tile
grid; and an **origin in level-0 pixels** for comparison across levels and overlay. A pipeline
that records only `(x, y)` cannot be audited later, and one that records level-0 origins but reads
at a coarser level requests the wrong bytes. Store all three, always.

### 3.3 The pyramid as a physical fact

The pyramid is not an optimisation added by software; it is what makes the slide viewable. Each
level halves the linear resolution, so each level has one quarter of the pixels of the one above
it. The whole pyramid therefore costs about one third more than level 0 alone:

```text
total = L0 + L0/4 + L0/16 + ... = L0 * 4/3
```

That 33 percent overhead is a real storage line item. For a laboratory at 180 TB per year it is
about 45 TB per year of pixels that exist only so a human can zoom out quickly.

### 3.4 What the model actually receives

The encoder receives a fixed-size tensor, for example 256 by 256 by 3 bytes. It does not receive
the slide, the level, the scanner identity, or the stain protocol. Everything the model can learn
about those factors must be either present in the pixels or absent from the label. This is why
pipeline decisions become model behaviour: choosing the analysis level chooses the object scale,
and choosing the stain normalisation chooses which colour differences the model can see.

> **Note:** A pipeline is not a preprocessing detail that sits before the science. At slide scale
> the pipeline determines the label distribution, the effective resolution, and the site
> signature. Treat its outputs as part of the dataset definition.

## 4. Definitions and vocabulary

The terms below are used throughout this file and the two that follow it. The "common misuse"
column records the ambiguity that causes the most production defects.

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Whole-slide image (WSI) | A digital representation of an entire glass slide, stored as a pyramidal tiled image with associated label and macro images | Used to mean any pathology image, including a single region-of-interest crop | File |
| Base magnification | The objective power at which level 0 was scanned, typically 20x or 40x | Treated as a synonym for resolution; two 40x scanners can differ in microns per pixel | x |
| Microns per pixel (MPP) | Physical size of one level-0 pixel; the only scale that is comparable across scanners | Omitted from model cards, so a 20x model is applied to 40x data | um/px |
| Pyramid level | An index into the downsampled representations of the slide, with level 0 at full resolution | Assumed to double exactly; some formats store non-power-of-two levels | Integer index |
| Downsample factor | The ratio of level-0 pixels to pixels at a given level, usually 2^level | Confused with the tile size in pixels | Dimensionless |
| Tile | A fixed-size square of pixels stored as one compressed unit, typically 256 by 256 or 512 by 512 | Called a patch when it has not yet passed the tissue mask | px |
| Patch | A tile selected for analysis, usually after masking and sometimes after resizing | Used interchangeably with tile, which hides whether masking happened | px |
| Tile grid | The arrangement of tiles covering a level, with columns and rows | Assumed to start at (0,0) in level-0 coordinates; it starts at the level origin | Grid |
| TILED_FULL | A DICOM dimension organisation type in which frames are stored in a defined order and per-frame positions may be omitted | Assumed for every DICOM WSI; TILED_SPARSE is also legal | DICOM code |
| TILED_SPARSE | A DICOM organisation in which each frame carries its own position and optical path | Ignored by readers that only handle TILED_FULL, producing scrambled images | DICOM code |
| Total Pixel Matrix | The DICOM attributes giving the full pixel extent of a slide at a level, including empty glass | Confused with the imaged volume in millimetres | px |
| Associated image | A non-pyramid image stored with the slide: label, macro, or thumbnail | The label image is read as data and leaks text into features or logs | Image |
| Optical path | The light path used for a frame, distinguishing brightfield from fluorescence channels | Assumed to be one path per slide; multiplexed slides have several | DICOM sequence |
| Focal plane | The z position of a frame; some scanners store multiple planes per tile | Flattened without recording which plane was chosen | Index |
| SVS | Aperio (Leica) single-file pyramidal Tagged Image File Format (TIFF) with vendor metadata and Joint Photographic Experts Group (JPEG) or JPEG 2000 tiles | Assumed to be a standard TIFF that any TIFF library can read correctly | Container |
| NDPI | Hamamatsu single-file TIFF-like container with non-standard tags and restart-marker JPEG levels | Opened with a generic TIFF reader, which fails or reads only the first level | Container |
| MRXS | 3DHISTECH multi-file container: an INI index plus a data directory of compressed tiles | Copied without the companion directory, producing a broken slide | Container |
| DICOM WSI | The DICOM VL Whole Slide Microscopy Image information object, defined in Supplement 145 | Treated as a viewer format only; it is also an archive and interchange format | Standard |
| OpenSlide | A C library with Python bindings that reads SVS, NDPI, MRXS, and other vendor formats | Assumed to be thread-safe per handle; it is not | Library |
| tiffslide | A pure-Python reader for pyramidal TIFF formats, API-compatible with OpenSlide for common operations | Assumed to support every vendor variant | Library |
| Tissue mask | A binary mask, usually computed on a thumbnail, marking pixels that are tissue rather than glass | Tuned once and never re-validated on a new scanner | Mask |
| Otsu threshold | A method that chooses a threshold maximising between-class variance of a grayscale histogram | Applied to colour images without converting to a suitable space | Method |
| Optical density (OD) | `-log10(I / I0)`, the absorbance of a pixel, linear in stain concentration | Computed without clamping zeros, producing infinities | Dimensionless |
| Stain vector | The direction in OD space corresponding to one stain, for example haematoxylin or eosin | Estimated per tile rather than per slide, adding noise | Unit vector |
| Stain normalisation | A transform that maps a slide's stain appearance toward a reference | Applied inconsistently between training and inference | Transform |
| Tile cache | A bounded store of recently read tiles, keyed by level and tile coordinate | Sized below the working set, so it never hits | Tiles |
| Range request | A Hypertext Transfer Protocol (HTTP) request for a byte range of an object, used to fetch one tile without the whole file | Issued per tile with no connection reuse, so latency dominates | HTTP |
| Object storage | A store addressed by key with whole-object or range reads, such as S3 or Azure Blob | Treated as a POSIX filesystem with cheap `stat` and append | Service |
| Manifest | A table of patch coordinates and metadata produced by the tiling stage | Regenerated per run, so two runs disagree and cannot be compared | Table |
| Shard | A fixed-size output file holding embeddings or patches for many slides | Sized so that one shard rewrite costs hours | File |
| Idempotency | The property that repeating an operation produces the same stored result | Assumed from a queue's at-least-once delivery, which does not provide it | Property |
| Backpressure | A mechanism that slows producers when consumers are saturated | Replaced by unbounded queues that grow until memory fails | Control |
| Provenance | The recorded origin of a derived artefact: scanner, scan time, software version, parameters | Reconstructed from filenames after an audit request | Record |

## 5. The mechanics, step by step

This procedure takes a directory of scanner output and produces a manifest of patch coordinates,
a set of embeddings, and the records needed to reproduce both. It assumes H&E brightfield slides
and a downstream model that consumes fixed-size patches. Steps 1 to 4 are per scanner population;
steps 5 to 15 are per slide; steps 16 to 18 are per pipeline.

The sequence is written as a line, but masking and normalisation are iterative: expect to revisit
steps 8, 9, and 14 against a labelled sample before the thresholds are stable.

### Step 1. Inventory the scanner output

**Input:** The scanner export directory or the laboratory information system (LIS) scan log.

**Output:** One row per slide: identifier, scanner make and model, container format, file size in
bytes, scan timestamp, stated microns per pixel, and objective power.

**Rationale:** Every later parameter is a function of the scanner population. A pipeline tuned on
one vendor's 0.25 um per pixel output will silently mis-scale a second vendor's 0.5 um per pixel
output. The inventory makes that population visible before per-slide work begins.

**Branch points:**
- One scanner, one format: proceed with fixed parameters, recorded as assumptions.
- Mixed fleet: stratify the manifest by scanner and validate the mask per stratum.
- Unsupported format: quarantine before extraction, not after.

**Edge case:** File size does not identify a truncated scan. A slide that failed mid-scan can be
half the expected size and still open. Step 2 catches it.

### Step 2. Validate the container

**Input:** One slide file, and for MRXS the companion directory.

**Output:** A pass or fail record with a level inventory: index, width and height in pixels, tile
size, and compression for each level.

**Rationale:** Vendor containers fail in format-specific ways. An SVS file is a TIFF whose first
directory is tiled, but a generic TIFF reader may return only the thumbnail. An NDPI file is
TIFF-like with unsorted tags and an out-of-line ASCII layout that libtiff rejects. An MRXS file is
an index plus a data directory, and copying the `.mrxs` file alone yields an unreadable slide.
Validation must use a format-aware reader and must check that level dimensions decrease
monotonically and that the tile grid covers the stated extent.

**Branch points:**
- Levels missing or non-monotonic: fail and quarantine.
- Microns per pixel absent: derive from objective power and camera pixel size, flag as derived.
- Only one level: usable for analysis, not for thumbnail masking; generate a thumbnail.

**Edge case:** A slide can pass validation and still be out of focus; that is a step 17 decision.

### Step 3. Pin the metadata

**Input:** The validated container and its vendor metadata.

**Output:** An immutable record with a content hash of the source, scanner model, scan time,
objective power, level-0 microns per pixel, stain protocol if present, and reader version.

**Rationale:** Metadata is evidence. Converters rewrite it, viewers normalise it, and a re-export
can change the stated microns per pixel. Pinning at ingestion lets a later audit distinguish "the
scanner said 0.25" from "the pipeline assumed 0.25". The content hash also makes step 16
idempotent.

**Branch points:**
- Complete and consistent: proceed.
- Contradictory (objective says 40x, microns per pixel says 0.5): prefer the physical measurement,
  flag the conflict for the laboratory.
- Absent: derive, flag as derived, exclude from validation without review.

**Edge case:** The label image usually contains text. Reading it copies patient identifiers into
logs. Treat the label as protected health information and exclude it from feature extraction.

### Step 4. Choose the serving format

**Input:** The validated container, the analysis plan, and the archive requirement.

**Output:** A decision to keep the native format, convert to DICOM WSI, convert to pyramidal
TIFF, or convert to an internal tile store, with conversion cost recorded.

**Rationale:** Conversion is neither free nor always reversible. It changes bytes on disk, the
metadata model, and often the compression. The decision follows from who reads the slide and for
how long, not from which format the team finds pleasant.

| Situation | Keep native | DICOM WSI | Pyramidal TIFF | Internal tile store |
|---|---|---|---|---|
| Vendor viewer is the primary reader | Yes | No | No | No |
| One archive for radiology and pathology | No | Yes | No | No |
| Research cohort, one reader, hashed input | Possible | Possible | Yes | Possible |
| Millions of random tile reads per training job | No | Possible | Possible | Yes |
| Viewer-neutral regulatory retention | No | Yes | No | No |

**Edge case:** Re-encoding JPEG tiles changes pixel values; inference must use the same conversion
as training.

### Step 5. Fix the analysis magnification

**Input:** The model's required microns per pixel and the slide's level inventory.

**Output:** The level index to read, and a flag if no level matches.

**Rationale:** Object scale is a model input even when it is not a tensor input. A model trained
at 20x and applied at 40x sees nuclei at twice the diameter and texture at twice the frequency.
The correct level is the finest level whose microns per pixel is at least the target: reading a
coarser level and upsampling adds no information, while a finer level can be downsampled exactly.

| Target versus slide | Outcome | Action |
|---|---|---|
| Slide finer, exact power-of-two downsample exists | Clean reduction | Read the matching level; no resampling |
| Slide finer, no exact level | Approximate reduction | Read nearest finer level, resize, record interpolation |
| Slide equal to target | Exact match | Read level 0 |
| Slide coarser than target | Detail cannot be recovered | Rescan at higher power, or accept and re-validate the model |
| Metadata missing | Scale unknown | Quarantine; do not guess |

**Edge case:** Some scanners store level 0 at 40x but label the file 20x because the viewer
defaults to 20x. Trust microns per pixel over the label.

### Step 6. Build the access plan

**Input:** The chosen level, the tile size, and whether patches need surrounding context.

**Output:** A function from patch index to (level, tile column, tile row) and to a level-0 origin
in pixels.

**Rationale:** Three coordinate systems coexist: level-0 pixels, level pixels, and tile indices.
The conversion is `level0 = level_pixel * downsample` and `level_pixel = tile_index * tile_size`.
Mixing them is the most common silent bug in the pipeline. Writing the conversion once, as a
tested function, removes the class of bug.

**Branch points:**
- Non-overlapping patches: stride equals tile size.
- Overlapping patches, or patches crossing tile boundaries: stride below tile size or stitching
  across four tiles, either of which multiplies bytes read and grows the cache working set.

**Edge case:** The last row and column may fall partly outside the image, and most readers pad
with white. A model trained on padded tiles learns the padding as a feature. Exclude edge tiles
or record that padding is present.

### Step 7. Generate the thumbnail

**Input:** The validated slide.

**Output:** A thumbnail of roughly 1,000 to 2,000 pixels on the long side, plus the factor that
maps thumbnail pixels to level-0 pixels.

**Rationale:** Masking at full resolution is wasteful. The thumbnail is small enough to hold in
memory and large enough that a 256-pixel tile maps to a few thumbnail pixels. That mapping factor
is what lets a mask computed on the thumbnail filter tiles at the analysis level.

**Branch points:**
- A pyramid level matches the target size: read it directly.
- No level matches: read a coarser level and resize, or read a grid of tiles and downsample.
- Thumbnail corrupted or missing: fail; masking cannot proceed.

**Edge case:** A thumbnail generated from the macro image has a different aspect ratio and cannot
be used for masking.

### Step 8. Segment the tissue

**Input:** The thumbnail as an RGB array.

**Output:** A binary mask at thumbnail resolution and the tissue fraction of the slide.

**Rationale:** Glass is bright and desaturated; H&E tissue is darker with a characteristic hue
distribution. The saturation channel separates them more reliably than luminance, because a pale
section can be as bright as tinted glass. Otsu's method on saturation, followed by morphological
closing and removal of small components, gives a mask that is stable across scanners without
per-slide tuning.

**Branch points:**
- Tissue fraction 0.05 to 0.95: normal, proceed.
- Below 0.05: a very small biopsy or a failed mask; inspect before dropping.
- Above 0.95: the mask included background; the threshold is wrong.

**Edge case:** Fat and mucus are pale and can fall below a saturation threshold. A mask that
excludes fat systematically excludes a tissue type, which becomes a subgroup failure.

### Step 9. Mask the artefacts

**Input:** The thumbnail, the tissue mask, and the slide metadata.

**Output:** An artefact mask and a per-slide artefact report.

**Rationale:** Marker ink, air bubbles, folds, and knife chatter are not biology, but they are
high-contrast and can dominate a patch. A model trained without artefact masking learns that ink
at the margin predicts cancer, because margins are inked on resections. Removing artefacts at the
mask stage is cheaper than teaching the model to ignore them.

**Branch points:**
- Ink: high saturation, low hue variance, very low luminance. Exclude.
- Bubbles: bright rings with sharp edges. Exclude the enclosed region.
- Folds: high optical density, low local texture. Flag rather than exclude; folds can overlie
  tumour.
- Blur: low variance of the Laplacian. Exclude or flag depending on the model.

**Edge case:** A detector tuned on one laboratory's ink colour misses another's. The detector
needs a labelled sample per site, not a universal threshold.

### Step 10. Enumerate candidate tiles

**Input:** Level dimensions, tile size, and stride.

**Output:** Candidate tile coordinates at the analysis level, each with its level-0 origin.

**Rationale:** The count is `ceil(width / tile) * ceil(height / tile)`. Enumerating explicitly,
rather than looping with floating-point steps, prevents off-by-one drift at the right and bottom
edges. Carrying the level-0 origin means masking and overlay never reconstruct it.

**Branch points:**
- Full grid: simplest and auditable; use for cohort work.
- Grid restricted to the mask bounding box: faster, but loses the fraction of slide examined.
- Random sample: acceptable for embedding-only work, not for diagnosis.

**Edge case:** A large tissue area with a small tile size can produce hundreds of thousands of
candidates; stream the list or store it per slide rather than in memory.

### Step 11. Filter by tissue fraction

**Input:** Candidate tiles with level-0 origins, the mask, and the thumbnail-to-level-0 factor.

**Output:** Kept patches, each with its measured tissue fraction.

**Rationale:** The threshold trades recall against cost. A low threshold keeps border tiles and
costs compute; a high threshold can drop a small focus of tumour at a tissue edge. The right value
is task-dependent and should be measured against annotations, not chosen because 0.5 looks
reasonable.

| Measured tissue fraction | Interpretation | Action |
|---|---|---|
| 1.00 to 0.50 | Mostly tissue | Keep |
| 0.50 to 0.20 | Border tile, may contain diagnostic edge | Keep and flag as border |
| 0.20 to 0.05 | Mostly glass with a tissue sliver | Keep only for micrometastasis or small specimens |
| 0.05 to 0.00 | Glass | Drop |
| Slide fraction below 0.02 despite a non-empty label | Suspected mask failure | Fail the slide; do not emit an empty manifest |

**Edge case:** A needle biopsy can have a whole-slide tissue fraction near 0.15. A global 0.5
threshold would discard most of it. Set the threshold per specimen type.

### Step 12. Decide materialise or stream

**Input:** Passes per slide, storage budget, latency budget.

**Output:** A pipeline mode: materialised patches, streamed tiles with a cache, or a manifest with
content hashes and on-demand extraction.

**Rationale:** Materialising trades storage for latency and reproducibility. Streaming trades
latency for storage but re-reads the source on every pass. The decision is usually settled by how
many times the same patch is read and whether the source can sustain the request rate.

| Condition | Materialise patches | Stream with cache | Manifest plus on-demand |
|---|---|---|---|
| Three or more passes over the same slide | Yes | No | No |
| One pass, storage constrained | No | Yes | Possible |
| Retraining on a fixed cohort | Yes, with content hashes | No | No |
| Reproduction of a specific clinical result | Yes, plus source hash | No | Yes |
| Object storage with high per-request latency | Yes | No | Possible |

**Edge case:** A patch store regenerated on every run is not a cache; it is a second source of
truth that will diverge from the first.

### Step 13. Extract tiles with bounded concurrency

**Input:** The kept patch list and a reader that can seek to a tile.

**Output:** Batches of pixel arrays delivered to the encoder.

**Rationale:** Throughput is usually limited by per-request latency, not bandwidth, because each
tile is a small read. With 32 requests in flight and 50 ms per request, 4,873 patches take about
7.6 s of waiting alone. Serial reads, or a new connection per tile, multiply that by an order of
magnitude. The reader is also not thread-safe per handle: use one handle per worker or a
thread-safe reader.

**Branch points:**
- Local filesystem: concurrency limited by seek rate; prefetch sequentially.
- Object storage or network file share: limited by request rate, connection pool size, and
  small-read latency, which is often worse on a file share than on object storage.

**Edge case:** Retries without jitter synchronise workers into a herd after an outage. Use
exponential backoff with jitter and a capped retry count.

### Step 14. Decide the stain normalisation policy

**Input:** The patch batch and the model's training-time preprocessing.

**Output:** A normalised batch, or an explicit decision not to normalise, recorded in provenance.

**Rationale:** Stain variation is a site signature. If training and inference normalise
differently, the model sees a distribution it was not trained on. The policy must be one value for
the life of the model.

| Situation | Policy | Reason |
|---|---|---|
| Single scanner, stable protocol | None | Adds variance without removing a known shift |
| Multi-site training | Per-slide Macenko to a fixed reference | Removes the dominant between-site colour shift |
| Single site with seasonal drift | Per-slide to reference, monitored | Stain-matrix change becomes a drift signal |
| Immunohistochemistry or multiplex | Do not apply an H&E method | The two-stain model does not transfer |
| Frozen section | Per-slide, separate reference | Ice and fixation change the colour distribution |

**Edge case:** Macenko estimation on a patch with almost no tissue gives an unstable stain
matrix. Estimate on tissue-rich patches per slide and apply the result to all.

### Step 15. Embed and write shards

**Input:** Normalised patches and an encoder.

**Output:** One row per patch: slide identifier, level, tile column, tile row, level-0 origin,
float16 embedding, and a checksum of the source tile.

**Rationale:** The embedding table is what later stages use. It must join back to pixels, or
attention maps cannot be rendered and errors cannot be investigated. Float16 halves storage with
little effect on linear-probe accuracy for most encoders, but validate that rather than assume.

**Branch points:**
- Shard per slide: simple, many small files, poor for sequential training reads.
- Shard per fixed row count: better for training, needs an index.
- One file per cohort: easiest to move, expensive to update.

**Edge case:** A non-deterministic row order makes two runs of the same pipeline differ. Sort by
slide and patch index before writing.

### Step 16. Register provenance and orchestrate

**Input:** The manifest, the embedding shard, the run parameters, and a queue of slides.

**Output:** A provenance record per artefact and at most one stored result per input hash.

**Rationale:** Queues deliver at least once. A worker that crashes after writing embeddings but
before acknowledging will be retried, and a naive pipeline writes the slide twice. Keying output
by input content hash and writing atomically makes the retry harmless. The provenance record,
holding input hash, reader version, weights hash, normalisation policy, tile size, level, and
timestamp, is what lets a later audit reproduce the result.

**Branch points:**
- New slide: enqueue at normal priority.
- Existing hash, same pipeline version: skip.
- Existing hash, new pipeline version: re-run into a new version prefix; never overwrite.
- Repeated failure: dead-letter queue with a failure class, not infinite retry.

**Edge case:** Slide identifiers reused across years, a real pattern in some laboratories,
collide in any store keyed by identifier alone. Key by accession plus scan timestamp, or by
content hash.

### Step 17. Monitor and gate

**Input:** Per-slide metrics: tiles enumerated, tiles kept, bytes read, read latency at the 50th
and 95th percentile, encoder seconds, embedding rows, artefact report.

**Output:** A pass, flag, or quarantine decision with a reason code, and a metrics stream.

**Rationale:** The pipeline fails in ways the model cannot reveal. A drop in tiles kept per
slide, a rise in bytes read per slide, or a change in the ratio of embedding rows to kept tiles
are early warnings. Without them the first symptom is a model metric moving weeks later.

**Branch points:**
- Metrics inside the historical band: pass.
- One scanner's tiles-kept distribution shifts: suspect mask or scanner change; flag.
- Bytes read per slide approaches the slide size: suspect mask failure or coordinate drift; fail.
- Repeated QC failure on one scanner: route to the laboratory, not to the model team.

**Edge case:** Averages hide failures. Track distributions and the count of slides below a
minimum patch count, not only the mean. A quarantined slide should never vanish silently: the
slides that fail QC are often the fatty, bloody, or poorly fixed ones, and dropping them biases
the cohort.

### Step 18. Manage the lifecycle

**Input:** Stored images, patches, embeddings, manifests.

**Output:** A retention policy with tiers, deletion rules, and a recomputation plan.

**Rationale:** Images dominate storage cost, embeddings are small, and patches sit between. The
best trade is usually to keep the source slide and the embeddings and delete materialised patches
when a model version retires. The retention period is set by accreditation and local law, not by
the data team.

**Branch points:**
- Source must be retained for the diagnostic record period: keep in the archive tier.
- Embeddings are cheap: keep hot for the model lifetime.
- Patches are reproducible from source plus manifest: delete after training, keep the manifest.

**Edge case:** Deleting the source and keeping only patches breaks re-running with a new model.
Keep the source for as long as the record requires.

## 6. Worked example

A regional laboratory scans 600 slides on a working day, 250 days a year, on one scanner model
that writes 40x images at 0.25 um per pixel. The average tissue section is 15.0 mm by 15.0 mm.
The laboratory wants to embed every slide with a vision transformer at 20x and asks three
questions: how much storage per year, how much accelerator time per day, and whether the tile
reader or the encoder is the constraint. Every number below is an assumption with a unit, and
Section 7 recomputes the same numbers in code.

### 6.1 The slide at level 0

Pixels per side:

```text
15.0 mm / 0.25 um/px = 15,000 um / 0.25 um/px = 60,000 px
```

Pixel matrix and raw bytes:

```text
60,000 x 60,000          = 3,600,000,000 px = 3.6 gigapixels
3,600,000,000 px x 3 B   = 10,800,000,000 B = 10.8 GB
```

At a 12:1 compression ratio, typical for JPEG quality 80 on an H&E section, level 0 costs
0.90 GB. The pyramid adds one third:

```text
0.90 GB x (1 + 1/4 + 1/16 + ...) = 0.90 GB x 4/3 = 1.20 GB per slide
```

The pyramid has nine levels: 60,000, then 30,000, 15,000, and so on down to 235 pixels, which is
below one 256-pixel tile. The exact count follows from halving until the long side fits a tile.

Annual storage at 600 slides per day and 250 working days:

```text
600 slides/day x 250 days/year x 1.20 GB/slide = 180,000 GB/year = 180 TB/year
```

That is image data alone. It is also the number that decides whether the archive is on-premises
or in object storage, and it is the number that a vendor quote of "under 2 GB per slide" is
quietly rounding up from.

### 6.2 The analysis level and the patch count

The model expects 0.50 um per pixel, which is 20x. The slide is 0.25 um per pixel, so the correct
level is level 1, exactly half the linear resolution:

```text
level 1 matrix = 60,000 / 2 = 30,000 px per side
level 1 pixels = 30,000 x 30,000 = 900,000,000 px
```

Tiles of 256 pixels at level 1:

```text
tiles per side = ceil(30,000 / 256) = 118
tiles total    = 118 x 118 = 13,924
```

The tissue mask keeps 35 percent of tiles:

```text
kept patches = 13,924 x 0.35 = 4,873 patches (rounded)
```

The 35 percent is measured, not assumed: it is the median tissue fraction across a labelled
sample of 200 slides from this laboratory. A pipeline that assumes 80 percent will over-provision
compute by more than a factor of two; one that assumes 10 percent will under-provision.

Bytes read per slide, given that a level-1 tile is about the same compressed size as a level-0
tile (16,297 B in this configuration):

```text
4,873 patches x 16,297 B = 79,400,000 B = 79.4 MB per slide
79.4 MB / 1,200 MB = 6.6 percent of the slide
```

The pipeline reads 6.6 percent of the slide's bytes because it skips glass. This is the central
economy of slide-scale processing: the mask is what makes the workload tractable, and a broken
mask is what makes it expensive.

### 6.3 Throughput: two clocks

Assume the encoder sustains 450 patches per second per accelerator, measured on 256-pixel inputs
with batch size 64. Compute time per slide:

```text
4,873 patches / 450 patches/s = 10.8 s
```

Assume tile reads go to object storage with 50 ms per request and 32 requests in flight:

```text
4,873 / 32 = 152.3 batches
152.3 x 0.050 s = 7.6 s of pure waiting per slide
```

If the stages run serially, the slide takes 18.4 s, and 600 slides take:

```text
600 x 18.4 s = 11,040 s = 3.07 h
```

With eight workers, each handling one slide at a time:

```text
3.07 h / 8 = 0.38 h = 23 min
```

Accelerator demand, counted properly in GPU-hours rather than in wall clock:

```text
600 slides x 10.8 s = 6,480 s = 1.80 GPU-hours per day
```

Two conclusions follow, and both matter for the design review. First, the daily embedding job
needs less than two accelerator-hours, so a single eight-GPU node is idle most of the day and
should be shared with training or with additional models. Second, the serial wall clock is
1.7 times the compute time, so a pipeline that does not overlap reads with compute wastes about
40 percent of its wall clock even though the accelerator is not the constraint.

### 6.4 Embeddings and the rest of the storage

The embedding for one patch at 1,024 dimensions in float16, plus 64 bytes of key and checksum:

```text
1,024 x 2 B + 64 B = 2,112 B per patch
4,873 x 2,112 B    = 10,300,000 B = 10.3 MB per slide
```

Annual embeddings:

```text
600 x 250 x 10.3 MB = 1,545,000 MB = 1.5 TB per year
```

So the split is 180 TB of images against 1.5 TB of embeddings per year, a ratio of about 117 to
1. That ratio is the argument for keeping embeddings hot, keeping images in a cheaper tier, and
never storing extracted patches as a third copy unless a specific workflow needs them.

### 6.5 Two diagnostics that reveal a broken pipeline

The pipeline above is healthy on average. Two measurements catch the failures that averages hide.

**Diagnostic 1: the distribution of bytes read per slide.** The median is 79.4 MB. On a healthy
fleet the 95th percentile is maybe 110 MB, from larger sections. Suppose the 95th percentile is
204 MB instead. Solving backwards, 204 MB / 16,297 B = 12,518 patches, which is 90 percent of the
13,924 tiles at level 1. A mask that keeps 90 percent of tiles is not segmenting tissue; it is
passing glass. The likely cause is a threshold tuned on scanner A applied to scanner B, whose
glass background is darker. The fix is per-scanner thresholds validated against a labelled
sample, not a global constant. The diagnostic works because it needs no model and no labels:
bytes read per slide is a pipeline metric that moves before any accuracy metric does.

**Diagnostic 2: rows per kept tile.** The embedding table should have exactly one row per kept
patch, so the ratio of rows to kept tiles should be 1.00 with a small rounding tail. Suppose
3 percent of slides show 2.00. That is not a rounding artefact; it is a retry that wrote the same
rows twice. The mechanism is a worker that wrote its shard, failed before acknowledging the
queue message, and was retried. Duplicate rows do two kinds of damage. In training they weight
the affected slides twice. In evaluation, if the duplicate carries a different split assignment,
the same patient's tissue appears on both sides of the split. The fix is a primary key of
(slide hash, level, tile column, tile row) with an upsert, or an atomic write to a
content-addressed path. The diagnostic is a single grouped count, and it should run after every
batch.

A third check belongs in the same place: the count of slides with fewer than 200 kept patches.
A slide with 4,873 patches that suddenly produces 40 is either a tiny biopsy or a masking
failure, and those two cases need different responses.

### 6.6 A varied assumption: analysis at 40x

Suppose the model is retrained at 0.25 um per pixel, so the pipeline reads level 0 instead of
level 1. The patch count changes by the square of the resolution ratio:

```text
19,329 patches, which is 4.0 times the 4,873 of the 20x case
```

The ratio is not exactly four because each tile grid rounds up; four times 4,873 is 19,492.

Everything downstream scales linearly with patch count, except storage, which scales with the
same factor:

| Quantity | At 20x (0.50 um/px) | At 40x (0.25 um/px) | Ratio |
|---|---:|---:|---:|
| Patches per slide | 4,873 | 19,329 | 4.0 |
| Compute per slide | 10.8 s | 43.0 s | 4.0 |
| Accelerator demand per day | 1.80 GPU-hours | 7.16 GPU-hours | 4.0 |
| Bytes read per slide | 79.4 MB | 315 MB | 4.0 |
| Embeddings per slide | 10.3 MB | 40.8 MB | 4.0 |
| Annual embeddings | 1.5 TB | 6.1 TB | 4.0 |

The conclusion changes in a specific way. At 20x, one eight-GPU node is over-provisioned and the
job is a nightly batch. At 40x, the daily demand is 7.16 GPU-hours, which still fits a single
node but no longer leaves it idle, and the read volume of 315 MB per slide means the storage
tier must sustain about 190 GB per day for the embedding job alone. The choice between the two
resolutions is therefore not a modelling preference that the infrastructure can absorb; it is a
capacity decision that must be made before the cohort is processed. The same arithmetic applies
to a tissue-fraction assumption: at a 0.15 fraction, patches fall to 2,089 and compute to 4.6 s,
so a capacity plan built on the 0.35 median over-provisions by 2.3 times for a biopsy-heavy
service.

## 7. Code, query, or calculation

The script below reproduces every number in Section 6 using only the Python standard library. It
is deliberately a model, not a reader: it does not open a slide. Its value is that the capacity
arithmetic is explicit, testable, and runnable in a review meeting without a GPU or a slide file.

Assumptions are constants at the top, so a reviewer can change one and see the effect. The four
that matter most are `RATIO` (compression), `TISSUE` (the share of tiles that pass the mask),
`PPS` (encoder throughput per accelerator), and `MPP_TARGET` (the analysis resolution).

```python
"""wsi_capacity.py -- storage, patch, and throughput arithmetic for a WSI pipeline.

Standard library only. Deterministic: every random draw is seeded.
Run: python wsi_capacity.py
"""

import math
import random
from collections import OrderedDict

AREA_MM = 15.0          # tissue extent, millimetres
MPP0 = 0.25             # level-0 microns per pixel (40x)
MPP_TARGET = 0.50       # analysis microns per pixel (20x)
TILE = 256              # tile edge, pixels
RATIO = 12.0            # raw RGB bytes / stored bytes, JPEG q80 H&E
TISSUE = 0.35           # measured share of tiles passing the mask
SLIDES_DAY = 600
PPS = 450.0             # patches per second per accelerator
LATENCY = 0.050         # seconds per tile request
INFLIGHT = 32           # concurrent requests
EMB_DIM = 1024


def levels(w, h, tile=TILE):
    """Power-of-two pyramid as (index, width_px, height_px, tiles_x, tiles_y)."""
    out = []
    i = 0
    while True:
        cw, ch = math.ceil(w / 2 ** i), math.ceil(h / 2 ** i)
        out.append((i, cw, ch, math.ceil(cw / tile), math.ceil(ch / tile)))
        if max(cw, ch) <= tile:
            return out
        i += 1


def pick_level(pyr, target):
    """Finest level whose pixel size is at least the target (never upsample)."""
    ok = [lv for lv in pyr if MPP0 * 2 ** lv[0] >= target - 1e-9]
    return min(ok, key=lambda lv: MPP0 * 2 ** lv[0]) if ok else pyr[-1]


class LRU:
    def __init__(self, capacity):
        self.capacity, self.store, self.hits, self.misses = capacity, OrderedDict(), 0, 0

    def get(self, key):
        if key in self.store:
            self.store.move_to_end(key)
            self.hits += 1
        else:
            self.misses += 1
            self.store[key] = True
            if len(self.store) > self.capacity:
                self.store.popitem(last=False)


def main():
    w = h = round(AREA_MM * 1000 / MPP0)
    pyr = levels(w, h)
    raw = w * h * 3
    stored0 = raw / RATIO
    stored = stored0 * 4 / 3

    print("== Slide geometry ==")
    print(f"level-0 matrix     : {w:,} x {h:,} px = {w*h/1e9:.1f} gigapixels")
    print(f"pyramid levels     : {len(pyr)}")
    for lv in pyr[:3]:
        print(f"  L{lv[0]} {lv[1]:>6,} x {lv[2]:>6,} px  mpp={MPP0*2**lv[0]:.2f}  tiles={lv[3]*lv[4]:>7,}")
    print(f"raw level-0 RGB    : {raw/1e9:.2f} GB")
    print(f"compressed L0      : {stored0/1e9:.2f} GB at {RATIO:.0f}:1")
    print(f"plus pyramid       : {stored/1e9:.2f} GB per slide")

    lv = pick_level(pyr, MPP_TARGET)
    tiles = lv[3] * lv[4]
    patches = round(tiles * TISSUE)
    tile_bytes = stored0 / (pyr[0][3] * pyr[0][4])
    read = patches * tile_bytes

    print("\n== Analysis level ==")
    print(f"target             : {MPP_TARGET:.2f} um/px -> level {lv[0]} at {MPP0*2**lv[0]:.2f}")
    print(f"tiles at level     : {tiles:,}")
    print(f"kept at {TISSUE:.0%} tissue : {patches:,}")
    print(f"bytes per tile     : {tile_bytes:,.0f} B")
    print(f"read per slide     : {read/1e6:.1f} MB ({100*read/stored:.1f}% of the slide)")

    compute = patches / PPS
    wait = patches / INFLIGHT * LATENCY
    print("\n== Throughput ==")
    print(f"compute per slide  : {compute:.1f} s at {PPS:.0f} patches/s")
    print(f"request wait       : {wait:.1f} s ({INFLIGHT} in flight, {LATENCY*1000:.0f} ms)")
    print(f"serial per slide   : {compute+wait:.1f} s")
    print(f"daily GPU demand   : {SLIDES_DAY*compute/3600:.2f} GPU-hours")
    print(f"one worker         : {SLIDES_DAY*(compute+wait)/3600:.2f} h")
    print(f"eight workers      : {SLIDES_DAY*(compute+wait)/3600/8*60:.0f} min")

    emb = patches * (EMB_DIM * 2 + 64)
    print("\n== Storage ==")
    print(f"embeddings/slide   : {emb/1e6:.1f} MB")
    print(f"images per year    : {SLIDES_DAY*250*stored/1e12:.0f} TB")
    print(f"embeddings per year: {SLIDES_DAY*250*emb/1e12:.2f} TB")
    print(f"ratio              : {SLIDES_DAY*250*stored/(SLIDES_DAY*250*emb):.0f}:1")

    print("\n== Tile cache, two passes over one slide ==")
    order = list(range(patches))
    random.Random(20261027).shuffle(order)
    for cap in (1000, 2500, 5000, 10000):
        c = LRU(cap)
        for _ in range(2):
            for t in order:
                c.get((0, t))
        total = c.hits + c.misses
        print(f"capacity {cap:>6,} tiles: hit rate {100*c.hits/total:5.1f}%")

    print("\n== Varied assumption: analysis at 40x ==")
    p40 = round(pyr[0][3] * pyr[0][4] * TISSUE)
    print(f"patches            : {p40:,} (x{p40/patches:.1f})")
    print(f"compute per slide  : {p40/PPS:.1f} s")
    print(f"daily GPU demand   : {SLIDES_DAY*p40/PPS/3600:.2f} GPU-hours")
    print(f"read per slide     : {p40*tile_bytes/1e6:.0f} MB")
    print(f"embeddings per year: {SLIDES_DAY*250*p40*(EMB_DIM*2+64)/1e12:.1f} TB")


if __name__ == "__main__":
    main()
```

Output on the author's machine, Python 3.11, unchanged assumptions:

```text
== Slide geometry ==
level-0 matrix     : 60,000 x 60,000 px = 3.6 gigapixels
pyramid levels     : 9
  L0 60,000 x 60,000 px  mpp=0.25  tiles= 55,225
  L1 30,000 x 30,000 px  mpp=0.50  tiles= 13,924
  L2 15,000 x 15,000 px  mpp=1.00  tiles=  3,481
raw level-0 RGB    : 10.80 GB
compressed L0      : 0.90 GB at 12:1
plus pyramid       : 1.20 GB per slide

== Analysis level ==
target             : 0.50 um/px -> level 1 at 0.50
tiles at level     : 13,924
kept at 35% tissue : 4,873
bytes per tile     : 16,297 B
read per slide     : 79.4 MB (6.6% of the slide)

== Throughput ==
compute per slide  : 10.8 s at 450 patches/s
request wait       : 7.6 s (32 in flight, 50 ms)
serial per slide   : 18.4 s
daily GPU demand   : 1.80 GPU-hours
one worker         : 3.07 h
eight workers      : 23 min

== Storage ==
embeddings/slide   : 10.3 MB
images per year    : 180 TB
embeddings per year: 1.54 TB
ratio              : 117:1

== Tile cache, two passes over one slide ==
capacity  1,000 tiles: hit rate   0.0%
capacity  2,500 tiles: hit rate   0.0%
capacity  5,000 tiles: hit rate  50.0%
capacity 10,000 tiles: hit rate  50.0%

== Varied assumption: analysis at 40x ==
patches            : 19,329 (x4.0)
compute per slide  : 43.0 s
daily GPU demand   : 7.16 GPU-hours
read per slide     : 315 MB
embeddings per year: 6.1 TB
```

### 7.1 What the output establishes

Three results are worth reading carefully. The cache hit rate is zero for capacities below the
working set and fifty percent above it: a cache that holds a thousand tiles buys nothing when a
slide needs 4,873, and a cache that holds the working set converts the second pass into a hit.
This is why cache sizing is a capacity decision, not a tuning parameter.

The eight-worker wall clock of 23 minutes is not eight times faster than the serial estimate of
3.07 hours, because the per-slide time includes a serial read stage. Overlapping the read and
compute stages, for example by prefetching the next batch while the encoder runs, is where the
real speed-up lives.

The 40x variant quadruples every linear quantity. The conclusion "one node is enough" survives,
but the margin does not, and the annual embedding store grows from 1.5 TB to 6.1 TB.

### 7.2 What the model does not capture

The script models tiles as independent and equally sized. Real compressed tiles vary with tissue
content, so the bytes-read estimate is a mean, not a bound. It ignores decompression cost, which
for JPEG 2000 can rival the encoder time on some hardware. It ignores the tail: a single slow
request does not delay a slide if other requests can proceed, but a storage outage does. And it
assumes the mask is correct, which is the assumption the diagnostics in Section 6 are designed to
test.

## 8. Data, standards, and regulatory context

A slide pipeline sits at the junction of an imaging standard, a laboratory accreditation regime,
and a privacy regime. Each constrains a different part of the design, and confusing a convention
with a requirement is a common source of over-engineering or, worse, of a compliance gap.

### 8.1 DICOM Whole Slide Imaging

DICOM Supplement 145, *Whole Slide Microscopic Image IOD and SOP Classes*, reached final text in
2010. It defines the VL Whole Slide Microscopy Image information object: a multi-frame image in
which each frame is a tile, organised into pyramid levels, with the physical geometry of the
slide expressed in millimetres.

Four attributes do most of the work in a pipeline:

- **Total Pixel Matrix Columns and Rows** give the full extent of a level, including empty glass.
  A pipeline that uses the imaged volume instead will over- or under-estimate the tile grid.
- **Dimension Organization Type** is `TILED_FULL` or `TILED_SPARSE`. Under `TILED_FULL`, frames are
  stored in a defined order (row direction, then column direction, then depth, then optical path,
  then segment) and per-frame position macros may be omitted. Under `TILED_SPARSE`, every frame
  carries its own position and a reader may assume nothing about ordering. A reader that only
  handles `TILED_FULL` will scramble a `TILED_SPARSE` slide rather than fail loudly.
- **Optical Path Sequence** distinguishes brightfield from fluorescence channels and identifies
  which frames belong to which path. Multiplexed slides have several paths over the same
  coordinates.
- **Plane Position (Slide)** gives each frame's position in the slide coordinate system, which is
  what allows two serial sections to be registered to each other.

DICOM also defines the associated images: the label, the macro overview, and the thumbnail. The
label image is a photograph of text and is therefore the highest privacy risk in the file.

### 8.2 Transport and enterprise integration

DICOMweb defines RESTful access. WADO-RS retrieves frames and metadata over HTTP, and a pipeline
can request a single frame rather than a whole series. STOW-RS stores instances. In practice,
vendor neutral archives (VNAs) vary in how completely they implement WSI-specific DICOMweb
resources, so a pipeline that depends on frame-level retrieval should be tested against the
archive it will actually use, not against a reference implementation.

HL7 version 2 and Fast Healthcare Interoperability Resources (FHIR) carry the order and result
context, not the pixels. The relevant FHIR resources are `ImagingStudy` for the series,
`Specimen` for the tissue, and `DiagnosticReport` for the result. The laboratory information
system remains the source of truth for accession numbers and specimen identity; the image archive
is the source of truth for pixels. A pipeline that invents its own identifier instead of joining
these two will duplicate cases.

### 8.3 Accreditation and validation

The College of American Pathologists (CAP), with the American Society for Clinical Pathology and
the Association for Pathology Informatics, published an updated guideline for validating whole
slide imaging for diagnostic purposes in 2021 (Evans et al., *Archives of Pathology and Laboratory
Medicine*, 2022). Its three strong recommendations are: at least 60 cases for one application,
with 20 more per additional application; intraobserver concordance of at least 95 percent between
digital and glass; and a washout of at least two weeks between the two readings. These apply to
diagnostic use of the scanner, not to a research pipeline, but they set the evidentiary bar that
any clinical deployment inherits.

### 8.4 Device regulation

The United States Food and Drug Administration (FDA) published technical performance guidance for
digital pathology whole slide imaging devices in 2016, covering colour reproducibility, focus,
tissue coverage, and quality control with quantitative action limits. Software that analyses
slides may be a device in its own right, depending on its intended use and claims.

The regulatory status of laboratory developed tests (LDTs) changed twice in two years and any
2026 document that repeats the 2024 position is stale. The FDA's May 2024 final rule (89 FR 37286)
would have phased out general enforcement discretion for LDTs. In March 2025 the United States
District Court for the Eastern District of Texas vacated the rule in its entirety in *American
Clinical Laboratory Association v. FDA*. The FDA then published a conforming rule in September
2025 (90 FR 45134) removing the added language from 21 CFR 809.3(a). As of October 2026 the
pre-2024 enforcement position is restored, and the question of LDT oversight remains with the
Department of Health and Human Services and, potentially, Congress. Treat this paragraph as a
dated summary, not as legal advice.

### 8.5 Privacy

Slide files contain direct identifiers in three places: the label image, the vendor metadata
(accession and sometimes patient name), and the filename. Pixel data itself can also carry
burned-in annotations. De-identification under the HIPAA Privacy Rule or the GDPR requires
removing all three, and the label image is the one most often missed because it is not part of the
pyramid. DICOM PS3.15 Annex E defines the attribute-level de-identification profile; it does not
remove text burned into pixels.

| Instrument | What it constrains | Binding or convention |
|---|---|---|
| DICOM PS3.3 Supplement 145 | Frame organisation, geometry, optical paths | Binding for DICOM WSI; optional for native formats |
| DICOMweb WADO-RS | Frame retrieval over HTTP | Binding where the archive exposes DICOMweb |
| HL7 v2 ORM/ORU | Order and result messages | Binding at the interface, version-negotiated |
| FHIR R4 ImagingStudy, Specimen | Resource identity and linkage | Convention until the local interface specifies it |
| CAP/CLIA WSI validation | 60 cases, 95 percent concordance, 2-week washout | Binding for diagnostic use in accredited laboratories |
| FDA WSI technical guidance (2016) | Colour, focus, QC action limits | Guidance for device submissions |
| HIPAA Privacy Rule, GDPR | Identifier removal | Binding by jurisdiction |
| DICOM PS3.15 Annex E | Attribute-level de-identification | Binding when the DICOM profile is claimed |

What these standards do not constrain is as important. None of them specifies a compression
ratio, a tile size, a tissue-mask threshold, an analysis magnification, or an embedding format.
Those are engineering choices, and they are where the failures in Section 10 live.

## 9. Industry practice and case evidence

Slide-scale infrastructure is built from a small number of components that most laboratories
assemble in similar ways: a scanner, a vendor or DICOM archive, a viewer, a compute layer, and a
model. What differs is where the conversion happens and how much of the pyramid is retained.

### 9.1 Scanners and throughput

Published manufacturer figures for sustained throughput are a useful starting point, provided
they are read with their scan area and magnification attached.

| Scanner | Published throughput | Conditions | Source type |
|---|---:|---|---|
| Leica Aperio GT 450 | 81 slides per hour | 15 mm by 15 mm at 40x, 0.26 um/px | Manufacturer specification |
| Hamamatsu NanoZoomer S360 | 82 slides per hour | 15 mm by 15 mm, 20x or 40x mode | Manufacturer brochure |
| Philips IntelliSite Ultra Fast | 60 s per slide | 15 mm by 15 mm at 40x equivalent | Manufacturer specification |

These figures are upper bounds under ideal slide handling. A vendor-published workflow study at
NeoGenomics reported about 41 slides per hour on a GT 450 at 40x in routine operation, against 25
slides per hour on an older Aperio AT2 at 20x. The gap between 81 and 41 is the difference between
a specification and a laboratory, and it is the number that capacity planning should use.

### 9.2 Formats in practice

Documented practice, as distinct from vendor marketing:

- Most scanners still write a native format by default: SVS for Leica Aperio, NDPI for Hamamatsu,
  MRXS for 3DHISTECH. These remain the working format for research cohorts.
- DICOM WSI is the archive format where pathology has been consolidated into a VNA alongside
  radiology. Adoption is uneven across vendors and archives.
- OpenSlide is the reference reader for native formats and the basis of most open-source tooling;
  tiffslide is a pure-Python alternative for pyramidal TIFF, and cuCIM adds GPU-accelerated reads.
- Conversion between formats is common at ingestion, and it is where metadata is most often lost.

An inference, not a documented fact: the format boundary is the highest-risk point in the
pipeline, because the coordinate system, the microns per pixel, and the compression all change at
once. Validate metadata before and after conversion, not only at the source.

### 9.3 Serving and storage patterns

Three patterns dominate. A **materialised patch store** writes 256-pixel tiles to a shard format
and trains from it; it is fast and reproducible and costs several times the storage of the source
slide. A **streaming reader** opens the slide at training time and reads tiles through a cache; it
saves storage and makes every epoch dependent on the archive's latency. A **hybrid** keeps
embeddings, which are small, and streams pixels only when a new model requires re-embedding.

The arithmetic in Section 6 explains the popularity of the hybrid: embeddings are about 1 percent
of image bytes, so they can be kept hot, while patches are a large multiple of the source.

### 9.4 What the evidence does not show

There is little public evidence on pipeline reliability at scale. Vendors publish throughput and
format support, laboratories publish validation concordance, and almost nobody publishes
tile-level defect or mask-failure rates. Expect to generate your own operational baseline in the
first month, and treat "it just works across scanners" as unverified until shown on your slides.

## 10. Failure modes and red flags

The table below lists failures that survive code review because each one produces a plausible
output. The early warning column is the metric or artefact that would catch it before a model
metric moves.

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Level mismatch | Patch scale looks wrong when a tile is rendered | Analysis level chosen by magnification label rather than microns per pixel | Model sees objects at the wrong scale; accuracy falls gradually | Store microns per pixel; assert level at load |
| Coordinate origin drift | Attention maps land beside the annotated region | Level-0 origin computed as tile index times tile size at the analysis level | Overlays and patch labels misalign; supervision is wrong | One tested coordinate function; overlay check per batch |
| Mask includes glass | Bytes read per slide near slide size | Threshold tuned on one scanner's background | Compute cost rises several-fold; embeddings are mostly background | Per-scanner thresholds; tissue-fraction histogram |
| Mask excludes pale tissue | Tissue fraction near zero on fatty slides | Saturation-only threshold | Whole tissue types dropped; subgroup failure | Validate mask by tissue type |
| Ink treated as tissue | High patch counts at resection margins | Pen is saturated and dark | Model learns margin ink as a cancer cue | Artefact mask with a per-site labelled sample |
| Duplicate rows after retry | Rows per kept tile above 1.0 | At-least-once queue with non-idempotent write | Inflated training weight; patient leakage across splits | Primary key on slide hash and tile; atomic write |
| Truncated file accepted | Level inventory shorter than usual | Partial scan that still opens | Missing region, silent | Validate extent against metadata |
| Reader not thread-safe | Intermittent corrupt tiles under load | Shared OpenSlide handle across threads | Random pixel corruption | One handle per worker |
| Cache below working set | Zero cache hits in the metrics | Cache sized by memory rather than working set | Second pass re-reads everything | Size to the per-slide working set |
| Label image leaks PHI | Identifiers in logs or features | Label read as an ordinary image | Privacy breach | Exclude label from features; redact metadata |
| Compression mismatch | Model trained on converted slides, served native | Two conversion paths | Subtle pixel shift; performance drift | One conversion path, recorded in provenance |
| Unpinned normalisation | Stain policy differs between train and serve | Preprocessing implemented twice | Colour distribution shift at inference | Single shared transform, versioned |
| Silent slide drop | Cohort counts do not reconcile | QC filter excludes without recording | Non-random missingness | Quarantine with reason codes |
| Identifier collision | Two patients' slides share an accession | Identifier reused across years | Mixed records | Key by accession plus timestamp or content hash |

Three of these deserve emphasis because they are invisible in aggregate. Duplicate rows raise a
training-set size by a few percent, which looks like a data-cleaning detail. Coordinate drift
produces attention maps that are plausible but offset, which passes a visual check at low zoom.
Silent drops reduce the denominator, which makes a bad model look stable.

## 11. Assumptions and limits

The arithmetic in this topic rests on assumptions that are reasonable for a mid-sized laboratory
and wrong for others. Each one is testable, and the test is cheaper than the consequence of
assuming it.

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Tissue extent of 15 mm by 15 mm | Converts physical area to pixels | Biopsies are millimetres; resections can exceed 25 mm | Measure the tissue bounding box distribution |
| 0.25 um per pixel at level 0 | Fixes the pixel matrix | 20x scanners and non-square pixels | Read the metadata, not the label |
| 12:1 compression | Predicts storage | Scanner settings, tissue density, JPEG 2000 | Sample 100 slides and compute bytes per pixel |
| 35 percent of tiles are tissue | Predicts patch count and compute | Specimen mix, mask threshold | Histogram the kept fraction per slide |
| Encoding at 450 patches/s | Predicts accelerator demand | Model size, batch size, precision | Benchmark the actual model on the actual hardware |
| 50 ms per tile request | Predicts read latency | Local disk, CDN, archive tier | Measure p50 and p95 from the real store |
| 32 requests in flight | Predicts concurrency benefit | Rate limits, connection pool | Load test until throughput plateaus |
| Tiles are independent and equally sized | Simplifies the model | Tissue-dependent compression; overlapping patches | Compare predicted to observed bytes read |
| Embeddings at float16 lose little accuracy | Justifies storage | Some encoders and some tasks | Ablate float32 against float16 per task |
| 250 working days | Annualises storage | 24/7 laboratories, seasonal demand | Use the laboratory's own calendar |
| One slide, one patient | Simplifies joins | Multiple blocks and recuts per case | Join on accession and block, not on file |
| Retry is harmless | Justifies at-least-once queues | Non-idempotent writes | Count rows per key after an induced failure |

The largest limit is that this topic treats the pipeline as a batch process. A clinical serving
pipeline adds latency budgets, per-request authentication, audit logging, and availability
requirements that change the architecture. The arithmetic still applies, but the conclusion about
which clock dominates often reverses: a single slide requested by a pathologist must start
returning tiles in under a second, and no amount of batch throughput fixes that.

## 12. Exercises

The exercises move from arithmetic to design. Each states a deliverable, the criteria a reviewer
would apply, and a realistic time. Work them in order; exercises 6 and 8 depend on decisions made
in 3 and 5.

### Exercise 1. Pyramid arithmetic for a second scanner

A scanner writes 0.50 um per pixel. A section is 20.0 mm by 15.0 mm and the mask keeps 30 percent
of tiles. Compute the pixel matrix, the compressed size at 12:1 with pyramid overhead, the tile
count at 256 pixels, and the patch count.

**Deliverable:** A one-page calculation with every unit shown.

**Acceptance criteria:** Pixel matrix 40,000 by 30,000; raw 3.6 GB; stored about 0.40 GB; 18,526
tiles; about 5,558 patches. A reviewer can follow each step without asking a question.

**Time:** 30 minutes.

### Exercise 2. Five-year storage budget

A laboratory scans 900 slides per day, 250 days a year, at 1.2 GB per slide, and must retain
images for five years. Hot storage costs 0.023 USD per GB-month, warm 0.0125, and archive 0.004.
Slides are read heavily for 30 days, occasionally for a year, and rarely after.

**Deliverable:** A tiering table with the volume in each tier per year and the five-year cost.

**Acceptance criteria:** Total volume 1.35 PB; the tier boundaries are stated as policies, not as
guesses; the calculation shows that the archive tier dominates the total cost even though it
holds the oldest data.

**Time:** 45 minutes.

### Exercise 3. Mask validation across three scanners

Three vendors serve one laboratory. Design a validation plan for the tissue mask.

**Deliverable:** A sampling plan, a threshold table per scanner, and a pass criterion.

**Acceptance criteria:** At least 30 slides per scanner, stratified by specimen type including a
fatty and a bloody case; the pass criterion is a maximum absolute difference in kept-tissue area
against a hand-drawn mask; the plan states what happens when a scanner fails.

**Time:** 60 minutes.

### Exercise 4. Diagnose the bytes-read spike

A pipeline reports median bytes read of 79 MB per slide and a 95th percentile of 204 MB. Tiles per
slide at the analysis level are 13,924, and a tile is 16,297 bytes. Explain the spike, name the
most likely cause, and state the first check.

**Deliverable:** A short diagnosis with the arithmetic.

**Acceptance criteria:** 204 MB corresponds to 12,518 patches, about 90 percent of tiles; the
diagnosis identifies mask failure rather than storage; the first check is tissue fraction by
scanner.

**Time:** 30 minutes.

### Exercise 5. Size the tile cache

A workflow makes two passes over each slide: one for embedding and one for a quality model. A
slide has 4,873 kept tiles. Measure, by reasoning from the cache model in Section 7, what cache
capacity is needed for the second pass to hit, and what happens below that capacity.

**Deliverable:** A capacity recommendation with a justification and a cost estimate in memory.

**Acceptance criteria:** The recommendation is at or above the per-slide working set; the answer
explains the step from zero to fifty percent; memory is computed from tile size and bytes/pixel.

**Time:** 30 minutes.

### Exercise 6. Write the coordinate contract

Define the record that identifies one patch, covering level, tile column and row, level-0 origin,
and the slide content hash. Write a unit test for the conversion between them.

**Deliverable:** A short specification and a runnable test.

**Acceptance criteria:** The test checks level 1, tile (3, 4), tile size 256, which must map to
level-0 origin (1,536, 2,048); the record cannot represent a patch without a level; the test fails
if the conversion omits the downsample factor.

**Time:** 45 minutes.

### Exercise 7. Critique a vendor claim

A vendor states: "Our scanner produces 2 GB slides, fully DICOM compatible, readable in any
viewer." Write a critique of each clause.

**Deliverable:** A three-paragraph critique with the questions a buyer should ask.

**Acceptance criteria:** Identifies that 2 GB is an observation, not a guarantee, and depends on
tissue area and compression; distinguishes DICOM conformance from support for `TILED_SPARSE` and
frame-level retrieval; notes that viewer support must be demonstrated on the buyer's own slides.

**Time:** 30 minutes.

### Exercise 8. Design an idempotent orchestration contract

A queue delivers at least once. Workers write embedding shards to object storage. Design the
contract that makes a retry harmless, and describe a failure-injection test that proves it.

**Deliverable:** A state machine and a test plan.

**Acceptance criteria:** Output keyed by input content hash and pipeline version; write is atomic
or upsert; the test kills a worker between write and acknowledgement and asserts exactly one row
per patch after recovery.

**Time:** 60 minutes.

## 13. Self-check answers

**Exercise 1.** Twenty millimetres at 0.50 um per pixel is 20,000 um / 0.50 um/px = 40,000 px;
fifteen millimetres is 30,000 px. The matrix is 1.2 gigapixels and the raw RGB size is
1.2e9 x 3 = 3.6 GB. At 12:1 the level-0 store is 0.30 GB and with the pyramid 0.40 GB. Tiles are
ceil(40,000/256) = 157 by ceil(30,000/256) = 118, so 18,526 tiles, and 30 percent is 5,558
patches. This is a 20x case of a larger area, which is why the count is close to Section 6's 40x
case rather than four times smaller.

**Exercise 2.** Annual volume is 900 x 250 x 1.2 GB = 270 TB, and five years is 1.35 PB. Thirty days
hot is 900 x 30 x 1.2 GB = 32.4 TB, about 2.4 percent. The next eleven months, about 247 TB, sit
warm, and the remaining 1.07 PB sits in archive. Archive dominates total cost because it holds
79 percent of the bytes. Compression and retention rules matter more than tier pricing.

**Exercise 4.** 204 MB / 16,297 B = 12,518 patches. Against 13,924 tiles, that is 89.9 percent,
not a plausible tissue fraction for a resection and far above the 35 percent median. The mask is
passing glass, most likely because a threshold was transferred from a scanner with a different
background level. First check the tissue-fraction histogram by scanner, then render the mask over
five affected slides.

**Exercise 5.** The working set is 4,873 tiles. A cache of 5,000 holds the slide, so the second
pass hits on all 4,873 requests and the overall rate is 50 percent. A cache of 2,500 holds half
the working set, but because the order is shuffled the second pass finds its tiles evicted and
the rate is 0 percent. Memory is 5,000 x 256 x 256 x 3 B = 983 MB uncompressed, or about 80 MB if
tiles are cached compressed.

**Exercise 6.** The conversion is `level0 = level_pixel * downsample` and
`level_pixel = tile_index * tile_size`, so level 1 with downsample 2 and tile (3, 4) gives
(3 x 256 x 2, 4 x 256 x 2) = (1,536, 2,048). A test that removes the downsample factor produces
(768, 1,024) and must fail. A round-trip test from level-0 origin back to tile indices catches the
inverse error.

**Exercises 3, 7, and 8 are design tasks** and are graded against the acceptance criteria. A
strong Exercise 3 plan reports mask behaviour on fatty and bloody tissue separately, where a
saturation threshold fails. A strong Exercise 7 critique refuses "fully DICOM compatible" without
a conformance statement and a read of the buyer's own `TILED_SPARSE` slides. A strong Exercise 8
answer proves idempotency by killing a worker, not by reasoning about the queue.

## 14. Interview questions

**1. Walk me through what happens to a 1.5 GB slide between the scanner and an embedding table.**
A strong answer names the container, the pyramid, the level selection, the mask, the tile read,
the encoder, and the shard write, with the approximate reduction at each stage. A weak answer
starts at "we resize the image" and never mentions coordinates.

**2. How do you choose the analysis magnification?**
Strong: from the model's training scale in microns per pixel, matched to a pyramid level, never
by upsampling, and recorded in provenance. Weak: "20x is standard".

**3. A model's AUROC drops four points after a scanner change. Where do you look first?**
Strong: pipeline metrics before model metrics, specifically tiles kept per slide and bytes read
per slide, then the mask, then stain, then the model. Weak: retrain on the new scanner
immediately.

**4. How do you size a tile cache?**
Strong: from the per-slide working set and the number of passes, with the observation that a cache
below the working set can hit zero. Weak: "as much as fits in RAM".

**5. Why is at-least-once delivery a problem for this pipeline?**
Strong: because writes are not idempotent by default, so a retry duplicates rows and can leak a
patient across splits; the fix is a key and an atomic write. Weak: "the queue handles it".

**6. What does `TILED_SPARSE` change for a reader?**
Strong: frames carry their own position and optical path, so the reader must use per-frame
functional groups and cannot assume a row-major order. Weak: "it is a DICOM detail".

**7. How would you store five years of slides on a fixed budget?**
Strong: separates images, patches, and embeddings by size and re-use, tiers by access pattern, and
notes that retention is set by accreditation. Weak: "put everything in the cloud".

**8. How do you stop patient identifiers leaking from a slide?**
Strong: names the label image, the vendor metadata, the filename, and burned-in annotation, and
proposes removal plus verification. Weak: "we de-identify the DICOM".

**9. When would you materialise patches rather than stream?**
Strong: when the same patch is read three or more times, or when a specific result must be
reproduced byte for byte, and with the storage multiple stated. Weak: "always, it is faster".

**10. Which single metric would you put on a dashboard first?**
Strong: the distribution of kept tiles per slide, because it moves before accuracy and needs no
labels. Weak: GPU utilisation.

## 15. Key terms

- **WSI** — whole-slide image; a pyramidal tiled representation of an entire glass slide.
- **MPP** — microns per pixel; the physical pixel size that makes scales comparable.
- **Pyramid level** — a downsampled representation, with level 0 at full resolution.
- **Tile** — a fixed-size compressed square, the unit of random access.
- **Patch** — a tile selected for analysis after masking.
- **Tissue mask** — a binary mask separating tissue from glass.
- **Otsu threshold** — a histogram-based threshold maximising between-class variance.
- **Optical density** — `-log10(I/I0)`; absorbance, linear in stain concentration.
- **Stain normalisation** — mapping a slide's colour toward a reference.
- **TILED_FULL** — DICOM organisation with defined frame order and omitted per-frame positions.
- **TILED_SPARSE** — DICOM organisation with explicit per-frame positions.
- **Optical path** — the light path a frame was acquired through.
- **Associated image** — label, macro, or thumbnail stored beside the pyramid.
- **SVS / NDPI / MRXS** — Aperio, Hamamatsu, and 3DHISTECH native containers.
- **DICOM WSI** — the VL Whole Slide Microscopy Image object from Supplement 145.
- **OpenSlide** — the reference library for reading native WSI formats.
- **Manifest** — the table of patch coordinates and metadata for a slide.
- **Shard** — a fixed-size output file holding many rows.
- **Working set** — the tiles a single pass over a slide must hold to avoid re-reading.
- **Range request** — an HTTP read of a byte range within an object.
- **Idempotency** — repeating an operation yields the same stored result.
- **Provenance** — the recorded origin and parameters of a derived artefact.
- **VNA** — vendor neutral archive; a multi-specialty image archive.

## 16. Sources

- DICOM Standards Committee. *Supplement 145: Whole Slide Microscopic Image IOD and SOP
  Classes*. Final Text, 2010. https://dicom.nema.org/MEDICAL/Dicom/Final/sup145_ft.pdf
- DICOM PS3.3 C.7.6.17.3, *Spatial Location and Optical Path of Tiled Images*. https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.17.3.html
- DICOM PS3.3 C.8.12.4, *Whole Slide Microscopy Image Module*.
  https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.12.4.html
- OpenSlide. *Virtual slide formats understood by OpenSlide*.
  https://openslide.org/formats/ (accessed 27 October 2026).
- Evans, A. J., Brown, R. W., Bui, M. M., et al. (2022). Validating Whole Slide Imaging Systems
  for Diagnostic Purposes in Pathology: Guideline Update from the CAP in Collaboration with ASCP
  and API. *Archives of Pathology and Laboratory Medicine*, 146(4), 440-450.
  doi:10.5858/arpa.2020-0723-CP
- US Food and Drug Administration. (2016). *Technical Performance Assessment of Digital Pathology
  Whole Slide Imaging Devices*. Guidance for Industry and FDA Staff.
- US Food and Drug Administration. (2024). *Medical Devices; Laboratory Developed Tests*. Final
  rule, 89 FR 37286.
- US Food and Drug Administration. (2025). *Medical Devices; Laboratory Developed Tests;
  Implementation of Vacatur*. Final rule, 90 FR 45134.
- *American Clinical Laboratory Association v. US Food and Drug Administration*, No. 4:24-cv-479
  (E.D. Tex., 31 March 2025).
- Leica Biosystems. *Aperio GT 450 Specifications*, MAN-0393 Rev L.
  https://www.leicabiosystems.com/
- Hamamatsu Photonics. *NanoZoomer S360 Brochure*, SBIS0144E.
  https://nanozoomer.hamamatsu.com/
- Philips. *IntelliSite Ultra Fast Scanner Brochure*.
  https://www.philips.com/healthcare/
- Macenko, M., Niethammer, M., Marron, J. S., et al. (2009). A Method for Normalizing Histology
  Slides for Quantitative Analysis. *IEEE International Symposium on Biomedical Imaging*,
  1107-1110. doi:10.1109/ISBI.2009.5193250
- Janowczyk, A., Zuo, R., Gilmore, H., Feldman, M., & Madabhushi, A. (2019). HistoQC: An
  Open-Source Quality Control Tool for Digital Pathology Slides. *JCO Clinical Cancer
  Informatics*, 3, 1-7. doi:10.1200/CCI.18.00157
- Campanella, G., Hanna, M. G., Geneslaw, L., et al. (2019). Clinical-Grade Computational
  Pathology Using Weakly Supervised Deep Learning on Whole Slide Images. *Nature Medicine*,
  25(8), 1301-1309. doi:10.1038/s41591-019-0508-1

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous: [`10_foundation-models.md`](10_foundation-models.md)
- Next: [`12_pathology-foundation-models.md`](12_pathology-foundation-models.md)
- Operations: [`13_digital-pathology-lab-operations.md`](13_digital-pathology-lab-operations.md)
- Survey file: [`../../course/10_medical_imaging.md`](../../course/10_medical_imaging.md)
- Lab catalogue: [`../../labs/`](../../labs/)
- Cloud and orchestration lab: [`../../labs/cloud/`](../../labs/cloud/)
