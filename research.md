# Research: VFX Compositing Techniques for Wavelet-Domain Image Enhancement

Notes from a research pass into whether techniques from VFX compositing / photo
retouching can be borrowed to improve the Haar wavelet enhancement pipeline in
[`backend/app/processing/wavelet_haar_transform.py`](backend/app/processing/wavelet_haar_transform.py).
Written as a guide for future development, not a finished spec — pick items up
as they become relevant.

## The core insight

A single-level 2D Haar transform splits an image into four sub-bands: `LL`
(approximation/tone), and `LH`/`HL`/`HH` (horizontal/vertical/diagonal detail).
That is structurally the same move VFX compositors make when they split a
render into passes (beauty, specular, AO, edge) or when photo retouchers use
**frequency separation** to split texture from tone onto separate layers they
can grade independently and recombine. The current pipeline
(`enhance_high_frequency_bands` in `wavelet_haar_transform.py`) already does
one round of this: boost the detail bands, then inverse-transform. Everything
below is either a refinement of that same idea, or an adjacent VFX technique
that maps cleanly onto the same band structure.

## Techniques, roughly in the order worth implementing

### 1. Multi-level decomposition (pyramid / mip-map analogy)

The current transform does exactly **one** level — one fine-detail band.
Burt & Adelson's 1983 Laplacian pyramid (the basis of most multi-band
compositing and image blending, and directly analogous to mip-maps/LOD levels
in 3D) recurses on the low-frequency band repeatedly, producing several
detail scales: fine grain, medium structure, macro tone. Applied here: run
`haar_transform_2d` again on its own `LL` output, `N` times, enhance each
level with its own factor/curve, then invert level-by-level back up. This is
the single highest-leverage change available — it turns "one detail slider"
into a proper multi-scale detail tool.

- Burt, P. & Adelson, E. (1983), *A Multiresolution Spline With Application to
  Image Mosaics* — https://ai.stanford.edu/~kosecka/burt-adelson-spline83.pdf
- Multiresolution pyramid blending walkthrough —
  https://blogs.mathworks.com/steve/2019/05/20/multiresolution-pyramids-part-4-image-blending/

### 2. Local Laplacian Filters — edge-aware, halo-free detail boosting

`enhance_high_frequency_bands` currently does a flat `*= factor` on every
detail coefficient. That can't distinguish a real edge from noise, so pushing
`factor` up ropes in ringing/halos around strong edges (visible around the
skull boundary in test renders already). Paris, Hasinoff & Kautz's *Local
Laplacian Filters* (SIGGRAPH 2011 — the algorithm behind Photoshop's
Clarity/Dehaze sliders) solve this with a nonlinear remapping function per
pyramid level that treats small-magnitude coefficients (fine detail)
differently from large ones (true edges), using only simple point-wise
nonlinearities and small Gaussian convolutions — no optimization or
postprocessing pass required, and it produces consistently high-quality
results without degrading edges or introducing halos. Swapping the flat
multiply for a per-band S-curve/soft-clip is a small code change with a
disproportionate quality gain.

- Paris, S., Hasinoff, S. W. & Kautz, J. (2011), *Local Laplacian Filters:
  Edge-Aware Image Processing with a Laplacian Pyramid* —
  https://dl.acm.org/doi/fullHtml/10.1145/2723694
- Reference implementation — https://github.com/psalvaggio/local_laplacian_filters

### 3. Denoise-before-regrain (wavelet soft-thresholding)

Standard VFX "degrain → work → regrain" workflow, translated to wavelet-speak:
soft-threshold the detail coefficients before amplifying them (Donoho's
classic wavelet denoising). Small-magnitude `LH`/`HL`/`HH` coefficients are
usually noise, not signal — boosting them with the current flat multiply
amplifies noise right along with real edges. Soft thresholding modifies
wavelet coefficients by reducing their magnitude based on a threshold chosen
from the estimated noise standard deviation, attenuating noise while
preserving significant image features and maintaining sharpness/texture more
naturally than hard thresholding. This is directly validated in MRI-specific
literature (Rician noise) and is the wavelet-domain equivalent of
denoise-then-regrain.

- Wavelet denoising overview —
  https://www.sciencedirect.com/topics/computer-science/wavelet-denoising
- Medical image denoising via optimal wavelet-coefficient thresholding —
  https://onlinelibrary.wiley.com/doi/10.1002/ima.22589

### 4. Guided/bilateral filtering as an edge-aware base-layer pass

In comp, a blur-and-subtract split between base and detail is rarely done
with a plain Gaussian if avoidable — the **guided filter** (edge-aware,
linear-time) is the modern default, because it avoids the gradient-reversal
artifacts a bilateral filter's base layer can introduce at edges. Applying
this to the reconstructed `LL` band before recombination would be cheaper and
cleaner than a naive box/Gaussian smooth, with applications documented for
detail enhancement, denoising, and tone mapping generally.

- He, K., Sun, J. & Tang, X. (2012), *Guided Image Filtering* —
  https://www.sensetime.com/xo/profile/upload/2024/05/23/2012%20Guided%20Image%20Filtering_20240523184437A013.pdf
- Guided filter overview and comparison to bilateral filtering —
  https://en.wikipedia.org/wiki/Guided_filter

### 5. Blend-mode recombination instead of linear add

Recombination today is pure linear arithmetic — multiply the detail bands,
then inverse-transform. Comp software instead recombines layers with
non-linear **blend modes** (Screen, Overlay, Soft Light, Linear Light)
specifically because linear addition clips highlights and crushes shadows.
Frequency-separation retouching workflows conventionally recombine the high-
frequency (texture) layer onto the low-frequency (tone) layer using Linear
Light for exactly this reason. Formulas (all trivial to vectorize in numpy,
values normalized to 0-1):

- Screen: `1 - (1-a)*(1-b)`
- Overlay: `a<0.5 ? 2ab : 1-2(1-a)(1-b)`
- Soft Light: `b<0.5 ? 2ab+a²(1-2b) : sqrt(a)(2b-1)+2a(1-b)`

Using one of these to blend the boosted detail layer back onto the base
(rather than adding raw coefficients pre-inverse-transform) would give a more
filmic "detail pops without blowing out" result.

- Frequency separation retouching technique (Linear Light recombination) —
  https://fstoppers.com/post-production/ultimate-guide-frequency-separation-technique-8699
- Blend mode formulas reference — https://godotshaders.com/snippet/blending-modes/

### 6. Luminance matte / holdout mask

Comp artists routinely qualify an edit by luma (a "luma key") so a grade only
hits, say, midtones — protecting highlights/shadows from a change meant for
the middle. Applied here: derive a soft mask from the reconstructed/`LL`
band's brightness and use it to gate how hard the detail boost hits near-black
(background) and near-white (skull/bone) pixels — exactly where wavelet
ringing is worst and where "extra detail" is least meaningful anyway.

No single canonical source for this one — it's a direct port of standard
luma-qualifier / garbage-matte practice in color grading and compositing
rather than a named published technique.

## Caution specific to this domain

Radiology literature is consistent: sharpening/wavelet-boost techniques can
introduce ringing that reads as structure that isn't actually there — a
"false detail" risk, not just an aesthetic nitpick. Sharpening is one of the
most-used enhancement methods, but alongside the edge-enhancement improvement
it produces, artifacts appear that can lead to misdiagnosis if the image is
read clinically. That's not a blocker for this project (a portfolio/demo tool
exploring VFX-inspired enhancement, not a diagnostic device), but it's worth a
standing code comment/README caveat if this pipeline is ever repurposed toward
anything diagnostically meaningful rather than visualization/exploration.

- Sharpening artifact risk discussion (X-ray/medical imaging) —
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9777674/
- Wavelet thresholding to reduce ultrasound artifacts —
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3023837/

## How this lands in the existing code

All six items are additive and independently toggleable; they don't require
restructuring what's already in
[`wavelet_haar_transform.py`](backend/app/processing/wavelet_haar_transform.py):

| # | Technique | Touches |
|---|-----------|---------|
| 1 | Multi-level decomposition | new `levels` param; recurse `haar_transform_2d`/`inverse_haar_transform_2d` on `LL` |
| 2 | Local-Laplacian-style nonlinear gain | replace the flat `*=` in `enhance_high_frequency_bands` with a per-band curve |
| 3 | Soft-threshold denoise | new step before the gain curve, applied to `LH`/`HL`/`HH` |
| 4 | Guided-filter base pass | optional filter on `LL` before recombination |
| 5 | Blend-mode recombination | swap the array-assignment recombine step for a blend-mode function |
| 6 | Luminance holdout mask | derived from `LL`, applied as a gate right before recombination |

Each would extend the existing `params` dict pattern already used for
`factor` in
[`processing_service.py`](backend/app/services/processing_service.py) (`_op_wavelet_enhance`),
so the API/job shape doesn't need to change — just new optional keys per
technique.

## Suggested next step

Start with **#1 (multi-level)** and **#2 (nonlinear gain)** together — they
give the biggest visible quality upgrade for the smallest amount of new
surface area, and everything else in this document builds naturally on top of
having multiple decomposition levels to apply it to.
