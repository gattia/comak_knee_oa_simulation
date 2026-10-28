# Figure 1: vendored images

Figure 1 (the pipeline overview) is built around one example knee, `9018389_00m_RIGHT`. Most of its
pictures are rendered from the data root by `pipeline/derive_figure1_panels.py`, into
`04_results/figure1_pipeline_overview/panels/`. The seven images here cannot be made that way, because
their source is outside the data root or they are drawn artwork. They are stored as files, with md5s
below.

| file | what it is | why it is vendored, not derived |
|---|---|---|
| `mri_raw.png` | sagittal DESS slice 65 of the example knee (OAI baseline, visit 00m, participant 9018389), 1st–99.7th percentile window, gamma 0.9, cropped rows 35–345 / cols 15–300, ×4 Lanczos | the OAI DICOM is not in the data root and may not be redistributed; only the segmentations are (`00_inputs/segmentations/00m/9018389/`) |
| `mri_seg.png` | the same slice with the segmentation overlaid: bone (labels 5/6/7) sand at 50 %, cartilage (1–4) blue at 75 %, menisci (9/10, from the `bones_cart_men_fib` network) orange at 75 % | the greyscale underneath is the DICOM; the labels are the two `.nrrd` files in `00_inputs/segmentations/00m/9018389/` |
| `icon_crowd.png` | pictogram of a group of people (the cohort) | artwork |
| `icon_mri.png` | pictogram of an MRI scanner (a visit on the thickness timeline) | artwork |
| `Heelstrike.png`, `Midstance.png`, `Toeoff.png` | walking silhouettes over the three pressure instants | artwork |

The figure recolours the icons and silhouettes (grey or ink) when it places them, so their own colour
does not matter.

```
4dd87d0b7c2ce991b371089fa22ab09b  mri_raw.png
f3f951c4f1ac52a8aba72bdad2cc28c2  mri_seg.png
0b3baaf361a356455f64cef6f86024f7  icon_crowd.png
f7b7d74af2d1b00495052b6855590e4f  icon_mri.png
18efef127068245ffab8db5a3b0e22b6  Heelstrike.png
a7eab5cd9de2b53738cb6120df0b990d  Midstance.png
c7790775549b729b31dd5a3dd36abf6a  Toeoff.png
```
