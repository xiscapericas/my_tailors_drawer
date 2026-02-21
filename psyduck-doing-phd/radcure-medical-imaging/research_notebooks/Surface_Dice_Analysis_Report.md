# Comparative Analysis of Dice and Surface Dice for GTVp Segmentation in RADCURE

## 1. Introduction

This report presents a comparative evaluation of Dice coefficient and
Surface Dice for the assessment of primary gross tumor volume (GTVp)
segmentation performance using the RADCURE dataset.

The motivation for introducing Surface Dice into the evaluation
framework arises from the nature of RADCURE annotations. The dataset was
delineated by radiotherapists, and therefore contour precision may vary
across cases. In particular, tumor boundaries may extend over adjacent
anatomical structures (e.g., skull, tongue), resulting in contour
irregularities and variability in ground truth definition. Under these
conditions, conventional overlap-based metrics such as Dice may unfairly
penalize otherwise clinically acceptable segmentations.

This analysis investigates whether Surface Dice provides complementary
or more appropriate performance characterization under such contour
uncertainty.

------------------------------------------------------------------------

## 2. Theoretical Background

### 2.1 Dice Coefficient

Dice measures volumetric overlap between prediction and ground truth:

Dice = 2 \|P ∩ G\| / (\|P\| + \|G\|)

It penalizes:

- Boundary inaccuracies
- Under-segmentation
- Over-segmentation
- Spatial misalignment

Dice is sensitive to small contour differences and may penalize
predictions even when spatial localization is clinically acceptable.

------------------------------------------------------------------------

### 2.2 Surface Dice

Surface Dice measures the agreement between predicted and ground truth
boundaries within a predefined spatial tolerance.

Instead of evaluating volumetric overlap, it evaluates how much of the
surfaces lie within a specified distance from each other.

Surface Dice:

- Is tolerant to small boundary variations within tolerance
- Penalizes large boundary displacements
- Captures spatial proximity even when volumetric overlap is limited

Thus, Surface Dice evaluates boundary agreement rather than pure volume
overlap.

------------------------------------------------------------------------

## 3. Quantitative Results

### 3.1 Correlation Between Metrics

A linear regression analysis between Dice and Surface Dice shows:

- R² = 0.71

This indicates a strong positive correlation, meaning both metrics
evaluate related aspects of segmentation performance. However, 29% of
variance remains unexplained, suggesting that the metrics are not
interchangeable and capture complementary information.

------------------------------------------------------------------------

### 3.2 Central Tendency Comparison

- Median Dice: 0.5878
- Median Surface Dice: 0.5776

Despite some large improvements in specific cases, the median Surface
Dice is slightly lower than Dice. This indicates that the apparent
improvement in average performance is driven by extreme cases rather
than a consistent global shift.

Percentage improvements exceeding 200% occur primarily in cases with
very low Dice values, where small absolute increases produce large
relative changes. Therefore, absolute differences are more reliable than
percentage improvement when interpreting results.

------------------------------------------------------------------------

## 4. Case Redistribution Analysis

Cases were grouped into four performance categories:

- LOW: 0–0.3
- MIDDLE: 0.31–0.5
- GOOD: 0.51–0.7
- ABOVE: 0.71–1

**Contingency table:** Dice category (rows) vs Surface Dice category (columns) (`dice_gtv_cat` × `sur_dice_gtv_cat`):

| dice_gtv_cat | ABOVE | GOOD | LOW | MIDDLE |
|--------------|-------|------|-----|--------|
| ABOVE        | 17    | 6    | 0   | 0      |
| GOOD         | 8     | 7    | 1   | 6      |
| LOW          | 0     | 0    | 11  | 3      |
| MIDDLE       | 0     | 6    | 1   | 6      |

Surface Dice redistributed cases rather than uniformly increasing performance:

- 17 cases improved category
- 14 cases decreased category

This near symmetry indicates that Surface Dice is not systematically more optimistic but instead shifts performance characterization.

------------------------------------------------------------------------

## 5. Behavioral Patterns Identified

### 5.1 Large Ground Truth with Under-Segmentation

In cases where the ground truth volume is substantially larger than the prediction:

- Dice may remain moderate due to partial overlap.
- Surface Dice decreases because large portions of the ground truth boundary are far from the predicted surface.

Surface Dice penalizes large boundary discrepancies more strongly than
Dice.

------------------------------------------------------------------------

### 5.2 Spatially Close but Poorly Overlapping Tumors

In cases where:

- The tumor is slightly displaced, or
- Mirrored but spatially close

Dice can be very low due to limited volumetric overlap. Surface Dice may remain moderate if boundaries are within tolerance.

This demonstrates that Surface Dice captures spatial proximity even when
overlap is poor.

------------------------------------------------------------------------

### 5.3 Importance of Slice Coverage

Across high-performing cases:

- Tumor detected in all slices
- Prediction consistently inside ground truth

Both metrics perform well when:

- Z-extent is correctly captured
- Tumor presence is consistently detected

Improving slice-level detection and spatial extent modeling may yield
greater performance gains than focusing solely on contour refinement.

------------------------------------------------------------------------

## 6. Conclusion

Surface Dice provides a valuable complementary metric for evaluating
tumor segmentation in datasets with contour uncertainty. While it does
not fundamentally alter overall performance trends, it offers additional
spatial insight, particularly in cases of boundary noise, spatial
displacement, or slice-wise detection inconsistencies.

For robust evaluation in RADCURE, both Dice and Surface Dice should be
reported, as together they provide a more comprehensive understanding of
segmentation behavior.
