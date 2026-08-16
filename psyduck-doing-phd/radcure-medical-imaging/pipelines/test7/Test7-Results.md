# Test 7 — Soft Probability Analysis of GTVp–Anatomy Competition

**Status:** results write-up (probability characterisation)  
**Config:** [`experiments/configs/test7_radheck_probability_outputs.yaml`](../../experiments/configs/test7_radheck_probability_outputs.yaml)  
**Runbook:** [`pipelines/radheck/Retrain-Radheck-Test7.md`](../radheck/Retrain-Radheck-Test7.md)  
**Changed from:** Test 5 (model + preprocess); same Tr/Va/Ts membership as Test 6

---

## 1. Objective

Test 7 investigates whether the **soft voxel-wise probability outputs** of the nnU-Net model contain useful tumour information that is lost when predictions are converted to hard multiclass labels using `argmax`.

The experiment builds directly on the model and dataset used in Test 5 and Test 6. No model retraining or additional TotalSegmentator processing is performed.

**Central hypothesis:**

> A voxel may contain meaningful evidence for GTVp even when an anatomical class has a higher predicted probability and therefore wins the multiclass argmax.

In particular, Test 7 investigates whether voxels with high GTVp probability — including voxels where another anatomical class wins the argmax — can still reliably correspond to GTVp in the ground truth.

A candidate operating rule considered in this experiment is:

$$
P(\mathrm{GTVp}) \geq 0.80
$$

with the goal of determining whether such voxels can be recovered as tumour without introducing an unacceptable number of false positives.

---

## 2. Motivation

Standard multiclass nnU-Net inference converts the predicted probability distribution at every voxel into a single class:

$$
\hat{y}(v) = \arg\max_c P(c \mid v)
$$

where $c$ represents one of the anatomical structures or GTVp. This representation discards the remainder of the probability distribution.

For example, consider a voxel with:

$$
P(\mathrm{GTVp}) = 0.80, \qquad P(\mathrm{Tongue}) = 0.85
$$

If tongue has the maximum probability, the final hard segmentation assigns the voxel to tongue, despite the model simultaneously expressing strong confidence that the voxel may belong to GTVp.

This is particularly relevant for head-and-neck tumour segmentation because tumours can infiltrate, replace, distort, or overlap anatomical structures represented by the anatomical segmentation classes.

Consequently, poor tumour segmentation may not always indicate that the network failed to recognise the tumour. In some cases, tumour evidence may already exist in the softmax output but be removed during the final multiclass decision.

Test 7 therefore evaluates the model **before this information is collapsed by argmax**.

---

## 3. Experimental setup

### 3.1 Model

Test 7 uses the trained Test 5 nnU-Net model:

| Item | Value |
|------|--------|
| Dataset | `Dataset650` |
| Trainer | `nnUNetTrainer_700epochs_NoMirroring` |
| Epochs | 700 |
| Mirroring | disabled |
| Split | same Tr / Va / Ts as Test 5 and Test 6 |

No retraining was performed for Test 7.

---

## 4. Pipeline

The Test 7 pipeline consists of four main stages.

### 4.1 Reuse of Test 5 data

The existing Dataset650 data and Test 5 / Test 6 split were linked into a clean Test 7 working directory.

This ensures that differences observed in Test 7 originate from the analysis of the probability outputs rather than changes in training data, preprocessing, anatomical segmentations, or model configuration.

### 4.2 Probability prediction

Inference was performed on `imagesTs` using the Test 5 checkpoint with probability saving enabled (`--save_probabilities`).

Instead of analysing only the final hard segmentation, this provides the voxel-wise softmax probability distribution across GTVp and the anatomical classes.

### 4.3 Slim probability archives

The original nnU-Net probability outputs are large. Raw probability volumes were converted into cropped `float16` probability archives containing the tumour region of interest and the relevant GTVp / anatomical probabilities.

Crops were axis-aligned with the corresponding ground-truth segmentation. This substantially reduces storage while retaining the information required for Test 7.

### 4.4 Probability analysis

Two main analyses were performed:

1. **Region-vs-GTVp probability curves** (primary quantitative analysis below)
2. **Probability visualisation** — full CT; CT + GTVp GT; soft overlay with alpha = $P(\mathrm{class})$; Test 6 / `MedicalImageVisualizer` colormap

---

## 5. Region-vs-GTVp probability curves

For each anatomical region, Test 7 compares the probability assigned to GTVp with the probability assigned to that anatomical structure.

The probability margin is defined as:

$$
\Delta P = P(\mathrm{GTVp}) - P(\mathrm{region})
$$

### Interpretation of the $x$-axis

| Condition | Meaning |
|-----------|---------|
| $\Delta P > 0$ | Model considers GTVp more probable than the anatomical structure |
| $\Delta P < 0$ | Anatomical structure has higher probability than GTVp |
| $\Delta P = 0$ | Approximately equal probability |

The $y$-axis is the empirical association with **ground-truth GTVp membership** for voxels at the corresponding probability margin.

The desired behaviour is not necessarily that every curve converges to $(1,1)$. An informative curve should generally demonstrate:

$$
P(\mathrm{GT}=\mathrm{GTVp}) \uparrow \quad \text{as} \quad P(\mathrm{GTVp}) - P(\mathrm{region}) \uparrow
$$

In other words, increasing relative confidence in GTVp should correspond to an increasing probability that the voxel actually belongs to the tumour.

---

## 6. Results

### 6.1 Overall behaviour

The pooled region curves demonstrate that the relationship between anatomical and tumour probabilities is highly structured rather than random.

For multiple anatomical classes, the probability of ground-truth tumour membership increases substantially as $P(\mathrm{GTVp}) - P(\mathrm{region})$ increases.

Several regions demonstrate a particularly sharp transition close to $\Delta P = 0$. This indicates that the point at which GTVp begins to outrank the competing anatomical class is strongly associated with actual tumour membership.

This behaviour is visible in regions including, among others:

- tongue
- soft palate
- oropharyngeal structures
- thyroid
- skull
- lateral and medial pterygoid regions
- internal jugular / vascular structures
- several other anatomical classes

The exact strength of this relationship varies considerably between regions.

### 6.2 Strong tumour–anatomy separation

A first group of structures demonstrates relatively clear separation between anatomical-dominant and GTVp-dominant voxels.

For these structures, voxels with $P(\mathrm{GTVp}) < P(\mathrm{region})$ tend to have a relatively low frequency of GTVp in the ground truth. Once $P(\mathrm{GTVp}) > P(\mathrm{region})$, the empirical GTVp frequency rises substantially, in some regions approaching very high values.

This suggests that the relative GTVp probability is a meaningful indicator of tumour membership, and that the soft probability distribution contains information that cannot be represented by the final multiclass label alone.

---

## 7. Tumour–anatomy competition

A second and particularly important pattern occurs when voxels remain associated with ground-truth GTVp despite substantial probability being assigned to an anatomical class.

Examples appear in structures such as:

- submandibular gland
- internal carotid artery
- tongue
- pterygoid regions
- other structures anatomically adjacent to common head-and-neck tumour sites

These regions are particularly relevant to the central Test 7 hypothesis:

> An anatomical class having substantial probability — or even exceeding the GTVp probability — does not necessarily imply that tumour evidence is absent.

This may occur because the model detects both the underlying anatomical context and features associated with tumour involvement.

However, the probability curves alone cannot establish that this represents genuine biological overlap or tumour infiltration. For this reason, the observed phenomenon is described conservatively as **tumour–anatomy competition**. Spatial inspection of the corresponding voxels is required before interpreting individual structures as demonstrating true tumour–organ overlap.

---

## 8. Tongue

The tongue provides one of the most interesting examples.

The tongue curve demonstrates substantial variability when $P(\mathrm{GTVp}) < P(\mathrm{Tongue})$, followed by a pronounced increase in ground-truth tumour membership as the GTVp probability approaches and exceeds the tongue probability. After the transition around $P(\mathrm{GTVp}) \approx P(\mathrm{Tongue})$, ground-truth GTVp membership remains high.

This indicates that the relative competition between tongue and GTVp is highly informative.

More importantly, the negative-margin region requires further investigation. If a meaningful population of voxels satisfies:

$$
P(\mathrm{Tongue}) > P(\mathrm{GTVp}) \quad \text{and} \quad \mathrm{GT} = \mathrm{GTVp}
$$

then these voxels would become false negatives under standard multiclass argmax despite the model retaining substantial GTVp probability.

This represents exactly the failure mode that Test 7 was designed to investigate.

---

## 9. Potential argmax information loss

The results support a distinction between two different types of segmentation error.

### Model recognition failure

The model assigns very low probability to GTVp ($P(\mathrm{GTVp}) \approx 0$). Tumour information is genuinely absent from the model output.

### Decision-rule failure

The model assigns substantial probability to GTVp, but another class receives a slightly higher probability ($P(\mathrm{region}) > P(\mathrm{GTVp})$). The voxel is therefore assigned to the anatomical structure by argmax.

If the voxel actually belongs to GTVp, the model has not completely failed to recognise tumour. Instead, useful tumour information existed in the probability distribution but was discarded by the final decision rule.

Test 7 provides evidence that the second type of error occurs within the Dataset650 model.

This distinction matters because the two failure modes require different solutions:

- **Recognition failures** may require changes to training, architecture, data, loss functions, or anatomical context.
- **Decision-rule failures** may potentially be addressed **without retraining the network**.

---

## 10. Interpretation of the $P(\mathrm{GTVp}) \geq 0.80$ hypothesis

One of the original motivations for Test 7 was to investigate the rule $P(\mathrm{GTVp}) \geq 0.80$ even when GTVp does not win the argmax.

The pooled probability-margin curves support the broader hypothesis that GTVp probability contains useful information beyond the hard segmentation.

However, the current curves do **not independently validate 0.80 as the optimal threshold**.

The threshold hypothesis requires direct analysis of voxels satisfying:

$$
P(\mathrm{GTVp}) \geq 0.80 \quad \text{and} \quad \arg\max_c P(c) \neq \mathrm{GTVp}
$$

The key quantity is then:

$$
P\bigl(\mathrm{GT}=\mathrm{GTVp} \mid P(\mathrm{GTVp})\geq 0.80,\; \arg\max(P)\neq \mathrm{GTVp}\bigr)
$$

This is the positive predictive value of rescuing voxels that have high GTVp probability but lose the multiclass competition.

Therefore, the current Test 7 results should be interpreted as **evidence supporting further evaluation of the rescue hypothesis**, rather than proof that 0.80 is already an optimal operating threshold.

---

## 11. Limitations

### 11.1 Sparse anatomical regions

A major limitation of the pooled figure is the variable number of observations available for different anatomical structures.

Some curves contain many probability bins and demonstrate clear trends. Other structures contain only one or a small number of observations. A single point at high ground-truth GTVp probability should therefore not be interpreted as strong evidence of tumour–anatomy interaction.

Future versions of the analysis should report the number of contributing voxels for every probability bin.

### 11.2 Number of patients

Voxel count alone is insufficient. A large number of voxels may originate from only one or two patients.

Consequently, the number of independent patients contributing to each region and probability interval should also be reported. Useful quantities include:

- number of voxels per bin
- number of patients per bin
- number of tumour-positive patients per anatomical region

### 11.3 Voxel-level dependence

Individual voxels are not statistically independent observations. Neighbouring voxels originate from the same tumour and the same patient and are therefore strongly correlated.

Pooling millions of voxels can consequently produce apparently strong statistical evidence even when the phenomenon is driven by a small subset of patients.

The patient should therefore be treated as the primary experimental unit for subsequent statistical evaluation.

### 11.4 Biological interpretation

A probability conflict between GTVp and an anatomical structure does not automatically demonstrate true tumour invasion of that structure.

Possible explanations include:

- genuine tumour infiltration
- partial-volume effects
- tumour boundaries
- image ambiguity
- registration or segmentation uncertainty
- anatomical distortion caused by the tumour
- model confusion
- class competition introduced by the multiclass formulation

Spatial visualisation is therefore necessary before assigning a biological explanation to individual curves.

---

## 12. Required follow-up analysis

The next analysis should directly test whether probability-based tumour rescue improves segmentation.

### 12.1 Baseline prediction

The standard prediction is:

$$
\hat{Y}_{\mathrm{baseline}} = \arg\max_c P(c)
$$

### 12.2 Probability rescue

A simple Test 7 rescue rule can be defined as:

$$
\hat{Y}_{\mathrm{rescue}} =
\begin{cases}
\mathrm{GTVp}, & P(\mathrm{GTVp}) \geq T \\
\arg\max_c P(c), & \text{otherwise}
\end{cases}
$$

where $T$ is the GTVp probability threshold.

The initial hypothesis uses $T = 0.80$, but multiple thresholds should be evaluated rather than assuming that 0.80 is optimal. Potential thresholds include:

`0.50`, `0.60`, `0.70`, `0.75`, `0.80`, `0.85`, `0.90`, `0.95`

---

## 13. Evaluation metrics

The baseline and rescue predictions should be compared **per patient**.

Primary metrics:

- Dice score
- tumour recall / sensitivity
- tumour precision
- false-positive volume
- false-negative volume

For each threshold and every test patient:

$$
\Delta\mathrm{Dice} = \mathrm{Dice}_{\mathrm{rescue}} - \mathrm{Dice}_{\mathrm{baseline}}
$$

$$
\Delta\mathrm{Recall} = \mathrm{Recall}_{\mathrm{rescue}} - \mathrm{Recall}_{\mathrm{baseline}}
$$

$$
\Delta\mathrm{Precision} = \mathrm{Precision}_{\mathrm{rescue}} - \mathrm{Precision}_{\mathrm{baseline}}
$$

The critical question is not simply whether additional tumour voxels can be recovered. The relevant question is:

> Can high-probability GTVp voxels lost through anatomical argmax competition be recovered while maintaining acceptable tumour precision?

---

## 14. Patient-level statistical analysis

For each anatomical structure and patient, the analysis should also compare ground-truth tumour membership under positive and negative probability margins, for example:

$$
P(\mathrm{GT}=\mathrm{GTVp} \mid \Delta P > 0)
\quad\text{versus}\quad
P(\mathrm{GT}=\mathrm{GTVp} \mid \Delta P < 0)
$$

Results should then be summarised across patients rather than treating every voxel as an independent observation.

Recommended summaries:

- median
- interquartile range
- mean where appropriate
- confidence intervals
- number of contributing patients

This would determine whether the patterns visible in the pooled curves are reproducible across the test cohort.

---

## 15. Recommended improvements to the curves

Future versions of the region probability plots should include:

| Improvement | Purpose |
|-------------|---------|
| Bin sample size $n_{\mathrm{voxels}}$ | Support per probability-margin bin |
| Patient count $n_{\mathrm{patients}}$ | Independence / generalisability |
| Confidence intervals | Uncertainty on empirical GTVp proportion |
| Minimum-support filtering | Remove or mark low-support bins / curves |
| Patient-level curves | Separate cohort-wide vs patient-specific effects |

These additions would make the figures considerably easier to interpret statistically.

---

## 16. Main findings

1. **Soft probability outputs contain information not represented in hard argmax segmentations.**

2. **The probability margin between GTVp and anatomical structures is associated with ground-truth tumour membership for multiple anatomical regions.**

3. **Several structures demonstrate a pronounced transition around** $P(\mathrm{GTVp}) \approx P(\mathrm{region})$, **suggesting meaningful tumour–anatomy probability competition.**

4. **Some ground-truth tumour voxels may retain substantial GTVp probability even when an anatomical class dominates the multiclass prediction.**

5. **This provides a plausible mechanism for false-negative tumour voxels produced by hard multiclass argmax.**

6. **The current analysis supports investigation of a GTVp rescue rule but does not yet establish 0.80 as the optimal threshold.**

7. **Sparse anatomical regions and voxel-level dependence currently limit the strength of conclusions that can be drawn from individual curves.**

---

## 17. Conclusion

Test 7 demonstrates that evaluating only the final hard multiclass segmentation does not fully characterise the behaviour of the Dataset650 model.

The softmax outputs reveal structured competition between GTVp and anatomical classes. Across multiple anatomical regions, increasing relative GTVp probability is associated with increasing ground-truth tumour membership, frequently with a pronounced transition near the point where GTVp and the competing anatomical class have equal probability.

Most importantly, the results indicate that some tumour information may survive within the soft probability distribution even when GTVp loses the multiclass argmax.

This suggests that a subset of apparent tumour false negatives may represent **decision-rule failures rather than complete tumour-recognition failures**.

The finding provides the rationale for testing a tumour-aware inference rule in which high-confidence GTVp predictions can override competing anatomical classes.

However, Test 7 should currently be considered a **probability-characterisation experiment rather than evidence for a validated post-processing method**. The proposed $P(\mathrm{GTVp}) \geq 0.80$ rescue rule must be evaluated directly against the standard argmax segmentation at the patient level.

If probability-based rescue consistently increases tumour recall and Dice without an unacceptable loss of precision, the result would demonstrate that anatomical context and tumour prediction do not necessarily need to be treated as mutually exclusive outputs.

Instead, anatomical segmentation could provide contextual information while tumour-specific decision logic determines the final GTVp prediction.

---

## 18. Position within the experimental framework

```text
Test 5
│
├── Can anatomical context improve GTVp segmentation?
│
▼
Test 6
│
├── Where and how does anatomical context affect tumour prediction?
│
▼
Test 7
│
├── Does useful tumour evidence remain in the soft outputs
│   when anatomy wins the hard multiclass prediction?
│
▼
Next experiment
│
└── Can tumour-aware probability rescue convert that
    information into improved patient-level segmentation?
```

Test 7 therefore provides a bridge between anatomical-context segmentation and a potential **tumour-aware decision framework**.

The key next step is not additional model training, but quantitative validation of whether the information identified in the soft outputs can produce a measurable and reproducible improvement in GTVp segmentation.
