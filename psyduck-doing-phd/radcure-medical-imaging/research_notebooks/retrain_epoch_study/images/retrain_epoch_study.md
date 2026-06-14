# RADHECK Retraining Study: Impact of Epoch Count on Generalisation

## Motivation

The first RADHECK retraining experiment was performed using the default nnU-Net configuration with 1000 epochs.

Analysis of the training curves suggested that model performance started to deteriorate before the end of training. Around epoch 700, the gap between training and validation losses began to increase, while several inflection points appeared in the validation curve. These patterns are commonly associated with overfitting, indicating that the model may have started memorising training data instead of improving generalisation.

## Training Progress – 1000 Epoch Configuration

![Training Progress 1000 Epochs](images/progress_test2_1000epoch.png)

## Goal

The objective of this experiment was to evaluate whether reducing the training duration from 1000 to 700 epochs could mitigate overfitting while maintaining or improving segmentation performance.

---

## Training Behaviour

The training curves obtained after reducing the total number of epochs to 700 show a more stable behaviour compared with the previous experiment.

## Training Progress – 700 Epoch Configuration

![Training Progress 700 Epochs](images/progress_test3_700_epoch.png)

Although the separation between training and validation losses is reduced, the same change point observed in the 1000-epoch configuration is still present. The main difference is that it now appears earlier in the training process.

This behaviour may be explained by the nnU-Net learning-rate schedule. Since nnU-Net applies an epoch-dependent learning-rate decay, reducing the total number of epochs shifts the final low-learning-rate phase earlier in training. Consequently, the same optimisation behaviour is observed at an earlier stage.

These observations suggest that:

* The model may still be under-optimised.
* The current learning-rate schedule may not be optimal for this dataset.
* Learning-rate reduction, rather than epoch count, may be the primary factor influencing convergence.
* The number of epochs alone does not appear to be the root cause of the observed performance limitations.

---

# Results on RADCURE

**Total cases:** 74

The comparison between the different retraining strategies on the RADCURE dataset shows very limited variation in segmentation performance.

The average Dice scores obtained were:

| Experiment | Description                                   | Mean Dice |
| ---------- | --------------------------------------------- | --------- |
| Test 1     | Retrain using RADCURE only                    | 0.377     |
| Test 2     | Retrain using RADCURE + HECKTOR (1000 epochs) | 0.383     |
| Test 3     | Retrain using RADCURE + HECKTOR (700 epochs)  | 0.383     |

## Relationship Between 1000 and 700 Epoch Models

![RADCURE Linear Regression](images/radcure_dataset_lineal.png)

The main observations are:

* Dice scores remain largely unchanged across all experiments.
* No significant improvement is observed when comparing the 700-epoch retraining against the 1000-epoch retraining.
* Case-by-case results are highly similar.

However, the relationship between both experiments is not perfectly linear, suggesting that the models are learning slightly different representations despite producing similar overall Dice scores.

This behaviour supports the hypothesis that the current model may have reached a performance bottleneck. Additional training epochs do not appear to provide further gains, indicating that limitations are more likely related to dataset quality, annotation variability, or intrinsic information available in the images rather than optimisation time.

---

# Results on HECKTOR

**Total cases:** 48

The HECKTOR dataset shows substantially different behaviour.

The average Dice scores obtained were:

| Experiment | Description                                   | Mean Dice |
| ---------- | --------------------------------------------- | --------- |
| Test 1     | Retrain using RADCURE only                    | 0.330     |
| Test 2     | Retrain using RADCURE + HECKTOR (1000 epochs) | 0.486     |
| Test 3     | Retrain using RADCURE + HECKTOR (700 epochs)  | 0.545     |

The inclusion of HECKTOR cases during retraining increased the average Dice score from **0.33 to 0.49**, representing an improvement of approximately **47%**.

Reducing the number of epochs from 1000 to 700 further increased performance from **0.49 to 0.55**, corresponding to an additional **12.15% improvement**.

## Relationship Between 1000 and 700 Epoch Models

![HECKTOR Linear Regression](images/hecktor_dataset_lineal.png)

The regression analysis between both configurations yields an R² value of approximately **0.59**, indicating a moderate positive relationship between the two models.

Several cases appear above the expected equality line, indicating that the 700-epoch model achieves higher Dice scores than the 1000-epoch model for a significant number of patients.

This behaviour suggests that reducing the training duration helps prevent overfitting on HECKTOR cases while preserving the knowledge transferred from RADCURE.

---

# Discussion

The experiments indicate that including HECKTOR cases during retraining is the main factor responsible for the observed performance improvement.

The increase from Test 1 to Test 2 demonstrates that the original model trained exclusively on RADCURE does not generalise well to HECKTOR patients. This suggests a dataset dependency and a domain shift between both cohorts.

The additional improvement obtained by reducing training from 1000 to 700 epochs supports the hypothesis that the longer training schedule introduces overfitting. While the effect is negligible on RADCURE, it becomes visible on HECKTOR, where the model benefits from earlier stopping.

Nevertheless, the persistence of similar change points in both training curves indicates that optimisation dynamics, particularly the learning-rate schedule, may still be limiting performance.

The fact that RADCURE performance remains stable while HECKTOR performance improves suggests that the model retains the knowledge learned from RADCURE while becoming better adapted to the HECKTOR domain.

---

# Conclusion

Including HECKTOR cases during retraining substantially improves segmentation performance on HECKTOR patients, increasing the average Dice score from 0.33 to 0.55.

Reducing the training duration from 1000 to 700 epochs further improves HECKTOR performance by approximately 12%, suggesting a reduction in overfitting effects.

In contrast, RADCURE performance remains largely unchanged across experiments, indicating that the model has likely reached a performance plateau on that dataset.

Overall, the results suggest that:

* Dataset composition has a greater impact on performance than the total number of training epochs.
* The model exhibits limited cross-dataset generalisation.
* Training beyond 700 epochs does not provide additional benefits for HECKTOR and may increase overfitting.
* Future improvements are more likely to come from data quality enhancements, domain adaptation strategies, and optimisation of the learning-rate schedule rather than increasing training duration.

---

