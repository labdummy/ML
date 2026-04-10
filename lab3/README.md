First iteration: deepMindV3 + efficientnetb4 model trained for 49 epochs (7+14+28 cosine annealing restarts) using multi-scale [328, 344, 352, 368] and medium augmentation pipeline. Then used ep20 and 39-49ep SMA models as teachers and carefully trained a student on a full dataset (no val split) results: .903 .904 teachers -> .908 student.
Used the final model to pseudo-labels 350 images, attempting to fine-tune / retrain on expanded dataset was only detrimental to the metrics.

Second iteration: SegFormer-B2 with MAnet neck trained on a expanded dataset with a similar pipeline. .910 dice. 

Attempting to ensemle / tta only worsen the metrics. 
