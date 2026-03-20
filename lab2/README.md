Final solution contains ensemble of yolo26m and yolo11m models. 
Everything was trained purely on Kaggle platform. 
Yolo26m was trained on a base 3908  image dataset using 80/20 split for 50 epochs, then tuned for 30 epochs on expanded 5000 image dataset. 
Yolo11m was trained solely on expanded dataset for 100 epochs.
Ensembling implemented using WBF with weights 1, 1.5 for y11, y26 respecrively. 
