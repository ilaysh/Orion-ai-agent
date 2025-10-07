 export QT_API=pyqt6

 mv logs/wake_success_*.wav models/orion_dataset/negatives/
 python tools/augment_orion_dataset.py
python tools/train_speechbrain_orion.py