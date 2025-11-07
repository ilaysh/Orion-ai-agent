 export QT_API=pyqt6

 mv logs/wake_success_*.wav models/orion_dataset/negatives/
 python tools/augment_orion_dataset.py
python tools/train_speechbrain_orion.py

Continue Orion v2 . Repo: https://github.com/ilaysh/Orion-ai-agent Keep same roadmap and Orion goals from previous chat. look at README.md for reference if needed.