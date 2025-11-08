 export QT_API=pyqt6

 mv logs/wake_success_*.wav models/orion_dataset/negatives/
 python tools/augment_orion_dataset.py
python tools/train_speechbrain_orion.py

python orion_image_generator.py   "a young woman inspired by Sakura Haruno from Naruto, pink short hair, red outfit, facing the camera, kunai held between her teeth, confident expression"   --from-img SakuraHaruno.webp --style semi --strength 0.7 --use-refiner   --width 896 --height 1152