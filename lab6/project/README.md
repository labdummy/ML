Модель предсказывает принадлежность последовательности из 8 изображений к одному из 3 классов (inaction, move, work)
Структура модели: DINOv2 (эмбеддинги) + TCN + Transformer + ансамбль 5 фолдов
На вход подается папка с 8 изображениями

Применение:
Скачать и распаковать веса моделей (https://drive.google.com/drive/folders/1oVyLxgofpMeG_NsxVZM0GTqkpWg6bpY4) в ./models

Сборка образа: 
docker build -t project .

Запуск:
docker run \
    -v /path/to/images:/images \
    -v /path/to/models:/models \
    project \
    python inference.py \
    --images /images \
    --models /models
    
Пример выхода:
Result: inaction (confidence=0.9929)
