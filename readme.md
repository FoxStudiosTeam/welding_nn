##### косая черта после названия ( / ) означает, что это папка
##### звездочка перед названием файла/папки ( * ) означает, что файл/папка скрыта
#### `/`
корневая директория
`difference.py` - наглядно демонстрирует обученность нейросети путем сравнения оригинальных изображений, размеченных людьми и результатом работы нейросети

`get_results.py` - генерирует csv файл "ответов"

`requiments.txt` - зависимости

`weights.pt` - веса дообученной yolo8

#### `*datasets/`
директория для датасетов
`evaluate/` - изображения и лейблы для проверки

`train/` - изображения и лейблы для тренировки

`evaluate.cache` - кэш для ускорения загрузки

`train.cache` - кэш для ускорения загрузки

#### `*runs/`
директория для результатов тренировок

#### `telegram_bot/`
директория, содержащая код для телеграмм бота

`tg_bot.py` - основной файл для запуска бота

`*telegram_token.py` - токен бота

#### `train/`
директория, содержащая код для тренировки нейросети

`compare_train_results.py` - сравнение результатов тренировки в виде графиков

`data.yml` - информация о датасете

`split_dataset.py` - разделение датасета на тренировочный и проверочный

`train.py` -  тренеровка

`yolo8n.pt` - стартовые веса yolo8