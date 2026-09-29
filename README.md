# Практика по основам нейронных сетей

Python 3.10+. Установка: `python -m pip install -r requirements.txt`.
Все команды выполняются из корня проекта.

1. `python -m nnlab.main` — полноценное меню создания, CSV, обучения, предсказания, сохранения/загрузки и графика.
2. `python -m nnlab.main --demo` — воспроизводимый пример с данными `data/input/students.csv`.
3. `python -m nnlab.prepare doctor` — окружение и subprocess-диагностика.
4. `python -m nnlab.prepare prepare --path data/input/images --ext png --output data/output/prepared.npy`.
5. `python -m nnlab.prepare prepare --path data/input/images --ext png --size 32 --output data/output/prepared32.npy`.
6. `python -m nnlab.parallel --path data/input/images --workers 2 --output data/output/performance_report.json`.
7. `python -m unittest discover -s tests -v` — численный градиент, загрузка, сохранение и завершение при ошибке.

Данные синтетические, созданы для проверки программы. Для MLP последняя колонка CSV — бинарная цель. При предсказании через @file.csv файл содержит заголовок и только признаки.
Масштабирование обучается только на train, сохраняется вместе с моделью и применяется к новым объектам. BCE и её градиент соответствуют друг другу; выходные узлы — независимые sigmoid.

CLI prepare объединяет изображения одинакового размера. CSV готовятся отдельно: ожидаются только признаки, совместимое число колонок, заголовок; применяется min–max. Для train/test используйте DatasetManager, а не нормализацию всего датасета в prepare.
Поддержаны DATA_PATH, OUTPUT_PATH, MODEL_PATH, LOG_LEVEL. CLI перекрывает ENV. `.env.example` показывает имена; файл автоматически не загружается.

Параллельная часть: producer → ограниченная очередь путей → reader threads → ограниченная очередь массивов → Pool.apply_async с callback. Счётчик защищён Lock. Варианты: оригинал, поворот, отражение, шум; seed зависит от номера файла. Результаты сортируются, чтобы сравнение с последовательным запуском было точным.
На маленькой выборке процессы могут работать медленнее. Это измеряется, а не скрывается. Итоговый объединённый массив остаётся в памяти; для больших данных следует писать блоки через memmap и не накапливать весь результат.
