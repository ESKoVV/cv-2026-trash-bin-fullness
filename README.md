# Trash Bin AI

Итоговый проект курса [«Компьютерное зрение»](https://github.com/ykvnkm/computer-vision-course), 2026, группа Т.МО41.

**Команда:** Буланов Родион, Еськов Владимир.

**Задача:** по фото найти мусорный контейнер и определить степень его заполненности (`empty`, `partial`, `full`, `overflow`), чтобы планировать вывоз мусора (ДНР).

Проект готовит единый YOLO-датасет и Google Colab-обучение для поиска мусорного контейнера и классификации его заполненности: `empty`, `partial`, `full`, `overflow`.

## Состав

- `data/raw` — оригинальные загрузки Roboflow (не попадают в Git);
- `data/merged` — подготовленный YOLO-набор;
- `scripts/download_datasets.py` — скачивание через официальный Roboflow SDK;
- `scripts/prepare_dataset.py` — проверка, исключение проблемных данных, нормализация и объединение;
- `scripts/validate_merged.py` — независимая проверка итоговой структуры;
- `reports` — лицензии, статистика, исключения и изображения с рамками;
- `notebooks/trash_bin_training_colab.ipynb` — обучение в Google Colab;
- `models` — место для локальных экспортов, если они понадобятся.

## Локальная подготовка данных

Локальное обучение не выполняется: GPU используется только в Colab.

1. Откройте терминал в корне `trash-bin-ai` и создайте окружение: `python3 -m venv .venv && source .venv/bin/activate`.
2. Установите зависимости: `pip install -r requirements-local.txt`.
3. Секретно передайте ключ только текущей оболочке: `read -rs ROBOFLOW_API_KEY; export ROBOFLOW_API_KEY`. Не вставляйте его в файл и не используйте `echo`.
4. Скачайте YOLO-версии: `python scripts/download_datasets.py`.
5. Подготовьте единый набор: `python scripts/prepare_dataset.py`.
6. Проверьте итог: `python scripts/validate_merged.py`.
7. Создайте и проверьте архив для Drive: `python scripts/package_dataset.py`. Будет создан `trash_bin_dataset.zip` с `data.yaml`, `images/` и `labels/` в корне.

После пункта 5 откройте `reports/dataset_report.md`, `reports/excluded_files.csv` и `reports/samples`. Не используйте набор для обучения, если в отчёте нет примеров хотя бы одного из четырёх классов или если `validate_merged.py` завершилась с ошибкой.

## Запуск в Google Colab

1. Загрузите [notebooks/trash_bin_training_colab.ipynb](notebooks/trash_bin_training_colab.ipynb) в Google Drive либо откройте [Google Colab](https://colab.research.google.com/), выберите **File → Upload notebook** и укажите этот файл.
2. В Colab выберите **Runtime → Change runtime type → T4 GPU**, затем **Save**. Первая ячейка должна показать GPU через `nvidia-smi`.
3. Загрузите созданный архив в `My Drive/trash-bin-ai/trash_bin_dataset.zip` либо включите `UPLOAD_FROM_COMPUTER = True` в notebook и выберите его с компьютера. Альтернатива: скопируйте содержимое `data/merged` в `My Drive/trash-bin-ai/dataset_merged`.
4. Запускайте ячейки сверху вниз. Notebook проверит структуру, загрузит `yolo26n.pt`, если она доступна для установленной Ultralytics, а иначе автоматически перейдёт на поддерживаемую `yolo11n.pt`. Это будет явно напечатано в выводе ячейки.
5. Обучение использует transfer learning, изображение 640×640, до 60 эпох и early stopping (`patience=12`). После него notebook построит графики и confusion matrix, проверит тестовый split и покажет предсказания.
6. Лучшие веса окажутся в `My Drive/trash-bin-ai/exports/best.pt`. В эту же папку notebook сохранит экспорты ONNX и NCNN для Raspberry Pi.

## Своя фотография

В последней ячейке notebook раскомментируйте строки, загрузите изображение в диалоге Colab и запустите ячейку. Будут показаны рамки и один из четырёх классов. Для практического применения всегда дополнительно проверяйте модель на данных с реальных камер и условиях освещения целевого объекта.

## Лицензии и атрибуция

Оба первичных набора опубликованы как CC BY 4.0. Их авторы, ссылки и условия приведены в [reports/dataset_licenses.md](reports/dataset_licenses.md); при распространении производных данных или модели сохраните атрибуцию и отметьте изменения.

## Кто что делал

> Будет заполнено к защите.
