# Конвертация датасета v2

Исходные `data/merged` и все прежние материалы не изменялись.

## Изображения и labels

| Split | Изображений | Label-файлов |
|---|---:|---:|
| train | 627 | 627 |
| val | 134 | 134 |
| test | 135 | 135 |

## Классы

| Старый класс | Рамок | Новый класс | Рамок после автоматического преобразования |
|---|---:|---|---:|
| empty | 768 | not_full | 885 |
| partial | 117 | not_full | 885 |
| full | 634 | full | 634 |
| overflow | 69 | overflow | 69 |
| — | — | unknown | 0 (не назначался автоматически) |

## Ручная проверка

Программно просмотрено изображений: **896**; рамок: **1588**.
В ручную проверку отправлено объектов: **1170**; отрисовано фрагментов: **1170**.

| Причина | Объектов |
|---|---:|
| closed_container_contents_not_visible | 568 |
| container_blurry_or_low_contrast | 110 |
| container_overexposed | 3 |
| container_too_dark | 4 |
| container_too_distant_or_small | 23 |
| full_or_overflow_requires_visual_fill_level_confirmation | 703 |

`unknown` указан только как предложение в `review.csv`; labels класса 3 не создавались автоматически. Проверяющий должен вручную проверить весь кадр на пропущенные контейнеры и мусор рядом, который не должен менять класс контейнера.

## Техническая проверка

PASSED: image-label пары, отсутствие лишних/пустых labels, YOLO-формат, class id 0–3, bbox в [0,1], положительные размеры, SHA-256 дубликаты между split и точные повторяющиеся bbox.

## Созданные файлы

- `data/merged_v2/data.yaml`
- `reports/v2_review/review.csv`
- `reports/v2_review/contact_sheets`
- `reports/v2_review/items`
