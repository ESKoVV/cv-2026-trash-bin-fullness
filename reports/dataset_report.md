# Отчёт о подготовке датасета

## Статус

**Готов к обучению.**

## Источники

- `data/raw/garbage-can-overflow` — классы: Broken trash can, Close_empty, Close_full, Healthy trash can, Open_empty, Open_full, Trash flow, closed, empty, full
- `data/raw/waste-bin-fill-level-detect2` — классы: empty, full, half-full, overflowing

## Итоговые данные

| Split | Изображений |
|---|---:|
| train | 627 |
| val | 134 |
| test | 135 |

| Класс | Рамок |
|---|---:|
| empty | 768 |
| partial | 117 |
| full | 634 |
| overflow | 69 |

Изображений-примеров с рамками: **12**.

## Исключения и проверки

Всего исключено файлов: **1547**.

| Причина | Количество |
|---|---:|
| ambiguous_class | 1092 |
| bbox_outside_image | 51 |
| empty_label | 391 |
| manual_visual_review | 13 |

Проверяются: отсутствие image/label, повреждённые изображения, пустые и некорректные labels, class id, диапазон и границы YOLO bbox, неоднозначные классы, SHA-256-дубликаты и пересечение split. Дубликат сохраняется только один раз, поэтому один и тот же байтовый файл не может попасть в разные split.

Неоднозначные классы (`healthy trash can`, `broken trash can`, `closed`, `emptyfull`) и любые неизвестные классы исключаются целиком вместе с изображением; подробности находятся в `excluded_files.csv`.

## Оценка пригодности

Во всех трёх split и всех четырёх целевых классах есть данные. Набор пригоден для первого обучения после успешного запуска `python scripts/validate_merged.py`; класс `partial` заметно меньше остальных и требует контроля метрик.
