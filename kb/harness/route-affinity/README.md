# route-affinity/ - харнесс гейта Maya-механики route affinity

Байт-в-байт копия скриптов кейса
[../../cases/route-affinity/TASK.md](../../cases/route-affinity/TASK.md)
(план гейта и пороги) и [../../cases/route-affinity/VERDICT.md](../../cases/route-affinity/VERDICT.md)
(результаты и вердикт). Живой каталог - `.tasks/route-affinity/`
(arm_*.json, serve-логи, judge/ - git-ignored). Код рычага - env-гейт
`FREETOKEN_ROUTE_AFFINITY` в `models/glm5_next/moe.py` +
`kernel/triton/moe_router.py` (бонус в selection-скор `ranked`, веса
неизменны; маска резидентности = строки LRU offload-кеша).

## Состав

- `run_arm.py` (gate-v2) - одна рука: бут прод-конфига (графы), fill 64k
  (session-tier restore), radix-reuse greedy декод 768 (ignore_eos),
  hit-строки строго по окну декода, затем батарея качества на том же буте:
  6 арифметика (авто-чек) + 4 проза (набор первого замера), батарея 24
  промптов кампании GGUF из `kb/cases/route-affinity/battery24.json`
  (4 с проверяемым ответом), NIAH-проба 20 игл на глубинах 40-60k
  (глубина калибруется по prompt_tokens филла; порядок от глубоких к
  мелким - radix переиспользует префикс), мульти-turn проба 5 ходов
  (цепочка арифметики с историей сообщений). Все тексты пишутся целиком.
  `--smoke` - укороченный прогон для валидации пайплайна (~7 мин).
- `make_judge_sheets.py` - слепые пары base vs X по всем текстам
  (56 пар на значение); детерминированная перетасовка A/B, ключ в
  отдельном файле; `--resample 10` - лист пересуда со сторонами наоборот.
- `score_judge.py` - раскрывает вердикты, считает долю "не хуже базы"
  (tie + X-лучше; гейт >= 85%) и строгую долю (tie = базе), пересуд на
  самосогласованность.
- `aggregate_gate.py` - медианы tok/s/p50/hit/C/NIAH по значениям,
  порог скорости vs базы (>= +10%).
- `compare_arms.py` - старый дифф текстов против базы (первый замер).

## Полный гейт 2026-11-15 (13 бутов)

| value | boots | median tok/s | vs base | hit % | NIAH | суд "не хуже базы" |
|---|---|---|---|---|---|---|
| 0 | 2 | 21.35 | - | 35.9 | 40/40 | - |
| 0.05 | 2 | 26.88 | +25.9% | 49.7 | 40/40 | 94.6% (r1) |
| 0.1 | 2 | 30.76 | +44.1% | 60.1 | 39/40* | 92.0% (r1+r2) |
| 0.15 | 2 | 34.59 | +62.0% | 68.3 | 40/40 | 96.4% (r1) |
| 0.2 | 2 | 38.66 | +81.1% | 74.0 | 40/40 | 92.0% (r1+r2) |
| 0.3 | 3 | 45.29 | +112.1% | 81.1 | 60/60 | 92.9% (r1+r2+r3) |

\* один промах = обрез бюджета (finish=length, игла найдена, ответ оборван
перед числом); в повторном буте 20/20. Полные таблицы и вердикт - в
[../../cases/route-affinity/VERDICT.md](../../cases/route-affinity/VERDICT.md).

## Запуск

```bash
# рука (~15-18 мин, GPU, сериально; батарея на том же буте):
python .tasks/route-affinity/run_arm.py --affinity 0.3 --tag aff030_b1
# слепые пары + суд (CPU):
python .tasks/route-affinity/make_judge_sheets.py --base aff000_b1 --x aff030_b1 --label X030r1 --resample 10
python .tasks/route-affinity/score_judge.py --label X030r1 --recheck
# гейт-таблица (CPU):
python .tasks/route-affinity/aggregate_gate.py aff000_b1 ... --out gate_table.md
```

Готчи: прод-конфиг зашит в BASE_ARGS (чекпоинт FTW-vision-new, 0.81/800k,
tier 10/50, порт 18801) - на другой машине поправить; GLM reasoning-парсер
кладёт работу в `reasoning_content` - батарея сравнивает content +
reasoning_content вместе, арифметике бюджет >= 96 (двоичный перевод в 96
не влезает - постоянный FAIL всех рук, встроенный контроль), NIAH-игле
бюджет 160 и отдельная строка (в смежной строке повторяющегося филлера
модель её теряла из-за обреза рассуждения); история мульти-turn собирается
из `content_raw` (пустой у reasoning-парсера - фолбэк на склейку, артефакт
одинаков для всех рук); `--moe-collect-stats` обязателен; hit-парсинг
только до конца декода (батарея не должна загрязнять).
