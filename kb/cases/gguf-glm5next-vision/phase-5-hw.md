---
title: "Фаза 5 mmproj-зрения GGUF glm5next: HW-волна e2e (boot + image-запрос)"
date: 2026-10-06
hardware: "RTX 5090 32GB, iommu=pt, VRAM свободна (облачная LLM); GLM-5.3-Flash-GGUF UD-Q3_K_XL + mmproj-BF16"
commits: "волна без кода: прогоны на HEAD d8dee13 (фазы 1-4 - e35577d..2b91236)"
status: "фаза 5 PASS; кейс закрыт полностью, см. [TASK.md](TASK.md)"
tags: [gguf, mmproj, vision, glm5next, hw-wave, e2e, image, ftw, nvfp4]
---

# Фаза 5: HW-волна e2e mmproj-зрения

Бриф - [TASK.md](TASK.md); CPU-фазы - [phases-1-2.md](phases-1-2.md) и
[phases-3-4.md](phases-3-4.md). Прогон 2026-10-06 на HEAD d8dee13 (vektory79):
серийно, один сервер за раз, trap+watchdog, SIGTERM->SIGKILL между прогонами.
Все прогоны PASS.

## Прогон 0: конвертация FTW-vision

`ft checkpoint --model GLM-5.3-Flash-UD-Q3_K_XL.gguf --mmproj mmproj-BF16.gguf
--out GLM-5.3-Flash-UD-Q3_K_XL-FTW-vision`: 336.5 s, 135.29 GiB в 18 шардах
(1355 weight + 126 experts_bank). Верификация: ровно 347 тензоров `visual.*`
в шардах и 15 `clip.*` KVs в source_metadata.gguf - PASS.

## Таблица прогонов

| run | путь | boot до serving | TTFT | decode tok/s | image-e2e |
|-----|------|-----------------|------|--------------|-----------|
| 1 | raw .gguf + --mmproj | 167 s | 4.70 s | 19.62 | OK |
| 2 | GGUF-FTW-vision, без флага | 47 s | 3.22 s | 20.29 | OK |
| 3 | NVFP4-референс | 64 s | 4.21 s | 11.64 | OK |
| 4 | raw .gguf text-only, без флага | 105 s | - | - | n/a |

Общая GLM-конфигурация прогонов 1-2: --moe-cache-auto --kv-reserve-tokens
70016 --memory-ratio 0.85 --max-prefill-length 4096 --moe-strategy hybrid
--moe-cpu-threads 16 --max-running-requests 1 (run1 + --mmproj); NVFP4 -
только --max-prefill-length 4096 --max-running-requests 1.

## Ответы на тестовую картинку

Картинка 448x448: красный фон, белая жирная надпись "FREE TOKEN 42" по центру.
Ответы содержательно совпали на всех трёх путях (после think-блока):

- run1 (raw + --mmproj): "На картинке написано FREE TOKEN 42 белым жирным
  шрифтом. Фон - красный."
- run2 (FTW-vision): аналогично, "Фон картинки - красный."
- run3 (NVFP4): "На картинке написано FREE TOKEN 42 белыми буквами.
  Фон - красный."

## Критерии приёмки

- (а) ответы run1/run2 совпадают с NVFP4-референсом: PASS
- (б) GGUF-FTW-vision бут без --mmproj, image-запрос работает: PASS
  (лог: "Multimodal enabled: Glm5NextMMProcessor, encoders ['vision'] on host")
- (в) text-only boot без флага не изменился: PASS (105 s в историческом
  коридоре 94-132 s, без "Multimodal enabled", текстовый completion OK)
- (г) VRAM возвращается между прогонами: PASS (1028-1042 MiB до/после каждого
  прогона - уровень десктопа; после - нет ft-процессов)

## Аномалии (наблюдения, не блокеры)

1. enable_thinking=false не убирает think-блок на всех путях, включая NVFP4
   (рассуждение в content, reasoning_content пуст). gguf/шаблонная особенность
   GLM, к vision отношения не имеет.
2. usage отсутствует в стриминговых ответах; в non-stream probe есть.
3. boot run1 167 s против 105 s text-only: визуальная башня грузится на host,
   ожидаемое удорожание.

## Указатели

- Артефакты прогона в .tasks/gguf-glm5next-vision/ (вне git, здесь дистиллят):
  сценарии hw-phase5-scenarios.md, отчёт hw-phase5-report.md, результаты
  hw-phase5-results/ (серверные логи, probe/image-json, VRAM до/после),
  клиент/раннер/верификация/генератор картинки.
- Новый чекпойнт: GLM-5.3-Flash-UD-Q3_K_XL-FTW-vision в каталоге unsloth
  UD-Q3_K_XL, 136 G; старый текст-only FTW не тронут.
