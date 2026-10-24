---
title: "Честный замер VRAM-пика ft serve: параллельный сэмплер, unique-nonce промпты, stdin-curl, kill по /proc/PID/exe"
date: 2026-10-24
hardware: "RTX 5090 32607 MiB"
commits: "vektory79 @ ddefb5e; метод применён и на стеке тега 2026.10.17 (b8c3794)"
status: "verified (замеры железа)"
tags: [vram, peak-free, nvidia-smi, unique-nonce, radix, curl, pgrep, методика]
---

# Честный замер VRAM-пика ft serve

Метрика - **peak-free** = 32607 MiB (номинал карты) минус максимум used за
весь прогон; второй якорь - free-before-load на буте. Сравнивать прогоны
можно только при одинаковом desktop-фоне (волна 2026-10-24: 796 MiB).
Четыре ловушки делают наивный замер ложным: одиночный сэмпл пропускает пик,
повторный промпт фейкит cache-хит, большое тело не лезет в argv, а убийство
сервера по pgrep бьёт по самому харнессу.

## 1. Параллельный 2 с сэмплер - одиночные сэмплы пропускают пик

Пик транзиентов живёт внутри префилл-чанка (единицы секунд): nvidia-smi
после ответа его уже не видит. Сэмплер - отдельный фоновый процесс с
cadence 2 с; фазовые теги пишет управляющий скрипт:

```bash
# фоновый сэмплер: CSV ts,phase,used_mib; PHASE меняет управляющий скрипт
while :; do
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk -v ts="$(date +%s)" -v ph="$PHASE" '{print ts "," ph "," $1}' >> vram.csv
  sleep 2
done &
```

Отчётное число - минимум free по фазе, не "последний сэмпл".

## 2. Unique-nonce промпты - иначе пик прячется за фальшивым cache-хитом

Session-tier restore и radix-матч честно возвращают кэш для повторов.
Инцидент волны 2026-10-24: "свежий" 8k-промпт без nonce вернул
cached_tokens=6592 - префилл почти весь ушёл в хиты, и пик остался на фазе
мелкого чата; замер показал 2+ GiB свободных там, где честный прогон держал
641 MiB. Правило: каждый измерительный промпт несёт уникальный nonce
(случайный суффикс), делающий префикс гарантированно новым.

## 3. Лестница глубины - растущий radix-префикс

Заполнить пул глубоко без повторного префилла всего с нуля: следующий
промпт = предыдущий плюс новый блок (~22k токенов). Radix отдаёт хиты по
наросшему префиксу, свежий блок честно префиллится, prompt_tokens растёт до
целевой глубины; минимум free фиксируется на каждом шаге. Так получена
лестница до 378.6k токенов (94.6% пула 400384) - числа в
[../topics/gguf-vram-headroom-post-round5.md](../topics/gguf-vram-headroom-post-round5.md).

## 4. Тело запроса - через stdin, не argv

JSON с 8k+ токенами (>128 КиБ) в аргументе curl умирает с "Argument list
too long" - лимит ядра MAX_ARG_STRLEN 128 КиБ на один аргумент:

```bash
# тело генерируем в stdout и кормим curl через stdin
python3 gen_prompt.py 8000 nonce-42 \
  | curl -s http://127.0.0.1:18801/v1/chat/completions \
      -H 'Content-Type: application/json' -d @- > resp.json
```

## 5. Остановка сервера - по /proc/PID/exe, не по pgrep -f

`pgrep -f 'ft serve'` матчит и командную строку самого харнесса (self-match):
один прогон может убить собственный скрипт вместо сервера. Дискриминатор -
бинарник процесса, а не cmdline:

```bash
for p in $(pgrep -f 'ft serve'); do
  exe=$(readlink -f "/proc/$p/exe" 2>/dev/null) || continue
  case "$exe" in */.venv/bin/python*) kill "$p";; esac
done
```

## Протокол прогона

1. Бут с целевым конфигом; зафиксировать free-before-load, план (слоты /
   страницы), "Free memory after initialization" и free после CUDA graphs.
2. Стартовать сэмплер (п.1), прогнать unique-nonce префилл целевой глубины
   (пп.2 и 4), записать peak-free по фазам.
3. Для лестницы глубины - шаги растущего префикса (п.3) с минимумом free
   на каждом шаге.
4. Остановить сервер по /proc/PID/exe (п.5) и убедиться, что VRAM вернулась
   к desktop-фону, - иначе следующий бут стартует с чужой ношей.
5. Сравнивать прогоны только при одинаковом desktop-фоне; машиночитаемый
   близнец - JSON с конфигом каждой точки (образец -
   [../baselines/dense-q80-gemm/vram-headroom-results.json](../baselines/dense-q80-gemm/vram-headroom-results.json)).

## Откуда метод

Обобщение сентябрьской волны VRAM-headroom
([vram-headroom.md](vram-headroom.md), харнесс
[../harness/ab-runner/vram_headroom_run.py](../harness/ab-runner/vram_headroom_run.py))
и готчей волны 2026-10-24: инцидент cached_tokens, MAX_ARG_STRLEN, self-match
pgrep. Матрица и разбор пост-round5-стека -
[../topics/gguf-vram-headroom-post-round5.md](../topics/gguf-vram-headroom-post-round5.md);
термины - [../glossary.md](../glossary.md).
