---
name: "ft-merge-round8-lessons"
description: "Round-8 merge lessons: checkout --theirs trap again, uv venv 3.14, constexpr splice, timings main-wins pattern"
type: project
lastUpdated: 2026-11-09T11:29
lastRecall: 2026-11-09T12:33
---

# Merge round 8 lessons (fe975da, split-round)

Merge upstream/main 8215a94 -> vektory79: fe975da (parents 5a170e8+8215a94), базовый ledger - merge-main-into-vektory79-skill; состояние и pending - ft-merge-round8-pending-handoff.

Уроки для скилла/kb:
1. Ловушка checkout --theirs ВОСПРОИЗВЕЛАСЬ второй раз (round 6 и round 8): checkout --theirs снёс веточные авто-слитые фичи на 8 файлах, поймано постфактум и восстановлено git checkout -m. Вывод: резолв по ханкам (git checkout -m / ручной union) должен быть ДЕФОЛТОМ с самого начала, а не восстановлением после --theirs.
2. .venv может отсутствовать в CPU-окне после смены окружения: создавать uv venv --python 3.14 явно; дефолт uv тянет cp315, для которого нет wheel torchvision.
3. Сплайс constexpr-тройки (triton/attention.py: HAS_KV_SCALE/KV_NVFP4/KV_OFFSET_DTYPE на 4 сигнатурах и 4 launch-сайтах) прошёл чисто - правило "сигнатура-прелюдия-call-sites как единое целое" работает, применять всегда.
4. Main-wins timings-цепочки (round 8) - образец no-dual-logic: веточный wire-блок снят как дубликат main ReqStats.timings+timings_block, при этом веточные внутренности (RequestTimingTracker, own-window forward_ms/warmup, stream-stall warnings fc5fd4a на openai_api.py:274-438) живы как потребители main-механики.
5. Сплит-раунд-протокол (CPU 1653 passed / boot-smoke + GPU pending) отработал штатно третий раз (rounds 6-8); 6 фейлов первого CPU-прогона оттриажены и исправлены внутри merge-коммита.

kb-distill и SKILL.md-фолд round-8 инсайтов - после закрытия cloud-окна.

6. kb-distill round 8 DONE: kb/cases/merge-round8-upstream-main/TASK.md + раунд 8 в kb/topics/merge-wave-procedure.md; Wave 6 фолд DONE: SKILL.md 316->325 (ханк-резолв дефолт, uv venv --python 3.14). Файлы незакоммичены - коммит за пользователем.
7. Новый baseline-класс рига: tests/models/dflash/test_draft.py bitwise window-restore assert (line 216, [4/8-dtype0]) = main-baseline с 08cee44 (DFlash-2 #857); A/B против MAIN-родителя (8215a94) подтверждает; bf16-параметр зелёный. Upstream-тест, ни разу не гонявшийся на риге = отдельный baseline-класс.
