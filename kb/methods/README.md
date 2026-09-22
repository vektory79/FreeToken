# methods/ — методики измерений и ведения кампаний

Каждый документ учит одному методу: как измерить, как сравнить, как не
испортить прогон. Перед запуском собственной волны прочитайте
[orchestration-notes.md](orchestration-notes.md), затем карточку нужного метода.

| Документ | Чему учит | Когда читать |
|---|---|---|
| [orchestration-notes.md](orchestration-notes.md) | Протокол волн: сериализация прогонов, STOP GATE, дайджест ловушек | Перед любой новой кампанией |
| [step0-profile.md](step0-profile.md) | Step-0 разбор первого префилл-чанка: GEMM/копии/прочее, режим BW или ALU | Прежде чем оптимизировать ядро: «есть ли вообще что выигрывать» |
| [step0-profile-mmq.md](step0-profile-mmq.md) | Тот же метод на инстансе MMQ префилла | Повторить разбор для другого ядра |
| [step0-source-read.md](step0-source-read.md) | Проверка выводов step-0 чтением исходников ядра (moe_q, IQ-близнецы) | Когда профиль противоречит ожиданиям |
| [ab-mtile.md](ab-mtile.md) | A/B свип по m-tile: одна переменная на boot, nsys steady-span | Шаблон A/B для kernel-knob |
| [ab-nsplit.md](ab-nsplit.md) | A/B волна NSPLIT 1 vs 2: сериализация, watchdog, машинночитаемый близнец | Шаблон A/B для двух значений knob |
| [quality-ab.md](quality-ab.md) | Батарея качеств A/B: greedy-промпты, вербатим+sha, разбор дивергенции | Когда скорость может менять выводы токенов |
| [v0-ab.md](v0-ab.md) | Hardware A/B v0 (сортировка пар экспертов): отчёт без выигрыша — тоже результат | Как оформить нулевой вердикт |
| [v2-ab.md](v2-ab.md) | Hardware A/B v2 grouped MMQ: лестница точек префилла | Шаблон A/B с лестницей размеров чанка |
| [v3a-ab.md](v3a-ab.md) | Hardware A/B свип v3a m-tile 4/8/16/32 | Шаблон мульти-точечного свипа |
| [v3a-surface-map.md](v3a-surface-map.md) | Карта поверхности реализации v3: кто x, кто y в moe.cuh, ловушки m-tile | Перед правками ядра MoE |
| [arbitration-battery.md](arbitration-battery.md) | Арбитражная батарея: бут-недетерминизм против эффекта конфигурации | Когда A/B «мигает» между boot'ами |
| [vram-headroom.md](vram-headroom.md) | Матрица memory-ratio / kv-reserve / max-prefill: сколько FREE VRAM покупает каждый knob и какой ценой | При OOM/падениях префилла и выборе конфига serving |
| [wave1-notes.md](wave1-notes.md) | Волна 1: N-split kda_in_proj, все изменения анкоммитнуты до валидации | Как организовать волну без коммитов |
| [wave2-candidate-a.md](wave2-candidate-a.md) | Волна 2: Candidate A, компилят-тайм варианты тайла + knob | Как подготовить вариант к измерению |

Типовые вопросы и входные точки:

- «Почему падает префилл / не хватает VRAM» — сначала
  [vram-headroom.md](vram-headroom.md), затем [../incidents/README.md](../incidents/README.md).
- «Как поставить A/B» — [ab-nsplit.md](ab-nsplit.md) как минимальный шаблон,
  [ab-mtile.md](ab-mtile.md) как шаблон свипа, гочвы прогонов в [../README.md](../README.md).
- «Изменилось ли качество» — [quality-ab.md](quality-ab.md), а при мигающих
  результатах — [arbitration-battery.md](arbitration-battery.md).

Статусы: все методы `validated` на RTX 5090, если в шапке документа не указано
иное; superseded-цепочек нет. Машинночитаемые близнецы результатов живут в
[../baselines/README.md](../baselines/README.md); скрипты, которыми методы
выполнялись, — в [../harness/README.md](../harness/README.md).