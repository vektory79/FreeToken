# topics/ — атомарные статьи по темам

Вики-слой kb: одна статья отвечает на один конкретный вопрос сессии —
вердикт в первых строках, затем механизм, затем числа с конфигурациями и
указатели «как измерить». Детали и артефакты — в связанных файлах других
секций; терминология — [glossary.md](../glossary.md).

| Статья | Вопрос, на который отвечает |
|---|---|
| [prefill-throughput.md](prefill-throughput.md) | Чем задаётся скорость префилла; текущие якоря GGUF/NVFP4; ловушки измерения чанков |
| [vram-budget.md](vram-budget.md) | Сколько VRAM нужно рецепту; почему reserve/moe-cache-size не освобождают VRAM; меню рецептов |
| [radix-cache-reuse.md](radix-cache-reuse.md) | Почему длинные промпты не переиспользовались и что починено (fix-1, fix-3) |
| [interleave-cache-wash.md](interleave-cache-wash.md) | Почему чередование диалогов вымывает GDN-снапшоты; железные руки Arm 1/Arm 2; чем закрыто флагом |
| [session-cache-tiering.md](session-cache-tiering.md) | Как выгружать сегменты сессий в RAM/SSD буферы фиксированного размера; фазы 1-2 подтверждены железом; кодеки QSA/KpoolDSA, KV-only tier, async prefetch; supersede вложенных сегментов - no-op на гибридном flush; параллельный pwrite-флаш L2 + групповой fdatasync + CRC32C (волна P3); two-phase сжатие тир-блоба (P9); экономика восстановления; ограничение restore при дивергенции истории - телеметрия ok/refused и вердикт P0 (preamble - re-prefill оправдан, mid-history move/вопрос - P1/P2 обязательны); фикс P1+P2 принят железом (контрольный ход после мутации HIT 89,088 ток / 26.7 с, refusals 0; коммиты 577d79c/197d029); P3 - алиасинг chain[0] у чатов общей преамбулы - закрыт симуляцией 2026-10-07 (возвраты в A восстановлены tier-restore, группирование/retention не требуются); fast-follow из 7 улучшений закрыт волнами 2026-10-08 (W2/W3/W4 закоммичены 1298048/c1d45c1/c2e9dbc; W1 стейджинг NO-GO; chain0 entropy won't-fix); дедупликация L2 по экстентам - shutdown-флаш 27B 8.17 GiB против якоря 23.81 GiB (-65.7%), HIT/целостность/латентность без регрессий; снапшот-онли триггер на admission - кейс NO-GO 2026-10-09 по премисе: на шве back-to-back ходов дерево холодное (владеет 5 из 1392 страниц), restore честно читает весь сегмент |
| [ft-serve-shutdown-signals.md](ft-serve-shutdown-signals.md) | Почему SIGTERM группы процессов от оркестратора убивал graceful-остановку ft serve; SIGTERM-обработчик воркеров спасает session-tier flush; проверка железом и настройки llama-swap |
| [moe-hybrid-cost-model.md](moe-hybrid-cost-model.md) | Из чего складывается шаг декода гибридного MoE; пределы флаговых рычагов |
| [ftw-load-path.md](ftw-load-path.md) | Как GGUF превращается в FTW и почему бут падает с 94-132 с до 52 с; контракт silent-None |
| [gguf-upstream-map.md](gguf-upstream-map.md) | Карта upstream llama.cpp GLM5NEXT для GGUF-портов; что проверять перед новым портом |
| [merge-wave-procedure.md](merge-wave-procedure.md) | Регламент слияния origin/main и upstream/main в ветку: волны, правила конфликтов, резолв целевого remote, продолжение прерванной волны |
| [decode-chain-microfusion.md](decode-chain-microfusion.md) | Почему сокращение пусков ядер не равно ускорению; четыре микрофьюжн-кандидата декода отклонены на железе 2026-09-28; дискриминатор - CTAs/SM, а не счётчик пусков |

Правила статей: YAML-шапка (title/date/hardware/branch-commits/status/tags),
<=~250 строк, все перекрёстные ссылки — markdown-ссылки на существующие
файлы kb, числа всегда со своей конфигурацией.