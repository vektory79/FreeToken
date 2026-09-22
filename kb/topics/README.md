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
| [session-cache-tiering.md](session-cache-tiering.md) | Как выгружать сегменты сессий в RAM/SSD буферы фиксированного размера; экономика восстановления (дизайн-анализ) |
| [moe-hybrid-cost-model.md](moe-hybrid-cost-model.md) | Из чего складывается шаг декода гибридного MoE; пределы флаговых рычагов |
| [ftw-load-path.md](ftw-load-path.md) | Как GGUF превращается в FTW и почему бут падает с 94-132 с до 52 с; контракт silent-None |
| [gguf-upstream-map.md](gguf-upstream-map.md) | Карта upstream llama.cpp GLM5NEXT для GGUF-портов; что проверять перед новым портом |
| [merge-wave-procedure.md](merge-wave-procedure.md) | Регламент слияния origin/main в ветку: волны, правила конфликтов, продолжение прерванной волны |

Правила статей: YAML-шапка (title/date/hardware/branch-commits/status/tags),
<=~250 строк, все перекрёстные ссылки — markdown-ссылки на существующие
файлы kb, числа всегда со своей конфигурацией.