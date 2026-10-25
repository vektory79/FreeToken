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
| [merge-wave-procedure.md](merge-wave-procedure.md) | Регламент слияния origin/main и upstream/main в ветку: волны, правила конфликтов, резолв целевого remote, продолжение прерванной волны; ловушки main-wins раунда 6 - потерянный веточный фикс register.py, осиротелый floor gate, замеренные счётчики |
| [round5-ggml-replatform-deferred-ports.md](round5-ggml-replatform-deferred-ports.md) | Что main пере-платформировал в GGUF-стеке в раунде 5 (vendored ggml, слот-пулы, deps) и карта порт-долга D1-D5 ветки со статусами |
| [glm5next-gguf-loader-port.md](glm5next-gguf-loader-port.md) | Как порт D4 (коммит 9314c9b) вернул glm5next GGUF-сервинг на вендоренном ggml: дублирующийся ключ схемы, GPU-касты под op_dequantize, pad_rows MMQ-хвоста, GGUF->FTW стриминг, corr-методика кросс-валидации |
| [hybrid-moe-slot-pools-port.md](hybrid-moe-slot-pools-port.md) | Как порт D3 (коммит 393b682) закрылся тремя гейтами конфига поверх слот-пулов main: format screen, header-отказ hybrid, auto floor gate; T2 не возвращён; гибрид 27.71 tok/s против offload 21.00 (+32%) |
| [round5-d1-kernel-wins-no-go.md](round5-d1-kernel-wins-no-go.md) | Почему порт D1 (grouped MMQ, m-tile, dense N-split, mhc NS) закрыт NO-GO на вендоренном ggml: 241-572 GB/s эффективных при 169-227 TF/s = ALU-крыша; SPLIT=128 оптимален; N-split -1.8..-16.5%; пять уроков о непереносимости старых процентов |
| [round5-d2-mmproj-vision-port.md](round5-d2-mmproj-vision-port.md) | Как порт D2 (коммит 7bc68b0) вернул mmproj-зрение: ампутация ре-платформы точечная (3 стыка), collision-гвардия _pack_kv, verify 347/347; эскалация - gguf->FTW carrier не несёт тензоры (register.py:469) |
| [round5-d5-nvfp4-kv-generic.md](round5-d5-nvfp4-kv-generic.md) | Как порт D5 (коммит 660c814) вернул nvfp4/fp8-KV read-путь generic-triton: store/read-асимметрия ре-платформы, E2M1-parity A/B против dequant-референса (Qwen3.8, 3.46x без деградации), dtype-гейт валидатора формата |
| [decode-chain-microfusion.md](decode-chain-microfusion.md) | Почему сокращение пусков ядер не равно ускорению; четыре микрофьюжн-кандидата декода отклонены на железе 2026-09-28; дискриминатор - CTAs/SM, а не счётчик пусков |
| [ftw-legacy-fold-zero-scratch.md](ftw-legacy-fold-zero-scratch.md) | Почему легаси FTW-носитель ловил пик 198 GiB (fold double-buffer против read-пула) и как zero-scratch affine remap (f12e554) опустил пик до пола рецепта 142.5 GiB; фьюз по сумме группы процессов |
| [gguf-vram-headroom-post-round5.md](gguf-vram-headroom-post-round5.md) | Почему рецепт сентября (0.82/400k/8191) перестал держать пик VRAM после re-platform раунда 5; новые внеплановые потребители; матрица 0.80-0.82 x чанк, лестница глубины, семантика L2-капа session tier |
| [gguf-serving-perf-matrix.md](gguf-serving-perf-matrix.md) | Сводная матрица производительности текущего стека: префилл/декод/VRAM по memory-ratio x чанк (5 конфигов), лестницы глубины до 378k, slot-floor границы, рецепты выбора конфига |

Правила статей: YAML-шапка (title/date/hardware/branch-commits/status/tags),
<=~250 строк, все перекрёстные ссылки — markdown-ссылки на существующие
файлы kb, числа всегда со своей конфигурацией.