# findings/ — прочные технические карты

Карты кода и upstream-исследований с якорями file:line. Статус: `validated`
означает, что якоря перепроверены на указанном коммите; якоря дрейфуют —
перепроверяйте перед правками.

## Подкаталоги

- [gguf-glm5next-hybrid/](gguf-glm5next-hybrid/) — карта гибридной машинерии,
  исследование ggml CPU-ядер (IQ-форматы, целочисленный порт), карта w1 CPU
  GEMV, оценка переделки task-06.
  Статус: validated. Вход: [hybrid-machinery-mapping.md](gguf-glm5next-hybrid/hybrid-machinery-mapping.md);
  для работы с ядрами — [ggml-cpu-kernel-study.md](gguf-glm5next-hybrid/ggml-cpu-kernel-study.md).
- [gguf-glm5next-path-a/](gguf-glm5next-path-a/) — виды тензоров llama.cpp,
  фазовые конфиг-ключи, карты тензоров (включая mmproj), верификация якорей.
  Статус: validated. Вход: [llamacpp-tensor-kinds.md](gguf-glm5next-path-a/llamacpp-tensor-kinds.md).
- [llamacpp-glm5next-snapshot/](llamacpp-glm5next-snapshot/) — пронумерованный
  снапшот upstream-реализации GLM5NEXT: архитектура, QKV-создание, пул кэша,
  общий список тензоров. Статус: validated, но WARNING — upstream-ревизия НЕ
  записана; детали в [README.md](llamacpp-glm5next-snapshot/README.md).
- [mmq-prefill-kernel/](mmq-prefill-kernel/) — исследование upstream ggml:
  moe-align (r1) и tile-glue (r2), база v2 grouped MMQ.
  Статус: validated. Вход: [research-r2-upstream-tile-glue.md](mmq-prefill-kernel/research-r2-upstream-tile-glue.md).
- [fix1-radix/](fix1-radix/) — якоря кода бага mamba_last_track_seqlen при
  chunked prefill и карта тестов/харнесса.
  Статус: superseded по коду — баг исправлен коммитом fix-1 (см.
  [../incidents/README.md](../incidents/README.md)); карта сохранена как
  исторический якорь. Вход: [code-anchors.md](fix1-radix/code-anchors.md).
- mmq-v3/ — пустой подкаталог: материалы v3 живут в
  [../cases/mmq-v3-stationary-moe/](../cases/mmq-v3-stationary-moe/) и в методах
  [v3a-surface-map.md](../methods/v3a-surface-map.md); кандидаты на удаление в
  следующем триаже.

## Навигация

- Работаете над ядром gguf — начните с [gguf-glm5next-hybrid/](gguf-glm5next-hybrid/)
  и [mmq-prefill-kernel/](mmq-prefill-kernel/).
- Конвертируете или чините FTW/llama.cpp-совместимость —
  [gguf-glm5next-path-a/](gguf-glm5next-path-a/) и
  [llamacpp-glm5next-snapshot/](llamacpp-glm5next-snapshot/).
- Разбираетесь в radix-кэше — [fix1-radix/](fix1-radix/) плюс брифы
  [../cases/fix3-snapshot-lru-refresh/TASK.md](../cases/fix3-snapshot-lru-refresh/TASK.md).