---
name: "chat-text-capture-gotchas"
description: "ft serve chat harness: stream text needs delta.reasoning_content; multiturn needs history; budget >=160"
type: project
lastUpdated: 2026-11-15T18:22
---

# Текстовый захват в chat-харнессе ft serve (2026-11-15)

- STREAM: только delta.content у reasoning-модели даёт ПУСТОЙ сохранённый текст -
  reasoning идёт в delta.reasoning_content; для вердицируемых/сравниваемых текстов
  копировать ОБА поля (text = content + "\n" + reasoning). Для подсчёта tok/s это
  было известно, для захвата текстов - нет.
- Мульти-turn: ft serve stateless per request; одиночные user-сообщения теряют
  историю (turn2 ответил "no prior context... first message"). Накапливать
  messages array; ассистентский ход = RAW message.content (пустой -> конкат
  reasoning_content).
- Цепочки ходов: бюджет >=160 токенов; 96 режет даже односложный ответ
  (finish_reason=length прячет ответ).

Why: два смоука route-affinity гейта (smk000/smk001) отделили баги харнесса от
реальных промахов модели.
How to apply: любой чат-харнесс с сохранением/оценкой ответов на этом риге.
