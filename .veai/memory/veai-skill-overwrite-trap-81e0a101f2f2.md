---
name: "veai-skill-overwrite-trap"
description: Orchestrator write_file creates .veai/skills SKILL.md but cannot overwrite; delegate edits to call_code_agent
type: project
lastUpdated: 2026-09-16T19:29
lastRecall: 2026-09-21T15:51
---

# write_file не перезаписывает существующие .veai/skills файлы

Наблюдение (2026-09-15, обновление merge-main-into-vektory79/SKILL.md после review-1): оркестраторский write_file СОЗДАЁТ новый файл в .veai/skills (allow_overwrite=false, успех), но повторная запись существующего файла падает с warning "Virtual file is not accessible for writing" - и с allow_overwrite=true тоже.

**Why:** ограничение виртуальной FS харнесса для .veai; после создания файла инструменты оркестратора (write_file) править его не могут.

**How to apply:** сопровождение/доработка skill-файлов - делегировать точечные правки call_code_agent (у него есть edit_file) с явными до/после формулировками правок; НЕ повторять идентичный падающий write-вызов. Создание НОВЫХ skill-файлов через write_file работает.
