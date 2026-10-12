---
name: "call-agent-payload-truncation"
description: "Oversized call_code_agent payload fails 'Field task required'; write spec to .tasks file, delegate compact call"
type: project
lastUpdated: 2026-10-11T18:56
---

# call_code_agent: слишком большой payload теряет поле task (2026-10-10, дважды за сессию)

Вызов call_code_agent с большим combined context+task падает с "failed to parse
arguments for tool 'call_code_agent': Field 'task' is required ... but it was missing
at path: $". Поле, скорее всего, обрезается при генерации вызова - текст ошибки сам
подсказывает ("it was likely truncated while this call was generated - split the work
into smaller calls"). Случилось на волнах: (1) kb-запись вердикта W0.5, (2) спека
stage-0 n-gram зонда.

How to apply: если промт агенту раздувается (полная спека + все числа + вердикт) -
сначала task_write_file спеки в .tasks/<рабочую папку>/, затем компактный
call_code_agent вида "прочитай файл X и исполни". Не ретраить тот же большой вызов
без изменений - упадёт так же.

