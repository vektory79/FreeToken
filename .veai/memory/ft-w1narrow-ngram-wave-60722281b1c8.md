---
name: "ft-w1narrow-ngram-wave"
description: "W1-narrow CLOSED: A/B NO-GO on target profile (ngram acceptance 0); stage0 method flaw; scaffold uncommitted"
type: project
lastUpdated: 2026-10-12T14:04
lastRecall: 2026-10-12T17:11
---

# Волна W1-narrow: ЗАВЕРШЕНА - A/B NO-GO на целевом профиле (2026-10-10), каркас остаётся

Финал: парное окно (64k fill radix-reuse, temp 0, 768 tok, hybrid якорь): off2 18.409
tok/s, k1 18.379 (-0.16%), k2 18.712 (+1.65%) - шум пары +-1.7%; acceptance = 0,
0 verify-шагов в ON-руках -> ON==OFF по построению. Гейт +12-15% - NO-GO на целевом
профиле кейса. Без регрессии (overhead ниже шума).

Why (главный урок): Stage-0 офлайн-метод мерил самопересечение потока filler (93%), а не
согласие модели с n-грамм-таблицей: живой greedy-декод продолжения не воспроизводит
шаблон (биграммы вне таблицы). Самопересечение текста - НЕ валидный прокси acceptance;
перед drafting-инфраструктурой мерить model-vs-table agreement на живом декоде. Высокий
radix-reuse ПРЕФИКСА не означает повторяемость ВЫВОДА.

How to apply: каркас (n-gram drafter + verify + GDN-откат через LinearStatePool.copy_from,
метрики, флаги --spec-* default OFF) собран, отревьюен (2 MAJOR fixed: кормление драфтера
на plain decode; non-hybrid reject счётчики), бутится на GLM-5.3-Flash (снят третий
молчаливый гейт _maybe_spec_verify_batch при page_size!=1; латентный KV пиннит 64 -
leak-until-finish на reject, wholesale free на finish). Код НЕ закоммичен (14 файлов
~625+ строк; варианты: коммит default-OFF / откат - решение пользователя). Окупится
только на повторяемом ВЫВОДЕ; для temp 1.0 нужен W3. Кавэт kpool-ring персистентной
деградации selection невалидирован (reject'ов не было).

Артефакты: .tasks/glm53-spec-decode-w1/ (DESIGN.md, ab_harness.md, AB_RESULTS.md,
W1NARROW_KB_ADDENDUM.md, run_ab.py, логи). Полный набор падений pytest идентичен
до/после волны (все предсуществующие/environment; счётчик плавает 8..19 от инвокации).
