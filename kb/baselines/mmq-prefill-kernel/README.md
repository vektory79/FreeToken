# mmq-prefill-kernel/ — базлайны MMQ префилла (v0/v2)

Содержимое: step0-сплит первого чанка ([step0_run.json](step0_run.json),
[step0_split.json](step0_split.json)), A/B результаты v0 и v2
([v0-ab-results.json](v0-ab-results.json), [v2-ab-results.json](v2-ab-results.json)),
пробы живости ([v2ab_probe.json](v2ab_probe.json),
[v2ab_probe_liveness.json](v2ab_probe_liveness.json)) и quality-этап
([divergence.json](quality/divergence.json), env0/env1 stage-суммари).

Якорные числа: v2 grouped MMQ +16.4% @8128 токенов префилла (GLM-5.3-Flash-UD-Q3_K_XL,
winner-флаги, RTX 5090); v0 (сортировка пар экспертов) — выигрыша нет.
Методики и вердикты — в [../../methods/v2-ab.md](../../methods/v2-ab.md) и
[../../methods/quality-ab.md](../../methods/quality-ab.md).

## Дистилляция дубликатов

Три сырых дампа удалены как дубликаты машиночитаемых итогов:

- `v0ab_probe.out` — дубль сводки [v0-ab-results.json](v0-ab-results.json);
- `v2ab_probe_analyze2.out` — дубль [v2-ab-results.json](v2-ab-results.json);
- `quality/analyze_output.txt` — дубль [quality/divergence.json](quality/divergence.json).

Полные оригиналы этих выводов не сохранялись: их значимая часть входит в
перечисленные JSON.