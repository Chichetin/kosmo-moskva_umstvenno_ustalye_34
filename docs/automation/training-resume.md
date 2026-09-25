# Продолжение автопилота обучения после новой сессии, /clear или сжатия контекста

Вставить в новую сессию Claude Code (каталог `/home/m1hairu/Documents/final_cosmohack`):

```text
Продолжи автопилот обучения FloodValue.

Прочитай docs/automation/training-autopilot.md и configs/training-autopilot.yaml
(ветка worktree-autopilot, .claude/worktrees/autopilot), затем
/home/m1hairu/Documents/final_cosmohack/autopilot/HANDOFF.md и state.json.

Не начинай работу заново. Сначала проверь реально работающие процессы (PID из state.json,
`ps -p`), сохранённые результаты в loeo_results/sweep/, текущий Git (`git fetch`, SHA main)
и остаток бюджета.

Восстанови последний незавершённый шаг. Не дублируй обучение, не обнуляй лимиты
и не заменяй принятую модель непроверенной. Test не читать.

Проверь наличие расписания (/loop). Если его нет — запусти одно; не создавай дубликат.

Выполни следующий безопасный шаг и обнови постоянное состояние.
```
