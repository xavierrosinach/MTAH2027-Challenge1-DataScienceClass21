# prova-xavier: reglas nuevas sobre `josep/mejoras`

Fecha: 13-nov-2026. Configuración: `xavier/prova-xavier.yaml`. `config.yaml` no cambia de comportamiento
(sigue dando el CSV de `josep/mejoras`, SHA256 `c53b0b1b…`): todo lo nuevo está detrás de opciones
apagadas ahí y encendidas en `xavier/prova-xavier.yaml`. El código de las reglas está en
`src/possession/context.py`; esta carpeta tiene la configuración, las herramientas y los resultados.

## Qué hay en esta carpeta

```
prova-xavier.yaml   la configuración con las reglas nuevas encendidas
diff_frames.py      una imagen por frame donde un CSV no coincide con el GT público (zoom al balón)
diag.py             diagnóstico por frame (evidencias, control, balón) de un tramo: diag.py 11340 11480
stats.py            estadísticas en el banco: huecos de balón, duración de paradas y posesiones
stats2.py           estadísticas en el banco: dónde cae el cambio de equipo dentro de un hueco
mkcfg.py            genera las variantes de config (una por regla) para el banco o para BAR-ATH
bench/              las cuatro condiciones del banco (suite_*.yml) y el resumen de los resultados finales (resultados/*.log)
```

## Método

1. Mirar frame a frame los fallos en los 5 minutos públicos (`xavier/diff_frames.py`) **solo para entender
   la causa**, nunca para elegir parámetros (RULES.md).
2. Formular la causa como regla general de fútbol o de tracking.
3. Medir la premisa con estadísticas en otros partidos (banco de Metrica) cuando se puede.
4. Decidir en el banco (`validation/bench.py`, 3 partidos x 4 semillas, 4 condiciones), con
   diferencias emparejadas por partido y semilla.
5. Comprobar en BAR-ATH sin ground truth (`validation/check_bar.py`).

El banco se ha corrido con `control_v5dfl_*` (entrenado solo con IDSSE) en lugar de v6, porque las
copias leave-one-out de v6 necesitan los datos IDSSE, que no están en este equipo. Mide bien la
diferencia entre con y sin reglas; los valores absolutos no son los de v6.

## Dónde se perdía nota

En los 5 minutos públicos el 63 % de los frames fallados eran paradas (GT `D`) que el modelo daba
como juego vivo. El etiquetador solo tenía razones "duras" para parar cuando el balón salía o había
un tag `SET PIECES`; el Smart Tagging casi no etiqueta faltas ni fueras de juego.

En el banco (`validation/errors.py`): con paradas perfectas la base llegaría a 0,642 (+0,058);
esta versión ya está en 0,621, así que en paradas queda del orden de +0,02. Con equipo perfecto, +0,27 (posesiones cortas de menos de 2 s, que con
este tracking no se distinguen: en los dos casos de los 5 minutos la evidencia es casi nula).

## Reglas aceptadas (encendidas en `xavier/prova-xavier.yaml`)

| Regla | Opción | Razonamiento | Banco (spellF1, condición degradada) |
|---|---|---|--:|
| Hueco que es parada | `gap_restart` | Tras una falta o un fuera de juego el balón se saca desde el sitio: reaparece donde se perdió. En juego vivo reaparece lejos. Estadística en el banco: huecos de >= 3 s con el balón a < 10 m son parada el 76-100 % de las veces; a > 10 m en < 5 s, el 0-40 % | 6 m: +0,019 (12/12); 10 m: +0,010 más (12/12) |
| Parada mínima de 3 s | `transition_cost.min_dead_s: 3` | Con `team_switch` 16, el camino equipo -> D -> otro equipo cuesta 15 y sale más barato que un cambio directo: aparecen paradas de 1 s. En el banco solo el 3,3 % de las paradas reales duran menos de 3 s (0,4 % menos de 1 s); el modelo daba 9 de 113 | +0,004 (11/12) |
| Saque desde la línea | `ball_dead.kick_from_line` | Córners y saques de banda empiezan sobre la línea: el balón va rápido pero a menos de 1 m de ella y no contaba como golpeo, así que la parada no se soltaba | +0,002 (10/12) |
| Saque falso | `restart_check` | Reanudar exige juego. Un golpeo no reanuda si el balón vuelve a pararse en < 6 s sin que lo toque un segundo jugador (el portero que lo coloca, devolverlo al sitio). Y parada -> un solo jugador -> balón solo otra vez sigue siendo parada | +0,0001 (neutro) |
| Balón falso | `ball_dead.isolated_max_player_speed: 1.6` | Balón quieto y solo mientras todos corren es un error del tracking. Umbral entre el p75 de frames parados (1,51 m/s) y el p25 de frames vivos (1,68 m/s) de todo el partido | +0,0006 (neutro) |
| Jugador solapado | `occlusion` | El tracking pierde a un jugador tapado por un rival; se mantiene como "fantasma" pegado a él | +0,002 (7/12) |

Las tres "neutras" no se ven en el banco porque en Metrica el balón desaparece durante las paradas y
la degradación no simula solapamientos. Se dejan encendidas: no empeoran y corrigen fallos reales
vistos en BAR-ATH.

## Resultado del conjunto (banco, spellF1)

| Condición | base | prova-xavier | diferencia | ejecuciones mejores |
|---|--:|--:|--:|--:|
| degraded | 0,584 | 0,621 | +0,037 | 12/12 |
| light | 0,644 | 0,688 | +0,045 | 12/12 |
| degraded+tags | 0,609 | 0,627 | +0,018 | 11/12 |
| degraded+coarseGT | 0,652 | 0,695 | +0,043 | 12/12 |

macroF1 y F1 vivo/muerto suben en las cuatro condiciones y en los tres partidos.

BAR-ATH sin ground truth (`check_bar.py`): 267 posesiones (base 268), 114 paradas (106), balón
muerto 45,2 % (40,4 %), saques etiquetados con parada 96,6 % (igual). `check_rules.py`: PASS.

5 minutos públicos (solo comprobación): frames distintos del GT 905 -> 450; macroF1 0,850 -> 0,938;
spellF1 0,663 -> 0,710.

## Probado y descartado

| Idea | Resultado en el banco | Motivo |
|---|---|---|
| Adelantar la parada al momento en que el balón se queda solo (`isolated_backdate`) | −0,003 (0/12) | En otros partidos un balón solo suele ser un pase largo en juego |
| Hueco parado por velocidad neta del balón (`gap_restart.max_speed` 1 / 1,5 / 2 / 3 m/s) | +0,005 a +0,019 en spellF1, pero F1 vivo/muerto baja y el partido 3 no mejora | Gana troceando posesiones, no acertando paradas |
| Hueco parado hasta 13 m | +0,005 (9/12), partido 3 negativo | No es consistente |
| Hueco mínimo de 2 s | −0,0005 (0/12) | |
| Parada mínima de 4 s | 0,000 | Sin ganancia sobre 3 s |
| `to_dead` 10 -> 12 | +0,0004 | Ruido |
| Versión inicial de "tramo de un solo jugador" (sin exigir balón solo después) | −0,0015 | Mataba saques reales cortos |
| Regla "el rival no tiene la posesión hasta que controla" | no implementada | El GT tiene posesiones de 0,8 s y 1,3 s: sí cambia con toques breves |
| Mover el cambio de equipo dentro de un hueco | no implementada | En el banco el cambio real cae al final del hueco (mediana 1,0), que es lo que ya hace Viterbi |

## Fallos que quedan en los 5 minutos (450 frames)

- 11351-11479: despeje de cabeza del Barça; el tracking pone al Athletic más cerca del balón.
- 17380-17452: el balón se sigue jugando tras el pitido y la parada llega tarde.
- 15889-15943: la parada empieza 2 s tarde (el balón rueda solo antes de darse por abandonado).
- 15346 y 15766: posesiones del Barça de 0,8 s y 1,3 s sin evidencia en el tracking.
- El resto son bordes de 10-20 frames en cambios de posesión y reanudaciones.

## Reproducir

Desde la raíz del repo:

```
python run.py --config xavier/prova-xavier.yaml
python xavier/diff_frames.py --config xavier/prova-xavier.yaml --pred outputs/submissions/prova-xavier.csv
python validation/check_bar.py --configs josep=config.yaml prova-xavier=xavier/prova-xavier.yaml
python validation/check_rules.py
```

Banco (clonar antes `https://github.com/metrica-sports/sample-data` en `../external/metrica-sample-data`).
Una condición tarda unos 8 minutos:

```
python xavier/mkcfg.py outputs/bench_cfg bench final=falso+oclus+rcheck+gap+gap10+md3+kline
python validation/bench.py --suite --bench xavier/bench/suite_degraded.yml --no-baselines --configs base=outputs/bench_cfg/base.yaml final=outputs/bench_cfg/final.yaml --out outputs/bench_cfg/degraded.csv
```

En `bench/resultados/` el modelo `f2` es esta versión, `final` la anterior (sin 10 m, parada mínima
ni saque desde la línea) y `base` la de `josep/mejoras`.
