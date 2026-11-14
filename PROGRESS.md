# Progreso

Panel para Josep. Se actualiza después de cada experimento. El detalle está en
[EXPERIMENTS.md](EXPERIMENTS.md) y el documento técnico en
[docs/technical_report.md](docs/technical_report.md).

**Actualizado:** 2026-11-14 18:23 (resumen de congelación)

**Calendario:**

- **Congelación de Orca:** 14-nov 20:00.
- **Revisión con Fable:** 14-nov por la noche y 15-nov.
- **Decisión con Xavier:** 16-nov.
- **Entrega:** 17-nov.

## Mejor versión (`josep/mejoras`)

| | banco spellF1 | banco macroF1 |
|---|--:|--:|
| **actual: + hueco que es parada a 10 m (#27, idea de Xavier)** (13-nov) | **+0.027 / +0.029** ⁴ | **+0.021** ⁴ |
| + saque desde la línea (#23, idea de Xavier) (13-nov) | +0.0023 / +0.0032 ³ | +0.0008 ³ |
| v6 + evidencia de equipo en los huecos (`gap_weight` 0.1) (13-nov) | 0.636 ² | 0.857 ² |
| v6 = modelo entrenado con 10 partidos (Metrica + IDSSE) (13-nov) | 0.629 ¹ | 0.856 ¹ |
| v4 + `dead_weight` 0.3 (13-nov) | 0.619 | 0.855 |
| v4 (modelo de control aprendido + movimiento + red + peso por posesión) | 0.613 | 0.852 |
| inicio del 12-nov (modelo de control v1) | 0.583 | 0.847 |
| main de Xavier | 0.455 | 0.812 |
| baseline oficial `baseline_smooth50` | 0.319 | 0.769 |

¹ Con semillas nuevas (4-7), en las que la versión anterior saca 0.616: ganancia emparejada
+0.013 (error típico 0.002). En el banco de 10 partidos (Metrica + IDSSE), +0.019.
² Semillas 12-15, donde v6 saca 0.634: +0.0019 (en las 8-11, +0.0011). Ganancia pequeña pero
constante.
³ Ganancia emparejada sobre la versión anterior: semillas 0-3 (0.6322 -> 0.6345) y 4-7.
⁴ Semillas 8-11 y 12-15 (confirmación). En el banco IDSSE con el balón escondido en tiempo muerto,
+0.024. En BAR-ATH: 130 paradas (antes 106), justo en el tope.

Mejora aproximada (las semillas no son las mismas): **+38 %** sobre main y **+97 %** sobre el
baseline oficial (banco de Metrica).

Ojo: el banco no es la nota real. Mide sobre otros 3 partidos con el tracking degradado como el
nuestro, y sirve para comparar versiones entre sí. La nota real solo se sabrá con los resultados
oficiales. CSV actual: SHA256 `567f0d71…`, 281 posesiones (media 12,4 s), 130 paradas, 45,4 % muerto.

## CONGELACIÓN (14-nov 20:00): resumen final del coordinador

- **Versión recomendada para entregar:** `josep/mejoras`, código y config de 9ada683. Lo que
  viene después es solo documentación.
  - CSV SHA256 `567f0d710f4d605f00489e474b068390edce119120b7bba29f8098b6f16f0aae`.
  - H 52362 / A 34398 / D 72097 / X 40462.
  - Regenerado dos veces con el mismo hash, y una vez desde un worktree limpio. `check_rules`
    PASS.
- **Qué lleva:**
  - el etiquetador de contexto con Viterbi;
  - el modelo de control v6 (red + lineal, entrenado con 3 partidos de Metrica y 7 de IDSSE);
  - `dead_weight` 0.3;
  - la evidencia de equipo en los huecos (0.1);
  - dos reglas de Xavier, validadas en el banco: el saque desde la línea y el hueco que es parada
    a 10 m.
- **Banco de Metrica:**
  - media de las 4 condiciones: la versión actual saca **0.662** con semillas 8-11 y **0.667** con
    semillas 12-15 (medido en la confirmación de #27);
  - condición degradada, semillas 0-3 (`errors.py`): 0.623, cuando v4 sacaba 0.573 y la primera
    versión con modelo aprendido, 0.540;
  - main de Xavier sacaba 0.455 (media de las 4 condiciones, semillas 0-3). Las semillas son
    distintas, así que la comparación es aproximada.
- **En nuestro partido** (sin etiquetas): 281 posesiones, 130 paradas (justo en el tope) y 45,4 %
  de tiempo muerto.
- **Experimentos del 12 y 13 de noviembre:** 21 cerrados (#8-#28), cada uno con su regla escrita
  antes de medir y criticado por Codex (consultas 01-23 en `C:\consultas-ia`). 6 aceptados y 15
  descartados, todos con su etiqueta en origin.
- **Pendiente para Josep y Xavier (16-nov):**
  - Revisar la nota de transparencia del documento técnico (§2.4) sobre el origen de las reglas
    de Xavier.
  - Pedir a Xavier que borre `xavier/README.md` de la rama pública: describe tramos del partido
    y su nota pública.
  - Decidir las líneas sobre el asistente de IA y Codex del documento técnico, y añadir los 4
    nombres.
  - Decidir qué pasa a `main` (R6: lo hacéis vosotros, no el coordinador).
  - Exportar el documento técnico como `DataScienceClass21.pdf` o `.md`, y entregar el CSV y el
    repo antes del 17-nov.
- **Riesgos conocidos:**
  - 130 paradas es más que en los partidos reales que tenemos (92-113);
  - las posesiones de menos de 2 s siguen sin verse;
  - el banco no es la nota real.

## Resumen a 13-nov 22:15 (para la decisión del 16)

- **Versión recomendada:** `josep/mejoras` 9ada683 y siguientes (después, solo documentación).
  - Es v6 + `gap_weight` 0.1 + dos reglas de Xavier: el saque desde la línea y el hueco que es
    parada.
  - CSV `567f0d71…`, reproducible desde un worktree limpio (comprobado a las 22:10).
- **Lo que ha aportado cada paso** (banco de Metrica; cada uno medido sobre la versión anterior,
  con semillas distintas; la nota real puede ser otra):

  | paso | banco |
  |---|---|
  | main de Xavier | 0.455 |
  | v4 (12-nov) | 0.613 |
  | `dead_weight` 0.3 | 0.619 |
  | v6, 10 partidos | 0.629 (semillas 4-7) |
  | huecos | 0.636 (semillas 12-15) |
  | saque desde la línea | +0.002 / +0.003 |
  | hueco que es parada | +0.027 / +0.029 (y +0.024 en IDSSE) |

  La versión actual da 0.623 en la condición degradada, semillas 0-3 (`errors.py`).
- **Experimentos del 12 y 13 de noviembre:** 21 cerrados con su regla escrita antes de medir
  (#8-#28). 6 aceptados (#9, #10, #16, #23, #27 y la herramienta IDSSE) y el resto descartados
  con etiqueta. Codex criticó cada cierre.
- **En nuestro partido** (sin etiquetas): 281 posesiones, 130 paradas y 45,4 % de tiempo muerto.
  Hay más paradas que en los partidos reales que tenemos (92-113), pero están dentro del tope
  fijado antes de medir.
- **Cola:** no queda ningún cambio de modelo con ganancia esperada razonable que se pueda validar
  antes de la congelación (consultas 13 y 23). Cada idea tiene su prueba o su motivo escrito.
- **Hasta la congelación** (14-nov 20:00):
  - vigilar el trabajo de Xavi y medirlo en el banco si lo sube;
  - mantener la documentación al día.

## En curso

| línea | rama | agente | estado |
|---|---|---|---|
| Vigilar el trabajo de Xavi (`origin/main`, `origin/xavi*/*`) | — | coordinador | revisión periódica (cada 1-2 h); última: 14-nov 17:21, sin novedades (`origin/main` 80ad34c, `origin/xavier/mejoras` 0b5782d). Sus 6 ideas, cerradas: 2 aceptadas y 6 descartes (con #21 a 15 m y #28) |

### Regla para la línea IDSSE (escrita el 12-nov, ANTES de medir v6)

Primera ronda de `exp/idsse`: v5all es v4 entrenado con 10 partidos (3 Metrica + 7 IDSSE).
- **Banco de Metrica:** +0.0029 (error típico 0.0022). light −0.002, 6/12 celdas: no pasa la
  regla de siempre. Por partido: el 3 sube +0.017; el 1 baja −0.005 y el 2 −0.004.
- **Banco DFL (7 partidos):** +0.034, 28/28 celdas.

El banco de Metrica favorece a los modelos entrenados solo con Metrica: la copia LOGO de v4
aprende de 2 partidos del mismo proveedor y formato que el partido evaluado, y BAR-ATH nunca
tendrá eso. El partido 3, el más distinto, ya se usaba como el mejor sustituto de un partido no
visto (EXPERIMENTS #5). El GT de IDSSE se construye como el oficial: estado y posesión del balón
del propio tracking.

Además, el worker encontró un fallo: el sentido de ataque se deduce de "media x del local < 0,5"
y falla en la 1.ª parte de un partido IDSSE. Comprobado sin etiquetas: en BAR-ATH y en Metrica
acierta en todas las partes. Se corrige comparando las medias de los dos equipos. En BAR-ATH no
cambia nada (hay que comprobar el mismo hash del CSV).

**v6 = v4 + 7 IDSSE con el sentido corregido.** Se acepta solo si se cumplen las 5 condiciones:
- (a) Banco de 10 partidos, semillas 0-3: mejora en las 4 condiciones y en la mayoría de las 40
  celdas, sin perder macroF1.
- (b) Banco de Metrica solo: la diferencia media no baja de 0 y el partido 3 tampoco.
- (c) Semillas 4-7: (a) y (b) mantienen el signo.
- (d) `check_control` en BAR-ATH: confianza y cambios de signo por minuto dentro de lo visto con
  v1-v4 (|llr| > 2 en el 76-82 % de los frames con balón; 6,4-7,7 cambios de signo por minuto).
- (e) `check_bar`: entre 240 y 300 posesiones (v4: 270), y paradas y set pieces con parada sin
  bajar.

**Resultado de v6 (13-nov, 02:00-03:00)**

| condición | resultado | ¿cumple? |
|---|---|---|
| (a) banco de 10 partidos, semillas 0-3 | +0.0177 (error típico 0.0010, 90 % de ejecuciones mejores); light +0.019, degraded +0.017, +tags +0.018, coarseGT +0.017; 36/40 celdas; macroF1 +0.0037 | sí |
| (b) banco de Metrica | +0.0094 (error típico 0.0022); las 4 condiciones suben (+0.008 a +0.011); 8/12 celdas; partido 3 +0.025, partido 1 +0.000, partido 2 +0.004 | sí (también pasa la regla de siempre) |
| (c) semillas 4-7 | Claude Code paró los dos bancos por falta de memoria (los lancé a la vez y uno era DFL) | **pendiente** |
| (d) `check_control` | BAR-ATH frente al banco: \|llr\| > 2 en el 72,1 % frente al 75,2 %; 7,13 frente a 7,16 cambios de signo por minuto | **no, tal como lo escribí** (72,1 < 76) |
| (e) `check_bar` | 272 posesiones (v4: 270); 104 paradas; 96,6 % de set pieces con parada; cambia el 1,4 % de los frames | sí |

Sobre (d): v6 es menos tajante que v4 en todas partes. En el banco baja del 79 % al 75 %, y en
BAR-ATH algo parecido. La relación entre BAR-ATH y el banco, que es lo que pretende medir el
control, es la misma que con v4. Pero el umbral que escribí antes de medir era absoluto y no se
cumple. No lo cambio por mi cuenta: decide Josep (ver "Para Josep").

Por el camino, con el sentido de ataque corregido, v4 sube en el banco DFL; antes se hundía en
WR9. Así que la ganancia de v6 en DFL es +0.021, no +0.034.

## Experimentos cerrados

Cada uno queda como etiqueta `aceptado/<grupo>/<idea>` o `descartado/<grupo>/<idea>`. En GitHub
están en "Tags", y en EXPERIMENTS.md, tabla "Every path explored".

| fecha | grupo | etiqueta | cambio | banco spellF1 (vs ref) | decisión |
|---|---|---|---|---|---|
| 11-nov | viterbi | `aceptado/viterbi/team-switch-16` | coste de cambio 20 -> 16 | 0.463 (+0.009) | aceptada |
| 12-nov | control | `aceptado/control/v1-control-aprendido` | modelo aprendido de quién controla | 0.583 (+0.119) | aceptada |
| 12-nov | control | `aceptado/control/v2-rasgos-movimiento` | rasgos de movimiento | 0.594 (+0.012) | aceptada |
| 12-nov | control | `aceptado/control/v3-red-promediada` | promedio con red antisimétrica | 0.605 (+0.010) | aceptada |
| 12-nov | control | `aceptado/control/v4-peso-posesion` | peso por duración de posesión | 0.613 (+0.008) | aceptada |
| 11/12-nov | todos | 13 etiquetas `descartado/...` | ver EXPERIMENTS.md | | descartadas |
| 12-nov | control | `descartado/control/semillas` | promedio de 5 redes con semillas distintas | 0.614 (+0.001, error típico 0.001) | descartada: pierde en light y +tags (ruido) |
| 13-nov | muerto | `aceptado/muerto/dead-weight-03` | peso del modelo de velocidad de balón muerto 0.5 -> 0.3 | 0.619 (+0.006; semillas 4-7 +0.009, 12/12) | aceptada |
| 13-nov | muerto | `descartado/muerto/to-dead-20` | coste de pasar a muerto 10 -> 20 | 0.616 (+0.003, 11/12) | descartada: en BAR-ATH las paradas bajan de 104 a 94 |
| 13-nov | reglas | (barridos, sin etiqueta) | `pass_llr`, `kick_team_llr`, `from_dead` | ±0.002 | sin efecto (EXPERIMENTS #9) |
| 13-nov | muerto | (barridos, sin etiqueta) | `dead_llr_clip`, `inside_llr`, `out_llr`, `isolated_llr`, `latch_llr`, `kick_llr` sobre la versión actual | ±0.001 | sin efecto; quitar `inside_llr` cuesta −0.020 (EXPERIMENTS #9) |
| 13-nov | datos | `aceptado/datos/v6-idsse` | modelo de control entrenado con 10 partidos (3 Metrica + 7 IDSSE) | 0.629 en semillas 4-7 (+0.013, error típico 0.002); 10 partidos +0.019 (39/40 celdas) | aceptada |
| 13-nov | control | `descartado/control/train-mix-light` | v6 entrenado también con copias `light` | +0.0027, pero coarseGT −0.0053 | descartada: corta más fino (+6 posesiones) |
| 13-nov | huecos | `descartado/huecos/interp-50` | interpolar huecos del balón hasta 1-2 s | +0.0005; con semillas 8-11, −0.0004 | descartada: ruido |
| 13-nov | muerto | `descartado/muerto/modelo-10-partidos` | modelo de balón muerto aprendido con 10 partidos, balón escondido en tiempo muerto | primer filtro: paradas en BAR-ATH 106 -> 82 | descartada; línea del muerto aprendido cerrada |
| 13-nov | huecos | `descartado/huecos/gap-weight-03` | evidencia de equipo en los huecos (`gap_weight` 0.3, modelo de v6) | +0.0050 (semillas 0-3), +0.0044 (4-7), 11-12/12 celdas | descartada por su regla: check_bar con 105 paradas, menos de 106 (umbral corregido solo para lo siguiente) |
| 13-nov | datos | `descartado/datos/v9-skillcorner` | modelo de control con 20 partidos (+10 SkillCorner) | −0.0063 (error típico 0.0015), 1/12 celdas, −6,7 posesiones | descartada: su GT grueso enseña a cortar más grueso |
| 13-nov | huecos | `aceptado/huecos/gap-weight-01` | evidencia de equipo en los huecos, peso 0.1 (confirmación con semillas nuevas) | 8-11 +0.0011, 12-15 +0.0019; check_bar 105 paradas (umbral ±2) | aceptada |
| 13-nov | control | `descartado/control/peso-v6` | peso de la evidencia de equipo 1.5 / 2 con v6 | +0.002 / +0.003, pero coarseGT −0.009 / −0.016 | descartada: corta más fino (5.º caso) |
| 13-nov | huecos | `descartado/huecos/forma` | peso de la forma en los huecos 0.05 / 0.2 | 0.2: +0.0010 dudoso; semillas 8-11 −0.0003 | descartada: ruido |
| 13-nov | control | `descartado/control/red-32` | red de 32 neuronas con 10 partidos | pérdida logarítmica de la red 0.2132 frente a 0.2120 | descartada en la regla 1, sin banco |
| 13-nov | huecos | `descartado/huecos/red-huecos` | red también para los frames sin balón | mejor pérdida logarítmica en huecos, pero +0.0001 en el banco (dudoso dos veces) | descartada |
| 13-nov | xavier | `descartado/xavier/parada-minima-3s` | parada mínima de 3 s (idea de Xavier) | +0.0045 (semillas 0-3), +0.0042 (4-7), 12/12 | descartada por su control: en BAR-ATH paradas 105 -> 100 y un saque de puerta sin parada |
| 13-nov | xavier | `aceptado/xavier/saque-linea` | un golpeo de un balón pegado a la línea reanuda (idea de Xavier) | +0.0023 (semillas 0-3), +0.0032 (4-7), 12/12 en las dos rondas | aceptada |
| 13-nov | xavier | `descartado/xavier/hueco-parada-15m` | hueco que es parada, radio 15 m (idea de Xavier) | 5 / 10 / 15 m: +0.013 / +0.027 / +0.032 (semillas 0-3, 12/12) | descartada: 136 paradas en BAR-ATH, tope 130. 10 m se confirma aparte (#27) |
| 13-nov | xavier | `aceptado/xavier/hueco-parada-10m` | hueco que es parada, radio 10 m (idea de Xavier, confirmación) | semillas 8-11 +0.027, 12-15 +0.029, IDSSE escondido +0.024; check_bar 130 paradas | **aceptada (actual)** |
| 13-nov | xavier | `descartado/xavier/parada-minima-sobre-huecos` | parada mínima de 3 s encima de #27 | +0.0057 (semillas 8-11, 12/12) | descartada: el mismo saque de puerta pierde su parada (96,6 -> 94,8 %) |
| 13-nov | xavier | `descartado/xavier/saque-falso` | saque falso (idea de Xavier) | +0.0005 / +0.0002 / −0.0000 (semillas 0-3, 4-7 y 8-11) | descartada: dudosa dos veces, casi nunca se activa |
| 13-nov | xavier | `descartado/xavier/fantasma` | jugador tapado como fantasma (idea de Xavier) | +0.0011 / −0.0006 / −0.0001 (semillas 0-3, 4-7 y 8-11) | descartada: dudosa dos veces; el banco no simula oclusiones |
| 13-nov | xavier | `descartado/xavier/balon-falso` | balón falso, 1.5 m/s (idea de Xavier, umbral nuestro) | +0.0010 / +0.0005 / +0.0005 | descartada: en BAR-ATH paradas 130 -> 125 y el mismo saque de puerta sin parada |

## Trabajo de Xavi

Se revisa en cada ciclo (`git fetch origin`). Su versión se mide en el banco cuando la suba; su
nota pública no cuenta para nada.

| fecha de la revisión | `origin/main` | ramas `origin/xavi/*` | novedades |
|---|---|---|---|
| 13-nov tarde | 80ad34c (11-nov, ya integrado en `josep/mejoras` el 12-nov) | ninguna | nada nuevo |
| 13-nov 17:11 | 80ad34c | ninguna | nada nuevo |
| 13-nov 18:34 | 80ad34c | **`origin/xavier/mejoras` 0b5782d** (13-nov 18:30, sobre nuestro af96572) | 6 reglas de balón parado detrás de opciones, en `xavier/prova-xavier.yaml`; se mide en el banco |

- **13-nov 18:30, `origin/xavier/mejoras` 0b5782d.** Su versión entera ("prova-xavier", sobre
  nuestro v6 sin `gap_weight`), medida en nuestro banco de Metrica contra `josep/mejoras`:
  - **semillas 0-3: +0.0326** (error típico 0.0022, 98 % de ejecuciones mejores), 12/12 celdas;
  - macroF1 +0.022;
  - por condición: light +0.046, degraded +0.033, +tags +0.015, coarseGT +0.038;
  - partido 3 +0.028;
  - `check_bar`: 267 posesiones, 114 paradas, 45,2 % muerto, set pieces con parada 96,6 %.

  Esto es solo informativo. Sus 6 ideas se prueban una a una (EXPERIMENTS #21-#26) en el orden
  que propuso Codex en la consulta 14: #22, #23, #21, #24, #26 y #25. Su código, con todo
  apagado, ya está en `josep/mejoras` (e6cd81b; mismo hash `1077ece7…` dos veces). La carpeta
  `xavier/` no se trae.
- **13-nov:** Josep cuenta que Xavi tiene un "0,7 de F1" sin subir. No sabemos sobre qué lo ha
  medido; probablemente sobre los 5 minutos públicos, que no deciden nada (R2). No cuenta hasta
  medirlo en el banco. Cuando lo suba:
  - se mide su versión entera en `C:\MTAH2027\wt-xavi`, con prioridad sobre la cola;
  - se separan sus ideas, cada una como un experimento de un solo cambio.

## Mapa de líneas (lo probado, el resultado y lo siguiente)

| línea | probado | resultado | siguiente |
|---|---|---|---|
| Modelo de control (quién tiene el balón) | v1-v4 (modelo aprendido, movimiento, red, peso por posesión); rasgos largos, de toques y de 0,5 s; la red sola; 5 semillas | 4 aceptados; el resto, ruido | #11 `train_mix` descartado (corta más fino); siguiente: evidencia mejor con el mismo corte (consulta 02-3, 01-3) |
| Datos de otros partidos | Metrica (3) e IDSSE (7) integrados | **v6 aceptado** (+0.013 en Metrica, +0.019 en 10 partidos); v9 con SkillCorner descartado (−0.006: su GT grueso enseña a cortar más grueso) | cerrada: no quedan datos abiertos con GT como el oficial y una granularidad parecida |
| Balón muerto | modelo aprendido (#7); `dead_weight`; `to_dead`, `from_dead`; 6 constantes de reglas | `dead_weight` 0.3 aceptado; `to_dead` 20 no transfiere; el resto, plano | #13 (aprendido con 10 partidos) también falla en BAR-ATH (106 -> 82 paradas): línea del muerto aprendido CERRADA. Solo transfiere el modelo ajustado en nuestro propio partido |
| Viterbi | `team_switch` 16 / 13 / 20; costes de muerto | 16 aceptado; el resto depende de la granularidad del GT | estado "disputado" (Codex 01), sin probar |
| Posesiones cortas (< 2 s) | peso por posesión 0.25 / 0.5 / 1; rasgos de toques y de 0,5 s | 0.25 aceptado; lo demás solo gana con GT fino | consulta 02 |
| Huecos sin balón | `gap_weight` (v3); interpolar 1-2 s (#12); `gap_weight` 0.3 con v6 (#14) | ruido; ruido; **señal real** (+0.004-0.005), pero descartada por 105 < 106 paradas | **#16 `gap_weight` 0.1 aceptado** (+0.001-0.002). Siguiente: #18, peso de la forma |
| Smart Tagging | evidencia débil de las fases de ataque en huecos (0.03) | sin medir en el banco (no hay sustituto de las fases) | motivo para no probarlo, o un sustituto desde eventos |

## Cola de ideas (en orden)

Ordenada por ganancia esperada frente a coste, con las consultas a Codex 01 y 02
(`C:/consultas-ia/consulta-NN-*.md`). Cada idea tiene una prueba o un motivo escrito.

1. **(hecho) #10 v6:** modelo entrenado con 10 partidos (Metrica + IDSSE). Aceptado el 13-nov.
2. **(hecho, descartado) #11 `train_mix`:** entrenar también con copias `light` (Codex 01-4). Corta
   más fino y pierde en coarseGT. Motivo para no probar más dosis ni degradaciones variadas (Codex
   02-1): solo moverían el mismo mando, el de cuán fino cortar.
2b. **(hecho, herramienta) Banco IDSSE con el balón escondido en tiempo muerto**
   (`validation/bench_dfl_dead.yaml`): ausencia del balón 49,8 % (BAR-ATH 40,7 %), huecos que
   empiezan en una línea 28,6 % (31,8 %). En
   BAR-ATH y en Metrica el balón casi nunca se ve con el juego parado; en IDSSE se ve siempre. Con
   esto, los cambios de balón muerto (oráculo +0.057 sobre v6) se podrían validar con dos
   proveedores. Sale de `errors.py` sobre v6.
3. **(hecho, descartado) #12 interpolar huecos más largos:** ruido; con semillas 8-11, −0.0004.
3b. **(siguiente) #14 evidencia de equipo en los huecos con el modelo de huecos de v6**
   (`gap_weight` 0.1 / 0.3; Codex 05, caminos 1 y 4). Barato, sin reentrenar.
3c. **Entrenar con ocultaciones del balón realistas (Codex 05-2).** Motivo para no probarlo: la
   degradación ya mete huecos con juego vivo (0,5 por minuto, unos 8 s), y el modelo de huecos de
   v6 se entrena con ellos. #14 dirá si esa evidencia sirve.
4. **Coste de cambio H↔A según la evidencia local de cambio** (Codex 02-3) o confianza según la
   calidad del tracking (Codex 01-3). Coste medio. El oráculo de cambios perfectos solo da ~+0.03,
   así que se espera poco.
5. **Decodificar los huecos largos por bloques**, con una sola hipótesis de posesión por hueco
   (Codex 02-4). Coste medio-alto; los huecos con juego vivo son el 10 % de los frames vivos.
6. **Estado "disputado" y corrector temporal aprendido** (Codex 01-6, 01-8). Coste alto: no da
   tiempo a cerrarlos antes de la congelación. Motivo para no probarlos.
7. **Modelo de secuencias de toques** para las posesiones < 2 s. Los rasgos de toques ya fallaron
   (#4), y dentro de las posesiones cortas perdidas la evidencia no ve nada (−0.1). Motivo para no
   probarlo: falta señal, no modelo.
8. **Smart Tagging como pista de equipo** (Codex 01-7). No hay un sustituto validable de las fases
   de ataque en el banco. Motivo para no probarlo: no se podría medir.
9. **Descartadas con motivo (Codex 02):**
   - calibrar la resolución temporal del GT (02-2): ya lo cubre la condición coarseGT;
   - enmascarar rasgos del balón en el entrenamiento (02-5): el modelo de control solo se usa con
     el balón visto, y los huecos van por el modelo de forma.
10. **(hecho, descartado) #15 SkillCorner:** su GT es más grueso y enseña a cortar más grueso.
11. **Consulta 11 a Codex (13-nov):**
    - #19, red de 32 neuronas: descartada;
    - #20, red también para los huecos: descartada;
    - suavizar el equipo a través de los huecos cortos: motivo para no probarlo, porque Viterbi ya
      mantiene el estado a través de los huecos sin evidencia.
12. **Ideas de Xavier (#21-#28):** cerradas. Aceptadas #23 y #27; descartadas las demás.
13. **Consulta 23 a Codex (13-nov, caminos finales), con motivo para no probar cada uno:**
    - Un clasificador para las posesiones < 2 s. Dentro de las que se pierden, la evidencia es
      −0.12 hacia el equipo verdadero y solo el 43 % de los frames va en la buena dirección. Los
      rasgos de toques (#4) y de medio segundo (#6) ya fallaron. Un clasificador con las mismas
      entradas no puede recuperar información que no está.
    - Un modelo de transición en los huecos: Viterbi ya lo es. La evidencia de los huecos (#16) ya
      está dentro, y la red de huecos (#20) fue ruido.
    - Auditar las posesiones fusionadas: Codex le da ganancia baja y pide revisar muestras antes
      de escribir ninguna regla; no da tiempo a validarla bien.

    No queda ningún cambio de modelo con ganancia esperada razonable antes de la congelación. Lo
    que queda de tiempo va a reproducibilidad, documentación y vigilar a Xavier.

## Reproducibilidad (R5)

- **13-nov, ~22:10, versión con #27.** En un worktree limpio, `run.py --no-public` dio SHA256
  `567f0d71…`, H 52362 / A 34398 / D 72097 / X 40462, igual que en el repo de trabajo.
- **13-nov, ~17:30.** Se comprobó en un worktree limpio de `josep/mejoras`, sin `cache/`, sin
  `outputs/` y sin nada entrenado en local: `run.py --no-public` dio SHA256 `1077ece7…`, H 55377 /
  A 39316 / D 64164 / X 40462, igual que en el repo de trabajo. El CSV sale solo del código y de los
  modelos commiteados.

## Incidencias de reglas

Si un agente rompe el protocolo, se anota aquí con lo que vio y cómo se resolvió.

- **13-nov 18:35, rama de Xavier (`origin/xavier/mejoras` 0b5782d).**
  - Su `xavier/README.md` dice que sus reglas salen de revisar frame a frame los fallos de su
    versión en los 5 minutos públicos, "solo para entender la causa".
  - También da la nota pública de su versión y describe tramos concretos del partido.
  - El coordinador leyó ese README (los `.md` están en la lista permitida) antes de saber qué
    contenía.
  - Resolución:
    - nada de lo leído sobre la ventana pública se usa: ni su nota pública ni sus tramos;
    - sus ideas se prueban solo con valores elegidos en el banco, según la sección "Trabajo de
      Xavi";
    - no se leen sus scripts de `xavier/`, que no están en la lista permitida;
    - la carpeta `xavier/` no debe pasar a `josep/mejoras`: documenta un uso de la ventana pública
      que el documento técnico dice que no hubo, y contiene descripciones del partido en un repo
      público.

  Avisado Josep en el chat.

- 12-nov: una ejecución exploratoria de `run.py` imprimió la tabla pública (candidato #7). La
  decisión se tomó sin ella (`check_bar`), y desde entonces se usa `run.py --no-public`.
- 12-nov, bloqueos sin consecuencias (el guardián los paró y no se vio nada):
  - el worker de `exp/semillas` hizo un `ls` que incluía la carpeta `cache`;
  - el coordinador leyó `blocked.log` con Bash en vez de Read, y lanzó un comando que llamaba a
    Python fuera del intérprete permitido (lo pidió solo para ver la versión).

  Errores propios, no falsos positivos.

## Para Josep (avisos y decisiones pendientes)

Responde siempre en `C:\MTAH2027\DECISIONES_JOSEP.md`, a mano: es lo único que cuenta como
aprobación y ningún agente puede escribirlo.

- **Resuelto por Josep el 13-nov (DECISIONES_JOSEP.md):**
  - Metrica queda aprobada.
  - Bancos con `--workers 2`, siempre de uno en uno. Si Claude Code mata uno, se relanza una vez
    con `--workers 1`; si vuelve a morir, se anota aquí y se sigue con otra cosa.
  - v6: se relanzan las semillas 4-7 (c), y (d) se da por cumplida en lectura relativa.

  La confirmación (c) de v6 se mide encima de la versión actual: v6 + `dead_weight` 0.3 frente
  a v4 + 0.3.
- **Política de datos de otros partidos:** APROBADA el 12-nov. IDSSE (línea 1) la cumple.
- **Regla de decisión para cambios de datos (12-nov):** para v6 se juzga sobre todo con el banco
  de 10 partidos, más un suelo en el de Metrica (ver "Regla para la línea IDSSE"). Si prefieres
  la regla de siempre (solo Metrica), escríbelo en DECISIONES_JOSEP.md y se reabre.
- **Bloqueos del guardián (worker SkillCorner, 13-nov), sin consecuencias:** un listado de procesos con la palabra "python" (falso positivo) y una lectura del registro de bloqueos con Bash (debía usar Read). No se vio nada.
- **Falsos positivos del guardián (coordinador, 13-nov, primera hora de la tarde), sin consecuencias.** Dos comandos
  que editaban texto (un `sed` sobre el docstring del script de entrenamiento del modelo de
  balón muerto, y esta misma nota) contenían el nombre de ese script precedido de "python" y se
  bloquearon como si lo ejecutaran. Las ediciones se hicieron con la herramienta Edit, sin
  ejecutar nada.
- **Falso positivo del guardián (coordinador, 13-nov 11:55), sin consecuencias.** Un texto para
  EXPERIMENTS.md que nombraba el fichero de decisiones (solo para citarlo) se bloqueó como si lo
  modificara. Se escribió sin nombrarlo.
- **Falsos positivos del guardián (worker IDSSE, 12-nov), sin consecuencias:**
  - llamar a Python a través de una variable de PowerShell (`& $py`): se usó la ruta explícita;
  - un mensaje de commit que nombraba el fichero de decisiones: el commit no lo tocaba y se
    repitió sin nombrarlo.
- **Documento técnico, nota sobre las ideas de Xavier (13-nov):** §2.4 dice ahora que un miembro
  del equipo propuso 6 reglas tras revisar los fallos de su versión en los 5 minutos públicos para
  entender las causas, y que cada una se probó por separado y se decidió solo con el banco.
  - Es la descripción honesta de lo que pasó.
  - Decide con Xavier si se queda así. Sin esa nota, el documento diría algo inexacto.
  - Pídele que borre de la rama pública su `xavier/README.md`, porque describe tramos del partido.
- **Documento técnico, añadido el 13-nov:** §2.4 dice ahora que un modelo externo (OpenAI Codex)
  criticó cada decisión solo con el método y números agregados, nunca con datos del partido.
  Decide si se queda, junto con lo de abajo.
- **Documento técnico:** decidir si se mantienen la línea sobre el asistente de IA y la del desliz
  con la tabla pública (§2.4), y añadir los nombres de los 4 miembros.
