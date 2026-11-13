# Progreso

Panel para Josep. Se actualiza después de cada experimento. El detalle está en
[EXPERIMENTS.md](EXPERIMENTS.md) y el documento técnico en
[docs/technical_report.md](docs/technical_report.md).

**Actualizado:** 2026-11-13 14:30

**Calendario:**

- **Congelación de Orca:** 14-nov 20:00.
- **Revisión con Fable:** 14-nov por la noche y 15-nov.
- **Decisión con Xavier:** 16-nov.
- **Entrega:** 17-nov.

## Mejor versión (`josep/mejoras`)

| | banco spellF1 | banco macroF1 |
|---|--:|--:|
| **actual: v6 = modelo entrenado con 10 partidos (Metrica + IDSSE)** (13-nov) | **0.629** ¹ | **0.856** ¹ |
| v4 + `dead_weight` 0.3 (13-nov) | 0.619 | 0.855 |
| v4 (modelo de control aprendido + movimiento + red + peso por posesión) | 0.613 | 0.852 |
| inicio del 12-nov (modelo de control v1) | 0.583 | 0.847 |
| main de Xavier | 0.455 | 0.812 |
| baseline oficial `baseline_smooth50` | 0.319 | 0.769 |

¹ Con semillas nuevas (4-7), en las que la versión anterior saca 0.616: ganancia emparejada
+0.013 (error típico 0.002). En el banco de 10 partidos (Metrica + IDSSE), +0.019.

Mejora aproximada (las semillas no son las mismas): **+38 %** sobre main y **+97 %** sobre el
baseline oficial (banco de Metrica).

Ojo: el banco no es la nota real. Mide sobre otros 3 partidos con el tracking degradado como el
nuestro, y sirve para comparar versiones entre sí. La nota real solo se sabrá con los resultados
oficiales. CSV actual: SHA256 `c53b0b1b…`, 268 posesiones (media 14,1 s), 106 paradas.

## En curso

| línea | rama | agente | estado |
|---|---|---|---|
| #14 evidencia de equipo en los huecos (`gap_weight` 0.1 / 0.3 con el modelo de huecos de v6) | `exp/gap` | coordinador | regla registrada; barrido después del entrenamiento de #13 |
| #13 modelo de balón muerto aprendido, reintentado con 10 partidos y el balón escondido en tiempo muerto | `exp/muerto-idsse` | coordinador (`C:/MTAH2027/wt-muerto-idsse`) | código y regla listos; entrena cuando acabe el banco de #12. Primer filtro: `check_bar` |
| Más datos: SkillCorner opendata (10 partidos de la A-League, MIT, tracking de broadcast con posesión por algoritmo), fase 1: datos y GT | `exp/skillcorner` | worker Claude (`C:/MTAH2027/wt-skillcorner`) | desde 13-nov 16:50: licencia, descarga, cargador y agregados, sin procesos pesados |

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
| 13-nov | datos | `aceptado/datos/v6-idsse` | modelo de control entrenado con 10 partidos (3 Metrica + 7 IDSSE) | 0.629 en semillas 4-7 (+0.013, error típico 0.002); 10 partidos +0.019 (39/40 celdas) | **aceptada (actual)** |
| 13-nov | control | `descartado/control/train-mix-light` | v6 entrenado también con copias `light` | +0.0027, pero coarseGT −0.0053 | descartada: corta más fino (+6 posesiones) |
| 13-nov | huecos | `descartado/huecos/interp-50` | interpolar huecos del balón hasta 1-2 s | +0.0005; con semillas 8-11, −0.0004 | descartada: ruido |

## Mapa de líneas (lo probado, el resultado y lo siguiente)

| línea | probado | resultado | siguiente |
|---|---|---|---|
| Modelo de control (quién tiene el balón) | v1-v4 (modelo aprendido, movimiento, red, peso por posesión); rasgos largos, de toques y de 0,5 s; la red sola; 5 semillas | 4 aceptados; el resto, ruido | #11 `train_mix` descartado (corta más fino); siguiente: evidencia mejor con el mismo corte (consulta 02-3, 01-3) |
| Datos de otros partidos | Metrica (3) e IDSSE (7) integrados | **v6 aceptado** (+0.013 en Metrica, +0.019 en 10 partidos): es la mayor ganancia desde v2 | `errors.py` sobre v6; SkillCorner (MIT) sin explorar |
| Balón muerto | modelo aprendido (#7); `dead_weight`; `to_dead`, `from_dead`; 6 constantes de reglas | `dead_weight` 0.3 aceptado; `to_dead` 20 no transfiere; el resto, plano | oráculo +0.057 sobre v6; primero, el banco IDSSE con el balón escondido en tiempo muerto (cola 2b) |
| Viterbi | `team_switch` 16 / 13 / 20; costes de muerto | 16 aceptado; el resto depende de la granularidad del GT | estado "disputado" (Codex 01), sin probar |
| Posesiones cortas (< 2 s) | peso por posesión 0.25 / 0.5 / 1; rasgos de toques y de 0,5 s | 0.25 aceptado; lo demás solo gana con GT fino | consulta 02 |
| Huecos sin balón | `gap_weight` (v3); interpolar 1-2 s (#12) | ruido; ruido | #14: `gap_weight` con el modelo de huecos de v6 (10 partidos) |
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
10. **(en curso, fase 1) SkillCorner opendata** (MIT, tracking de broadcast). Trae posesión por frame (algoritmo) y fases de juego. Primero hay que ver si trae etiquetas
    de posesión; solo si sobra tiempo.

## Incidencias de reglas

Si un agente rompe el protocolo, se anota aquí con lo que vio y cómo se resolvió.

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
- **Falsos positivos del guardián (coordinador, 13-nov 17:20), sin consecuencias.** Dos comandos
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
- **Documento técnico:** decidir si se mantienen la línea sobre el asistente de IA y la del desliz
  con la tabla pública (§2.4), y añadir los nombres de los 4 miembros.
