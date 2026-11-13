# Progreso

Panel para Josep. Se actualiza después de cada experimento. El detalle está en
[EXPERIMENTS.md](EXPERIMENTS.md) y el documento técnico en
[docs/technical_report.md](docs/technical_report.md).

**Actualizado:** 2026-11-13 05:45

**Calendario:**

- **Congelación de Orca:** 14-nov 20:00.
- **Revisión con Fable:** 14-nov por la noche y 15-nov.
- **Decisión con Xavier:** 16-nov.
- **Entrega:** 17-nov.

## Mejor versión (`josep/mejoras`)

| | banco spellF1 | banco macroF1 |
|---|--:|--:|
| **actual: v4 + `dead_weight` 0.3** (13-nov) | **0.619** | **0.855** |
| v4 (modelo de control aprendido + movimiento + red + peso por posesión) | 0.613 | 0.852 |
| inicio del 12-nov (modelo de control v1) | 0.583 | 0.847 |
| main de Xavier | 0.455 | 0.812 |
| baseline oficial `baseline_smooth50` | 0.319 | 0.769 |

Mejora: **+35 %** sobre main y **+92 %** sobre el baseline oficial.

Ojo: el banco no es la nota real. Mide sobre otros 3 partidos con el tracking degradado como el
nuestro, y sirve para comparar versiones entre sí. La nota real solo se sabrá con los resultados
oficiales. CSV actual: SHA256 `b122c3fc…`, 268 posesiones (media 14,1 s), 106 paradas.

## En curso

| línea | rama | agente | estado |
|---|---|---|---|
| Más partidos: IDSSE (7 partidos DFL, CC BY 4.0) para entrenar y ampliar el banco | `exp/idsse` | coordinador (`C:\MTAH2027\wt-idsse`) | v6 medido: cumple (a), (b) y (e); (c) parado por falta de memoria; (d) ver abajo. Herramientas IDSSE y sentido de ataque ya en `josep/mejoras` (638444d, mismo hash) |
| Modelo de control entrenado también con copias `light` (`train_mix`, idea 4 de Codex) | `exp/mezcla` | coordinador (`C:\MTAH2027\wt-mezcla`) | v7 entrenado (pérdida en partido no visto 0.169 / 0.159 / 0.182, frente a 0.160 / 0.159 / 0.179 de v4); el banco lo paró la falta de memoria: espera a Josep |

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
| 12-nov | control | `aceptado/control/v4-peso-posesion` | peso por duración de posesión | 0.613 (+0.008) | **aceptada (actual)** |
| 11/12-nov | todos | 13 etiquetas `descartado/...` | ver EXPERIMENTS.md | | descartadas |
| 12-nov | control | `descartado/control/semillas` | promedio de 5 redes con semillas distintas | 0.614 (+0.001, error típico 0.001) | descartada: pierde en light y +tags (ruido) |
| 13-nov | muerto | `aceptado/muerto/dead-weight-03` | peso del modelo de velocidad de balón muerto 0.5 -> 0.3 | 0.619 (+0.006; semillas 4-7 +0.009, 12/12) | **aceptada (actual)** |
| 13-nov | muerto | `descartado/muerto/to-dead-20` | coste de pasar a muerto 10 -> 20 | 0.616 (+0.003, 11/12) | descartada: en BAR-ATH las paradas bajan de 104 a 94 |
| 13-nov | reglas | (barridos, sin etiqueta) | `pass_llr`, `kick_team_llr`, `from_dead` | ±0.002 | sin efecto (EXPERIMENTS #9) |
| 13-nov | muerto | (barridos, sin etiqueta) | `dead_llr_clip`, `inside_llr`, `out_llr`, `isolated_llr`, `latch_llr`, `kick_llr` sobre la versión actual | ±0.001 | sin efecto; quitar `inside_llr` cuesta −0.020 (EXPERIMENTS #9) |

## Cola de ideas (en orden)

1. (en curso) Más partidos: IDSSE. Cumple la política general de Josep del 12-nov: descarga
   pública sin registro (figshare) y licencia CC BY 4.0, comprobadas en la API de figshare.
2. (hecho) Constantes de reglas: `dead_weight` 0.3 aceptado; el resto, sin efecto o sin
   transferir.
3. (en curso) Modelo de control entrenado también con copias `light` (`train_mix`), para que no
   dependa de una sola degradación. Es la idea 4 de Codex (consulta del 13-nov,
   `C:/consultas-ia/ideas-posesion.md`).
4. Otras ideas de Codex, por probar:
   - escalar la confianza del modelo según la calidad del tracking (distancia del balón al
     jugador más cercano, jugadores visibles);
   - un estado "disputado" en Viterbi;
   - un corrector temporal aprendido sobre los log-odds.
5. Modelo de secuencias de toques, para las posesiones de menos de 2 s (los rasgos de toques ya
   fallaron).
6. Smart Tagging como pista de equipo (no se puede validar en el banco).
7. Otros datasets abiertos para más adelante: SkillCorner opendata (licencia MIT, tracking de
   broadcast). Falta comprobar qué etiquetas trae y si sirven.

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

- **Metrica necesita su propia línea de aprobación (12-nov).** Su repo no tiene fichero de licencia.
  El README solo dice "be responsible with the use of this data" y "acknowledge the source", así
  que la política general (licencia abierta explícita) no la cubre. Todo lo aceptado (banco y
  modelos v1-v4) depende de ella. Se sigue usando porque ya estaba en uso y `check_rules.py` la da
  por aprobada. Si estás de acuerdo, añade en DECISIONES_JOSEP.md, "Aprobaciones concretas", una
  línea como esta: "Metrica Sports sample data (3 partidos anonimizados, github
  metrica-sports/sample-data, sin licencia explícita): aprobada para entrenar y validar, citando la
  fuente."
- **Política de datos de otros partidos:** APROBADA el 12-nov. IDSSE (línea 1) la cumple.
- **MEMORIA, todo parado (13-nov 05:40).** Claude Code mata los bancos que corren en segundo
  plano cuando falta memoria (6,7 GB libres de 15,4) y no deja relanzarlos sin tu permiso. Esta
  vez fue un solo banco de Metrica (v7, `train_mix`), con el entrenamiento ya hecho. Opciones:
  - (a) me autorizas a relanzar los bancos con `--workers 2`: la mitad de memoria y el doble de
    tiempo, ~20 min por banco;
  - (b) arrancas Claude Code con `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1` (solo puedes
    hacerlo tú);
  - (c) cierras programas para liberar memoria.

  Pendientes de banco: v7 (`exp/mezcla`) y v6 (`exp/idsse`).
- **v6 (13-nov), dos decisiones tuyas:**
  1. ¿Relanzo las semillas 4-7 de v6 (condición c)? Claude Code las paró por falta de memoria y
     no deja relanzarlas sin tu permiso. Iría un banco cada vez: hay 8 GB libres de 15,4.
  2. ¿Vale (d)? El umbral absoluto que escribí no se cumple (72,1 % < 76 %), pero la relación
     con el banco es la misma que con v4. Si dices que no, v6 queda descartado.

  Mientras tanto, `josep/mejoras` es v4 + `dead_weight` 0.3. Si sigue adelante, v6 se vuelve a
  medir encima (semillas 0-3 y 4-7, Metrica y DFL), un banco cada vez.
- **Regla de decisión para cambios de datos (12-nov):** para v6 se juzga sobre todo con el banco
  de 10 partidos, más un suelo en el de Metrica (ver "Regla para la línea IDSSE"). Si prefieres
  la regla de siempre (solo Metrica), escríbelo en DECISIONES_JOSEP.md y se reabre.
- **Falsos positivos del guardián (worker IDSSE, 12-nov), sin consecuencias:**
  - llamar a Python a través de una variable de PowerShell (`& $py`): se usó la ruta explícita;
  - un mensaje de commit que nombraba el fichero de decisiones: el commit no lo tocaba y se
    repitió sin nombrarlo.
- **Documento técnico:** decidir si se mantienen la línea sobre el asistente de IA y la del desliz
  con la tabla pública (§2.4), y añadir los nombres de los 4 miembros.
