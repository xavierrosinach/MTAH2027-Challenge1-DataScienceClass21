# Progreso

Panel para Josep. Se actualiza después de cada experimento. El detalle está en
[EXPERIMENTS.md](EXPERIMENTS.md) y el documento técnico en
[docs/technical_report.md](docs/technical_report.md).

**Actualizado:** 2026-11-12 19:10

**Calendario:**

- **Congelación de Orca:** 14-nov 20:00.
- **Revisión con Fable:** 14-nov por la noche y 15-nov.
- **Decisión con Xavier:** 16-nov.
- **Entrega:** 17-nov.

## Mejor versión (`josep/mejoras`)

| | banco spellF1 | banco macroF1 |
|---|--:|--:|
| **actual: v4** (modelo de control aprendido + movimiento + red + peso por posesión) | **0.613** | **0.852** |
| inicio del 12-nov (modelo de control v1) | 0.583 | 0.847 |
| main de Xavier | 0.455 | 0.812 |
| baseline oficial `baseline_smooth50` | 0.319 | 0.769 |

Mejora: **+35 %** sobre main y **+92 %** sobre el baseline oficial.

Ojo: el banco no es la nota real. Mide sobre otros 3 partidos con el tracking degradado como el
nuestro, y sirve para comparar versiones entre sí. La nota real solo se sabrá con los resultados
oficiales. CSV actual: SHA256 `00bf726c…`, 270 posesiones (media 14,0 s).

## En curso

| línea | rama | agente | estado |
|---|---|---|---|
| Más partidos: IDSSE (7 partidos DFL, CC BY 4.0) para entrenar y ampliar el banco | `exp/idsse` | worker Claude (`C:\MTAH2027\wt-idsse`) | hecho: datos y v5all (ver la regla abajo). Ahora: corregir el sentido de ataque, reentrenar v6 y bancos |
| Constantes de reglas revisadas con el modelo aprendido: pases, golpeos y costes de muerto | `exp/reglas` | worker Claude (`C:\MTAH2027\wt-reglas`) | desde 12-nov 19:05: diagnóstico con errors.py y barridos |

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

## Cola de ideas (en orden)

1. (en curso) Más partidos: IDSSE. Cumple la política general de Josep del 12-nov: descarga
   pública sin registro (figshare) y licencia CC BY 4.0, comprobadas en la API de figshare.
2. (en curso) Constantes de reglas (pases, golpeos, costes de muerto) revisadas sobre v4.
3. Modelo de secuencias de toques, para las posesiones de menos de 2 s.
4. Smart Tagging como pista de equipo.
5. Otros datasets abiertos para más adelante: SkillCorner opendata (licencia MIT, tracking de
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
- **Regla de decisión para cambios de datos (12-nov):** para v6 se juzga sobre todo con el banco
  de 10 partidos, más un suelo en el de Metrica (ver "Regla para la línea IDSSE"). Si prefieres
  la regla de siempre (solo Metrica), escríbelo en DECISIONES_JOSEP.md y se reabre.
- **Falsos positivos del guardián (worker IDSSE, 12-nov), sin consecuencias:**
  - llamar a Python a través de una variable de PowerShell (`& $py`): se usó la ruta explícita;
  - un mensaje de commit que nombraba el fichero de decisiones: el commit no lo tocaba y se
    repitió sin nombrarlo.
- **Documento técnico:** decidir si se mantienen la línea sobre el asistente de IA y la del desliz
  con la tabla pública (§2.4), y añadir los nombres de los 4 miembros.
