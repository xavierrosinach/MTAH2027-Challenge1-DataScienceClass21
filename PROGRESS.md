# Progreso

Panel para Josep. Se actualiza después de cada experimento. El detalle está en
[EXPERIMENTS.md](EXPERIMENTS.md) y el documento técnico en
[docs/technical_report.md](docs/technical_report.md).

**Actualizado:** 2026-11-12

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
| Más partidos: IDSSE (7 partidos DFL, CC BY 4.0) para entrenar y ampliar el banco | `exp/idsse` | worker Claude (`C:\MTAH2027\wt-idsse`) | desde 12-nov 18:10: descarga, cargador y GT; después, entrenar con 10 partidos y banco |
| Promedio de 5 redes con semillas distintas | `exp/semillas` | worker Claude (`C:\MTAH2027\wt-semillas`) | desde 12-nov 18:05: implementación y banco |

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

## Cola de ideas (en orden)

1. (en curso) Más partidos: IDSSE. Cumple la política general de Josep del 12-nov: descarga
   pública sin registro (figshare) y licencia CC BY 4.0, comprobadas en la API de figshare.
2. (en curso) Promedio de varias redes entrenadas con semillas distintas.
3. Modelo de secuencias de toques, para las posesiones de menos de 2 s.
4. Smart Tagging como pista de equipo.
5. Otros datasets abiertos para más adelante: SkillCorner opendata (licencia MIT, tracking de
   broadcast). Falta comprobar qué etiquetas trae y si sirven.

## Incidencias de reglas

Si un agente rompe el protocolo, se anota aquí con lo que vio y cómo se resolvió.

- 12-nov: una ejecución exploratoria de `run.py` imprimió la tabla pública (candidato #7). La
  decisión se tomó sin ella (`check_bar`), y desde entonces se usa `run.py --no-public`.

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
- **Documento técnico:** decidir si se mantienen la línea sobre el asistente de IA y la del desliz
  con la tabla pública (§2.4), y añadir los nombres de los 4 miembros.
