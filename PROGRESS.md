# Progreso

Panel para Josep. Se actualiza después de cada experimento. El detalle está en
[EXPERIMENTS.md](EXPERIMENTS.md) y el documento técnico en
[docs/technical_report.md](docs/technical_report.md).

**Actualizado:** 2026-11-12 · **Congelación:** 15-nov 23:59 (después solo se prepara la entrega).

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

## Experimentos

| fecha | rama | cambio | banco spellF1 (vs ref) | decisión |
|---|---|---|---|---|
| 11-nov | `exp/team-switch-16` | coste de cambio 20 -> 16 | 0.463 (+0.009) | aceptada |
| 12-nov | `exp/control-aprendido` | modelo aprendido de quién controla | 0.583 (+0.119) | aceptada |
| 12-nov | `exp/rasgos` | rasgos de movimiento | 0.594 (+0.012) | aceptada |
| 12-nov | `exp/nolineal` | promedio con red antisimétrica | 0.605 (+0.010) | aceptada |
| 12-nov | `exp/peso-posesion` | peso por duración de posesión | 0.613 (+0.008) | **aceptada (actual)** |
| 12-nov | 13 ramas | ver tabla "Branches" de EXPERIMENTS.md | | descartadas |

## Cola de ideas (en orden)

1. Más partidos de otros equipos con tracking + eventos, para entrenar y ampliar el banco.
   Primero hay que comprobar la licencia.
2. Promedio de varias redes entrenadas con semillas distintas.
3. Modelo de secuencias de toques, para las posesiones de menos de 2 s.
4. Smart Tagging como pista de equipo.

## Para Josep (avisos y decisiones pendientes)

- Documento técnico: decidir si se mantienen la línea sobre el asistente de IA y la del desliz con
  la tabla pública (§2.4), y añadir los nombres de los 4 miembros.
