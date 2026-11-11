# MTAH2027-Challenge1-DataScienceClass21

Código del equipo **Data Science Class 21** para el reto inicial de More Than a Hack 2027: dado un
partido completo, producir una etiqueta por frame indicando qué equipo tiene la posesión del balón
(`H` local, `A` visitante, `D` balón muerto, `X` sin respuesta).

## Estructura

```
config.yaml              todos los parámetros, comentados en castellano (empezar aquí)
run.py                   datos + config -> CSV de entrega, comparación con las baselines y vídeos
src/possession/
  config.py              carga config.yaml y resuelve rutas
  data.py                lee el tracking (ATD) y el Smart Tagging, con caché en cache/
  context.py             el etiquetador: evidencias por frame + Viterbi sobre {H, A, D}
  diagnostics.py         comprobaciones sin ground truth (output.diagnostics: true)
  submission.py          escribe el CSV, validador oficial y tabla comparativa
  data_proc.py           cruza ATD + Smart Tagging + homografía en tablas (las usan los vídeos)
  video.py               vídeos de revisión con el tracking y nuestra etiqueta
docs/technical_report.md documento técnico (entregable)
```

## Requisitos

- Python 3.12
- El repositorio oficial clonado al lado: `../more-than-a-hack-2027`
  (se usan su `games.py`, el validador, el evaluador y sus baselines)
- Los datos del partido en `../data` o en la ruta de la variable `MTAH_DATA_DIR`

```
More Than a Hack - Data Science 21 Class/
├── data/                                   datos (no versionados)
├── more-than-a-hack-2027/                  repo oficial
└── MTAH2027-Challenge1-DataScienceClass21/ este repo
```

## Ejecutar

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows  (source .venv/bin/activate en Linux/macOS)
pip install -r requirements.txt
python run.py
```

La primera ejecución tarda 1-2 minutos (lee el tracking y lo guarda en `cache/`); las
siguientes, unos segundos. La salida es solo esto:

```
written outputs/submissions/DataScienceClass21.csv   H ...  A ...  D ...  X ...

public 5 min (only a check, not for tuning):
rank  submission   macroF1  spellF1  poss ...
   1  DataScienceClass21 ...
   2  baseline_smooth125 ...
   ...
```

1. Escribe `outputs/submissions/DataScienceClass21.csv`.
2. Lo pasa por el validador oficial (solo imprime algo si falla).
3. Imprime **una tabla que compara todos los modelos** con los 5 minutos públicos: nuestro CSV
   y cualquier otro que haya en `outputs/submissions/`, más las cuatro baselines oficiales.
4. Si `video.enabled` es `true`, genera los vídeos de revisión.

**Importante:** los 5 minutos públicos son 17 posesiones. La tabla solo sirve para comprobar que
nada está roto y ver distancias grandes entre modelos. Elegir parámetros con ella está prohibido
(RULES.md) y además engaña: con `output.bootstrap: 1000` se ve que el margen de ruido es enorme.

## Cómo funciona el etiquetador (`src/possession/context.py`)

El balón falta en el 41 % de los frames evaluados, en huecos largos, así que no basta con mirar
quién está más cerca. Para cada frame se suman evidencias de tres estados (H, A, D) y Viterbi
elige la secuencia más probable del partido entero, con un coste por cada cambio de estado.

**¿Balón muerto?**
- Modelo de balón muerto: regresión logística sobre la velocidad de los jugadores, entrenada con
  pseudo-etiquetas automáticas (balón fuera del campo, inicio de un tag `SET PIECES`, antes del
  saque del 2.º tiempo = muerto; balón moviéndose a los pies de un jugador = vivo).
- Reglas del balón: visto fuera del campo; perdido al salir o yendo a cruzar una línea; abandonado
  (nadie a < 6 m durante 2 s); tras salir sigue muerto hasta el siguiente golpeo (el saque).
- Goles: muerto desde el último balón cerca de la portería hasta el tag `Kick Off (After Goal)`.
- Un tramo D necesita alguna de estas razones; si solo hay jugadores lentos, pasa a vivo.
- Con el balón visible, reanudar solo es barato en el momento de un golpeo.

**¿Qué equipo?**
- Cercanía al balón (solo con contacto: con el balón suelto la posesión sigue al último equipo).
- Pases completados: toque de un jugador y control de un compañero a >= 5 m; dos seguidos bastan
  para una posesión corta. Cuentan los pases de primeras; un roce corto del rival no los corta.
- Quien golpea el balón da media evidencia de cambio a su equipo: decide quién saca, pero un
  despeje solo no cambia la posesión.
- Porteros: el tracking a veces intercambia sus identidades; su equipo se fija por la portería
  que defienden.
- Donde no se ve el balón: modelo de forma de los equipos (anchura, profundidad, avance; 2.ª
  pasada, entrenado con los frames de contacto claro) y, débil, las fases del Smart Tagging.
- Balón parado con equipo en el tag: ese equipo saca.

**Viterbi:** cambiar H↔A cuesta 20, pasar a D 10, reanudar 5; un tramo D dura al menos 1 s. Como
mira el partido entero, el cambio cae donde cambia la evidencia (cuando el nuevo equipo toca el
balón), sin el retraso de un filtro causal.

Ninguna constante se ha elegido con la nota pública: salen de razonar sobre el fútbol, de mirar
los datos y los vídeos, y de los diagnósticos sin ground truth (`output.diagnostics: true`).

## Vídeos de revisión

En `config.yaml`:

```yaml
video:
  enabled: true          # generar vídeos al ejecutar run.py
  blocks: [1T_01]        # qué bloques: [1T_01], [1T], [2T_03, 2T_04]; [] = todos
```

`python run.py` escribe `outputs/videos/<parte>T_<nº>_<desde>-<hasta>.mp4`, bloques de 5 minutos
de reloj de partido (`1T_01_00m00-04m59.mp4`, ...). Cada vídeo muestra el vídeo táctico con el
tracking encima (círculos azules H, rojos A, balón amarillo, cruz gris = último balón visto), un
minimapa y **nuestra etiqueta en grande arriba a la derecha** con el color del equipo (gris = D).
Cada bloque tarda ~1 minuto y ocupa ~120 MB. La primera vez se construyen las tablas de
`data_proc.py` en `cache/` (~30 s). Necesita `opencv-python-headless`.

## Datos

Los datos del partido no están ni pueden estar en este repositorio (`.gitignore` los excluye,
también los vídeos `.mp4`).
