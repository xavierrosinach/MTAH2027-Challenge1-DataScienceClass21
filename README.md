# MTAH2027-Challenge1-DataScienceClass21

Código del equipo **Data Science Class 21** para el reto inicial de More Than a Hack 2027: dado un
partido completo, producir una etiqueta por frame indicando qué equipo tiene la posesión del balón
(`H` local, `A` visitante, `D` balón muerto, `X` sin respuesta).

## Estructura

```
config.yaml              todos los parámetros, comentados en castellano (empezar aquí)
run.py                   datos + config -> CSV de entrega, comparación con las baselines y vídeos
create_video_gt.py       vídeo de los 5 min públicos: etiqueta del GT y la nuestra lado a lado
src/possession/
  config.py              carga config.yaml y resuelve rutas
  data.py                lee el tracking (ATD) y el Smart Tagging, con caché en cache/
  broadcast.py           tracking del broadcast: alineación con el táctico, plano en directo y balón
  context.py             el etiquetador: evidencias por frame + Viterbi sobre {H, A, D}
  control.py             modelo aprendido de qué equipo controla el balón
  jersey.py              equipo de cada track por el color de la camiseta en el vídeo táctico
  dead.py                modelo aprendido de balón muerto (opción apagada)
  whistle.py             silbatos del árbitro en el audio del broadcast (descartado, apagado)
  diagnostics.py         comprobaciones sin ground truth (output.diagnostics: true)
  submission.py          escribe el CSV, validador oficial y tabla comparativa
  data_proc.py           cruza ATD + Smart Tagging + homografía en tablas (las usan los vídeos)
  video.py               vídeos de revisión con el tracking y nuestra etiqueta
models/                  pesos del modelo de control (entrenado en otros partidos, nunca en BAR-ATH)
docs/technical_report.md documento técnico (entregable)
```

## Resultado

En los 5 minutos públicos (solo comprobación, 17 posesiones): **macroF1 0,934 y spellF1 0,764**
(16 posesiones predichas). Con el modelo de control anterior (v6) daba 0,935 y 0,723.
La mejor baseline oficial (`baseline_smooth125`) da 0,581 y 0,424.

`python run.py --no-public` escribe el CSV con SHA256 `ed310554…`.

Las decisiones no se han tomado con esa cifra, sino en un banco de validación con otros partidos
(ver "Cómo se ha validado").

## Requisitos

- Python 3.12
- El repositorio oficial clonado al lado: `../more-than-a-hack-2027`
  (se usan su `games.py`, el validador, el evaluador y sus baselines)
- Los datos del partido en `../data` o en la ruta de la variable `MTAH_DATA_DIR`: el tracking
  táctico y el Smart Tagging, y el tracking del broadcast (si falta, el etiquetador funciona igual
  sin las dos reglas que lo usan)

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

La primera ejecución tarda unos minutos (lee los dos trackings y los guarda en `cache/`); las
siguientes, alrededor de un minuto. `python run.py --no-public` escribe el CSV e imprime su
SHA256, sin la tabla ni los vídeos. La salida normal es solo esto:

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
- Hueco que es parada: si el balón desaparece al menos 3 s y reaparece a menos de 10 m de donde
  se perdió, es una falta o un fuera de juego (se saca desde el sitio).
- Saque desde la línea: córners y saques de banda empiezan sobre la línea; ese golpeo también
  cuenta como reanudación.
- Saque falso: un golpeo no reanuda si el balón vuelve a pararse en menos de 6 s sin que lo toque
  un segundo jugador (el portero que lo coloca, devolverlo al sitio).
- Un tramo D necesita alguna razón; si solo hay jugadores lentos, pasa a vivo.
- Con el balón visible, reanudar solo es barato en el momento de un golpeo.
- Recorte de los bordes de las paradas: en los bancos, las paradas que decide el etiquetador
  empiezan 0,3-0,5 s antes y se reanudan 0,1-0,25 s después que en el ground truth. Tras decidir,
  el inicio de cada parada se retrasa 0,3 s y la reanudación se adelanta 0,15 s.

**¿Qué equipo?**
- Modelo de control (`control.py`): una red pequeña promediada con un modelo lineal, que estima
  qué equipo controla el balón a partir de las distancias al balón y del movimiento de los
  jugadores respecto a él. Está entrenado en 10 partidos ajenos (3 de Metrica y 7 de IDSSE) con el
  tracking degradado para que se parezca al nuestro; nunca ha visto BAR-ATH. Sustituye a la
  simple cercanía al balón. Versión v8 (`models/control_v8_*.json`): al entrenar se añaden copias
  de los mismos partidos con un tracking menos degradado (condición `light` del banco), que tienen
  más cambios de posesión limpios para aprender.
- Pases completados: toque de un jugador y control de un compañero a >= 5 m; dos seguidos bastan
  para una posesión corta. Cuentan los pases de primeras; un roce corto del rival no los corta.
- Quien golpea el balón da evidencia a su equipo: decide quién saca, pero un despeje solo no
  cambia la posesión.
- Porteros: el tracking a veces intercambia sus identidades; su equipo se fija por la portería
  que defienden.
- Camiseta (`jersey.py`): cuando dos rivales se solapan, el tracker a veces da el track al
  equipo contrario y un jugador pasa a contar como rival. Cada 0,2 s se lee del vídeo táctico el
  color del torso de cada track; un modelo color -> equipo se ajusta con el equipo de los
  metadatos como objetivo (ninguna etiqueta humana) y, por track, se decide el equipo más
  probable en el tiempo. Corrige 240 tramos (1,5 % del tiempo de track). La primera ejecución
  lee el vídeo (~10 min) y deja las muestras en `cache/`.
- Donde no se ve el balón: el modelo de control sigue opinando con poco peso (0,1 por frame, solo
  con la posición de los equipos), más el modelo de forma de los equipos (anchura, profundidad, avance; 2.ª
  pasada, entrenado con los frames de contacto claro) y, débil, las fases del Smart Tagging.
- Balón parado con equipo en el tag: ese equipo saca.

**Viterbi:** cambiar H↔A cuesta 16, pasar a D 10, reanudar 5; un tramo D dura al menos 3 s. Como
mira el partido entero, el cambio cae donde cambia la evidencia (cuando el nuevo equipo toca el
balón), sin el retraso de un filtro causal.

## Qué aporta el broadcast (`src/possession/broadcast.py`)

La base es siempre el tracking táctico: ningún jugador se añade ni se mueve. Del broadcast se
toman tres cosas.

1. **Alineación.** Las dos grabaciones no están sincronizadas. Como ambos trackings están en
   coordenadas de campo, el desfase es el que pone el balón del broadcast encima del balón del
   táctico. Se busca solo, por parte, en ventanas de un minuto, y sale constante en cada parte.
2. **Plano en directo.** Un frame del broadcast solo se usa si muestra el mismo instante: sus
   jugadores tienen que caer sobre los del táctico (mediana < 2 m). Así se descartan repeticiones,
   primeros planos y planos de grada. Queda el 61 % del partido.
3. **Balón donde el táctico no lo ve.** En esos frames en directo se usa el balón del broadcast si
   falta el del táctico (3,8 % del partido).
4. **Segunda opinión sobre quién tiene el balón.** El táctico pierde a veces a un jugador cuando
   se solapa con un rival, o le da el track al equipo contrario; la otra cámara suele verlo bien.
   Por eso el modelo de control se aplica también a lo que ve el broadcast por sí solo (su balón,
   sus jugadores, sus equipos) y, donde las dos cámaras ven el balón, se promedian las dos
   evidencias, con el táctico pesando más (`control.second_view_weight: 0.3`). Las cámaras discrepan sobre el equipo en el 7 %
   de esos frames. No se copian jugadores de una cámara a la otra: emparejarlos falla justo en
   los duelos.
5. **La realización como motivo de parada.** La TV corta a repeticiones y primeros planos sobre
   todo con el balón parado: está fuera del plano general en el 50-66 % de los frames claramente
   parados y en el 12 % de los claramente vivos. Una parada que pasa más de la mitad de su
   duración en corte de TV tiene motivo y no se convierte en juego vivo. Recupera, por ejemplo,
   paradas por tarjeta o por protestas que el Smart Tagging no etiqueta.

## Cómo se ha validado

Ninguna constante se ha elegido con la nota pública (RULES.md). Las reglas salen de razonar sobre
el fútbol y de mirar los datos y los vídeos; después se deciden en un banco de validación con
otros partidos (los tres partidos públicos de Metrica, con el tracking degradado para parecerse
al nuestro: jugadores fragmentados, balón con ruido y con huecos), comparando cada cambio con
diferencias emparejadas por partido y semilla. El banco no forma parte de este repositorio.

| Cambio | Efecto en spellF1 (banco) |
|---|---|
| Modelo de control aprendido (10 partidos ajenos) frente a la cercanía al balón | 0,583 -> 0,629 |
| Modelo de control v8: entrenado también con copias `light` (frente al v6) | light +0,008/+0,009, degraded +0,001/+0,002, degraded+tags +0,001/+0,003, GT grueso −0,005/0,000; macroF1 +0,0008 (dos bancos, tres rondas cada uno) |
| Evidencia de equipo en los huecos (`gap_weight` 0,1) | +0,002 |
| Saque desde la línea | +0,002 a +0,003 |
| Hueco que es parada (a 10 m) | +0,027 a +0,029 |
| Balón del broadcast donde falta el del táctico | +0,019 a +0,023 (optimista: el banco tiene el doble de juego vivo sin balón que nuestro partido) |
| Parada mínima de 3 s y "reanudar exige juego" | +0,005 a +0,008 |
| Realización de TV como motivo de parada | +0,005 |
| Recorte de los bordes de las paradas | +0,003 a +0,004 |
| Segunda opinión del broadcast, peso 0,3 (dos cámaras simuladas: cada una pierde a sus jugadores en los duelos y da algunos tracks al equipo contrario; el balón del broadcast tiene errores grandes el 10 % del tiempo) | +0,005 a +0,011 en las cuatro condiciones, 9 a 12 de 12 ejecuciones mejores |

El banco simula el broadcast (su balón, la realización de TV y, para la segunda opinión, sus
jugadores) con tasas medidas sin etiquetas en nuestro partido. La segunda opinión se probó con
pesos 0,3 / 0,5 / 0,7; 0,3 es el más constante entre condiciones.

Lo que el banco no puede medir se ha comprobado en el propio partido, sin etiquetas humanas:

| Métrica sin etiquetas | Qué mide | Resultado |
|---|---|---|
| Esconder y rellenar | Se esconde el balón del táctico en tramos donde las dos cámaras lo ven y se compara con las etiquetas del táctico completo | El balón del broadcast recupera el 96-97 % de las etiquetas dentro del hueco; sin rellenar, el 72-78 % |
| Testigo independiente | El broadcast tiene su propio tracker: si un arreglo del táctico lo acerca a lo que ve el broadcast, es que acierta | Donde la camiseta cambia la decisión del táctico (0,9 % de los frames), el broadcast da la razón a la camiseta el 72,5 % de las veces |
| Consenso antes y después | Cuando las cámaras discrepan, se mira qué equipo tenía el balón con acuerdo claro de ambas justo antes y justo después | Error del táctico solo 4,5 %, del broadcast solo 4,1 %, de la media 3,6 % (anclas a 4 s) |

**Balón perdido.** El táctico no ve el balón en el 40,7 % de los frames evaluados, pero casi todo
son paradas: no hay huecos de entre 0,5 y 5 s (el tracker ya los rellena) y el 90 % de los frames
sin balón quedan como balón muerto. Juego vivo sin balón del táctico son unos 270 s del partido,
y en el 55 % de ellos el broadcast sí lo ve. Ahí se usa el balón del broadcast con los jugadores
del táctico; usar en su lugar solo lo que ve el broadcast no reproduce mejor las etiquetas
(96-97 % frente a 97 %).

Los comentarios de `config.yaml` citan `EXPERIMENTS.md` (con números como #27 o #44) y scripts
de `validation/`: son el registro de experimentos y el banco, que viven en las ramas de trabajo
y no en `main`. Las opciones marcadas "SIN VALIDAR" están apagadas.

## Seis ideas probadas y descartadas (código presente, opciones apagadas)

Se implementaron seis ideas más, cada una detrás de su opción en `config.yaml`. Ninguna pasa la
validación, así que todas están apagadas y la entrega no cambia.

| # | Idea | Opción | Por qué está apagada |
|---|---|---|---|
| 1 | Silbato del árbitro en el audio del broadcast (`whistle.py`) | `context.whistle.enabled` | Los "silbatos" que detecta no caen más cerca de los saques ni de los balones fuera que frames al azar (8 % frente a 13,5 %; 9 % frente a 8 %): el detector no distingue el silbato del ruido del estadio |
| 2 | Golpeos y pases que ve el broadcast donde el táctico no vio ninguno | `context.passes.second_view_weight` | Banco: +0,001 a +0,002 con las semillas 0-3, pero −0,001 a +0,001 con las 4-7: no se confirma |
| 3 | Sin evidencia de equipo con el balón por el aire | `context.control.air_neutral` | El banco no simula balones aéreos, así que no se puede medir; quita evidencia en el 15 % de los frames con los dos balones |
| 4 | Camiseta también en los tracks del broadcast | `context.jersey.broadcast` | El banco no tiene vídeo, así que no se puede medir |
| 5 | Peso de la segunda opinión según cuánto coinciden las cámaras | `context.control.second_view_reliability.enabled` | Banco: −0,0001 a −0,0021 en las cuatro condiciones |
| 6 | Toque visto por las dos cámaras a la vez | `context.passes.two_camera_touch.enabled` | Banco: entre −0,0005 y +0,0012 (ruido); solo encuentra 5 toques en BAR-ATH |

Con las seis encendidas a la vez, los 5 minutos públicos bajaban de 0,935 / 0,723 a 0,931 / 0,697.

## Vídeos de revisión

En `config.yaml`:

```yaml
video:
  enabled: true          # generar vídeos al ejecutar run.py
  blocks: [1T_01]        # qué bloques: [1T_01], [1T], [2T_03, 2T_04]; [] = todos
```

`python run.py` escribe `outputs/videos/<parte>T_<nº>_<desde>-<hasta>.mp4`, bloques de 5 minutos
de reloj de partido (`1T_01_00m00-04m59.mp4`, ...). Cada vídeo muestra el vídeo táctico con el
tracking encima (círculos azules H, rojos A, balón amarillo, cruz gris = último balón visto;
cuadrados con "TV" = jugadores que solo ve el broadcast, rombo = balón que solo ve el broadcast;
los colores son los del equipo ya corregido por la camiseta), un
minimapa y **nuestra etiqueta en grande arriba a la derecha** con el color del equipo (gris = D).
Cada bloque tarda ~1 minuto y ocupa ~120 MB. La primera vez se construyen las tablas de
`data_proc.py` en `cache/` (~30 s). Necesita `opencv-python-headless`.

## Vídeo comparado con el ground truth público

```bash
python create_video_gt.py                          # con config.yaml
python create_video_gt.py --config mi_variante.yaml
```

Ejecuta el etiquetador con esa configuración (no escribe ningún CSV) y genera
`outputs/videos/GT_vs_<config>.mp4`: los 5 minutos que tienen ground truth público (frames
11185-18684), con el mismo vídeo, tracking y minimapa que los vídeos de revisión y, arriba a la
derecha, **dos letras grandes: `GT` y `modelo`**. Cuando no coinciden, las dos llevan un **marco
amarillo**. Además imprime cuántos frames difieren y el reparto H/A/D de cada uno:

```
ventana publica: frames 11185-18684 (7500 frames), distintos del GT: 1158 (15.4 %)
  H: GT  3454   modelo  3824
  A: GT  2577   modelo  2651
  D: GT  1469   modelo  1025
```

Como el nombre del vídeo lleva el de la configuración, cada variante tiene el suyo y se pueden
comparar. `--out ruta.mp4` cambia el destino. Tarda ~2 minutos. Usa las opciones de la sección
`video` de la configuración (`mode`, `scale`, `minimap`), aunque `video.enabled` esté en `false`.

**Para qué sirve y para qué no:** CHALLENGE.md presenta estos 5 minutos como la referencia de cómo
se etiqueta cada situación (cuándo empieza una D, cuándo hay cambio de posesión). Sirve para
entender la definición y convertir lo que se vea en reglas generales, que luego se comprueban en
todo el partido. **No** sirve para ir tocando parámetros hasta que desaparezcan los marcos
amarillos: son 17 posesiones, ajustar con ellas está prohibido (RULES.md) y empeora el resultado
en el partido completo.

También se puede activar en los vídeos de bloques con `video.public_gt: true` (solo tiene sentido
en `1T_01`; el vídeo se guarda con el sufijo `_GT`).

## Datos

Los datos del partido no están ni pueden estar en este repositorio (`.gitignore` los excluye,
también los vídeos `.mp4`).
