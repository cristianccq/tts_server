## Sintetizador de voces (DCTTS y TACOTRON)

* Clonar este repositorio
* Dependencias :
	- Python 3
	- librosa
	- falcon (para el demo server)

* Las oraciones a sintetizar se guardan en : MOS/sents/es.txt
   Modificar para producir nuevos audios.

## Usar DCTTS

* Entrar a la carpeta dc_tts
* ejecutar `python synthesize.py`


## Usar Tacotron

* Entrar a la carpeta tacotron
* ejecutar `python synthesize.py`


Los resultados se guardan en la carpeta "es/samples"

## TTS SERVER

* instalar dependencias como falcon
* ejecutar demo_server.py

Ir a la web (http://localhost:9000/)