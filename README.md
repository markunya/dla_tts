# Text-to-Speech на базе HiFi-GAN

Данный проект реализует фреймворк для обучения нейросетевой модели **HiFi-GAN**, а также для инференса **text-to-speech (TTS)** с использованием предобученной модели **Tacotron2**.

---

## Установка

```bash
git clone https://github.com/markunya/dla_tts.git
cd dla_tts
conda create -n dla_tts python=3.10.18
conda activate dla_tts
pip install -r requirements.txt
```

---

## Загрузка весов моделей

### HiFi-GAN

Для загрузки весов обученного генератора HiFi-GAN выполните:

```bash
mkdir -p weights
gdown 1MSrxOx3Ek3nLBl97ZpU4DU7fgrhaG47J -O weights/hifigen.ckpt
```

### Модели для нейросетевого MOS

Для загрузки весов моделей, необходимых для подсчёта нейросетевого MOS, выполните:

```bash
bash src/metrics/download_weights.sh
```

---

## Analysis Notebook

В файле `analysis.ipynb` представлен сравнительный анализ сгенерированных аудиосэмплов.

[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/markunya/dla_tts/blob/main/analysis.ipynb)

---

## Demo Notebook

Файл `demo.ipynb` демонстрирует:

* выполнение инференса на тестовом наборе;
* вычисление метрик качества;
* подключение пользовательского датасета через ссылку на Yandex Drive.

[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/markunya/dla_tts/blob/main/demo.ipynb)
