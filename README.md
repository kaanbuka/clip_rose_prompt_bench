# CLIP Zero-Shot Rose Classification

> Prompt-strategy benchmarking for zero-shot classification of three visually similar rose species using OpenAI CLIP.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-ee4c2c.svg)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Model-CLIP%20ViT--B%2F32-orange.svg)](https://github.com/openai/CLIP)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

🇬🇧 [English](#english) · 🇹🇷 [Türkçe](#türkçe)

---

## English

### Overview

This project benchmarks **prompt engineering strategies** for zero-shot image classification with OpenAI's CLIP model on three botanically similar rose species. Five strategies (`simple`, `descriptive`, `scientific`, `contextual`, `ensemble`) are compared end-to-end: text-prompt design → CLIP encoding → cosine-similarity scoring → metric reporting → publication-ready plots → interactive Streamlit dashboard.

The goal is to quantify how much downstream accuracy depends on prompt phrasing alone, with no model fine-tuning.

### Key Features

- **Zero-shot pipeline** - no training required; classes defined entirely by natural-language prompts.
- **Five prompt strategies** with a built-in ensemble that averages embeddings across all strategies.
- **Reproducible reporting** - CSV summary, per-image scores, confusion matrices, and four plot types.
- **Interactive dashboard** - Streamlit UI for browsing accuracy, confusion matrices, and per-image scores.
- **Configurable backbone** - switch between `ViT-B/32`, `ViT-L/14`, or `RN50` via a single CLI flag.

### Classes

| Key | Common name | Turkish |
|---|---|---|
| `rosa_canina` | Dog Rose | Kuşburnu |
| `rosa_damascena` | Damask Rose | Isparta Gülü |
| `rosa_multiflora` | Multiflora Rose | Çok Çiçekli Gül |

### Prompt Strategies

| Strategy | Description |
|---|---|
| `simple` | Minimal label-only prompts (`"a photo of a rosa canina"`). |
| `descriptive` | Visual attributes - colour, petal count, structure. |
| `scientific` | Botanical / taxonomic phrasing with Latin binomials. |
| `contextual` | Habitat, scene, and end-use cues (perfumery, hedgerows, …). |
| `ensemble` | Mean of L2-normalised class embeddings from all four strategies. |

### Current Benchmark

From [`results/summary_metrics.csv`](results/summary_metrics.csv):

| Strategy    | Accuracy | Macro F1 | Macro Precision | Macro Recall |
|-------------|---------:|---------:|----------------:|-------------:|
| contextual  |   0.7667 |   0.7677 |          0.7863 |       0.7667 |
| scientific  |   0.7000 |   0.6932 |          0.7238 |       0.7000 |
| ensemble    |   0.7000 |   0.6934 |          0.8061 |       0.7000 |
| simple      |   0.6333 |   0.6278 |          0.8254 |       0.6333 |
| descriptive |   0.5333 |   0.5000 |          0.6667 |       0.5333 |

> Backbone: CLIP ViT-B/32 · 30 images (10 per class) · seed 42.

### Project Structure

```text
clip_rose_classifier/
├── clip_classifier.py      # Core CLIP inference pipeline
├── prompt_strategies.py    # Prompt sets per strategy
├── evaluate.py             # Metrics + CSV/TXT report generation
├── visualize.py            # Accuracy / confusion / score plots
├── dashboard.py            # Streamlit dashboard
├── main.py                 # CLI entry point
├── config.py               # Shared settings
├── requirements.txt
├── data/
│   └── sample_images/      # Input images organised per class
│       ├── rosa_canina/
│       ├── rosa_damascena/
│       └── rosa_multiflora/
└── results/                # Generated outputs (CSV + PNG)
```

### Installation

```bash
git clone https://github.com/<your-username>/clip_rose_classifier.git
cd clip_rose_classifier

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> The OpenAI CLIP package is installed directly from GitHub (see `requirements.txt`).

### Usage

**1. Add images** under each class folder:

```text
data/sample_images/
├── rosa_canina/         *.jpg | *.jpeg | *.png | *.webp
├── rosa_damascena/
└── rosa_multiflora/
```

**2. Run the experiment:**

```bash
python main.py
```

Optional flags:

```bash
python main.py \
  --image_dir data/sample_images \
  --results_dir results \
  --model ViT-B/32
```

**3. Launch the dashboard:**

```bash
streamlit run dashboard.py
```

### Output Files

| File | Purpose |
|---|---|
| `results/summary_metrics.csv` | Per-strategy accuracy + macro precision/recall/F1. |
| `results/per_image_scores.csv` | Softmax score per (image, strategy, class). |
| `results/confusion_matrices.txt` | Plain-text confusion matrices per strategy. |
| `results/accuracy_by_strategy.png` | Ranked accuracy bar chart. |
| `results/confusion_matrix_<strategy>.png` | Row-normalised heatmap per strategy. |
| `results/score_heatmap.png` | Mean correct-class softmax score, strategy × class. |
| `results/score_distribution.png` | Violin plot of correct-class score distributions. |

### Methodology

1. Build per-class text embeddings: tokenize every prompt, encode with CLIP's text tower, L2-normalise, then mean-pool per class.
2. For the **ensemble**, repeat step 1 for every strategy and average the resulting class-embedding tensors before re-normalising.
3. For each image: encode with CLIP's vision tower, L2-normalise, and compute cosine similarity against each strategy's class embeddings.
4. Apply temperature scaling (×100) and softmax to obtain per-class probabilities.
5. Aggregate predictions and produce metrics + plots.

### Notes

- Results vary with dataset composition and image quality - keep class counts balanced for fair comparisons.
- Larger backbones (`ViT-L/14`) usually improve accuracy at the cost of memory and latency.
- Prompt phrasing is the primary lever here; no weight updates are performed.

### References

- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (2021).
- [OpenAI CLIP repository](https://github.com/openai/CLIP).

### License

Released under the [MIT License](LICENSE).

---

## Türkçe

### Genel Bakış

Bu proje, OpenAI CLIP modeli ile **prompt mühendisliği stratejilerini** botanik açıdan birbirine benzeyen üç gül türü üzerinde sıfır-atış (zero-shot) görüntü sınıflandırma için kıyaslar. Beş strateji (`simple`, `descriptive`, `scientific`, `contextual`, `ensemble`) uçtan uca karşılaştırılır: metin promptu tasarımı → CLIP kodlaması → kosinüs benzerliği → metrik raporlama → yayınlanabilir grafikler → etkileşimli Streamlit paneli.

Amaç, model ince-ayarı yapmadan yalnızca prompt ifadesinin doğruluğa ne kadar etki ettiğini ölçmektir.

### Öne Çıkan Özellikler

- **Sıfır-atış akış** - eğitim gerektirmez; sınıflar tamamen doğal dildeki promptlar ile tanımlanır.
- **Beş prompt stratejisi** ve tüm stratejilerin gömme vektörlerini ortalayan dahili bir **ensemble**.
- **Yeniden üretilebilir raporlar** - CSV özeti, görüntü başına skorlar, karışıklık matrisleri ve dört farklı grafik.
- **Etkileşimli panel** - doğruluk, karışıklık matrisleri ve görüntü skorlarını incelemek için Streamlit arayüzü.
- **Yapılandırılabilir omurga** - tek bir CLI bayrağı ile `ViT-B/32`, `ViT-L/14` veya `RN50` arasında geçiş.

### Sınıflar

| Anahtar | İngilizce | Türkçe |
|---|---|---|
| `rosa_canina` | Dog Rose | Kuşburnu |
| `rosa_damascena` | Damask Rose | Isparta Gülü |
| `rosa_multiflora` | Multiflora Rose | Çok Çiçekli Gül |

### Prompt Stratejileri

| Strateji | Açıklama |
|---|---|
| `simple` | Sade, etiket düzeyinde promptlar (`"a photo of a rosa canina"`). |
| `descriptive` | Görsel özellikler - renk, taç yaprak sayısı, yapı. |
| `scientific` | Botanik / taksonomik dil ve Latince binom adlandırma. |
| `contextual` | Habitat, sahne ve kullanım alanı ipuçları (parfümeri, çitler, …). |
| `ensemble` | Dört stratejinin L2-normalize sınıf gömme vektörlerinin ortalaması. |

### Mevcut Sonuçlar

[`results/summary_metrics.csv`](results/summary_metrics.csv) dosyasından:

| Strateji    | Doğruluk | Macro F1 | Macro Precision | Macro Recall |
|-------------|---------:|---------:|----------------:|-------------:|
| contextual  |   0.7667 |   0.7677 |          0.7863 |       0.7667 |
| scientific  |   0.7000 |   0.6932 |          0.7238 |       0.7000 |
| ensemble    |   0.7000 |   0.6934 |          0.8061 |       0.7000 |
| simple      |   0.6333 |   0.6278 |          0.8254 |       0.6333 |
| descriptive |   0.5333 |   0.5000 |          0.6667 |       0.5333 |

> Omurga: CLIP ViT-B/32 · 30 görüntü (sınıf başı 10) · tohum 42.

### Proje Yapısı

```text
clip_rose_classifier/
├── clip_classifier.py      # Çekirdek CLIP çıkarım hattı
├── prompt_strategies.py    # Strateji başına prompt setleri
├── evaluate.py             # Metrikler + CSV/TXT rapor üretimi
├── visualize.py            # Doğruluk / karışıklık / skor grafikleri
├── dashboard.py            # Streamlit paneli
├── main.py                 # CLI giriş noktası
├── config.py               # Ortak ayarlar
├── requirements.txt
├── data/
│   └── sample_images/      # Sınıf başına klasörlenmiş giriş görselleri
│       ├── rosa_canina/
│       ├── rosa_damascena/
│       └── rosa_multiflora/
└── results/                # Üretilen çıktılar (CSV + PNG)
```

### Kurulum

```bash
git clone https://github.com/<kullanici-adi>/clip_rose_classifier.git
cd clip_rose_classifier

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> OpenAI CLIP paketi doğrudan GitHub'dan kurulur (bkz. `requirements.txt`).

### Kullanım

**1. Görselleri** ilgili sınıf klasörlerine yerleştirin:

```text
data/sample_images/
├── rosa_canina/         *.jpg | *.jpeg | *.png | *.webp
├── rosa_damascena/
└── rosa_multiflora/
```

**2. Deneyi çalıştırın:**

```bash
python main.py
```

İsteğe bağlı bayraklar:

```bash
python main.py \
  --image_dir data/sample_images \
  --results_dir results \
  --model ViT-B/32
```

**3. Paneli başlatın:**

```bash
streamlit run dashboard.py
```

### Çıktı Dosyaları

| Dosya | Amaç |
|---|---|
| `results/summary_metrics.csv` | Strateji başına doğruluk + macro precision/recall/F1. |
| `results/per_image_scores.csv` | (Görüntü, strateji, sınıf) bazında softmax skoru. |
| `results/confusion_matrices.txt` | Strateji başına metin formatında karışıklık matrisleri. |
| `results/accuracy_by_strategy.png` | Sıralı doğruluk çubuk grafiği. |
| `results/confusion_matrix_<strateji>.png` | Strateji başına satır-normalize karışıklık ısı haritası. |
| `results/score_heatmap.png` | Doğru sınıf için ortalama softmax skoru, strateji × sınıf. |
| `results/score_distribution.png` | Doğru sınıf skor dağılımının violin grafiği. |

### Metodoloji

1. Her sınıf için metin gömme vektörleri oluştur: tüm promptları tokenize et, CLIP metin kulesiyle kodla, L2-normalize et, sınıf bazında ortalamasını al.
2. **Ensemble** için, 1. adımı her strateji için tekrarla, sonuçtaki sınıf gömme tensörlerini ortala ve yeniden normalize et.
3. Her görüntü için: CLIP görüntü kulesiyle kodla, L2-normalize et ve her stratejinin sınıf gömmelerine karşı kosinüs benzerliği hesapla.
4. Sıcaklık ölçeklemesi (×100) ve softmax uygulayarak sınıf olasılıklarını üret.
5. Tahminleri toplulaştır, metrikleri ve grafikleri üret.

### Notlar

- Sonuçlar veri kümesinin yapısına ve görüntü kalitesine bağlı olarak değişir - adil karşılaştırma için sınıf sayıları dengeli tutulmalıdır.
- Daha büyük omurgalar (`ViT-L/14`) genelde doğruluğu artırır; karşılığında bellek ve gecikme maliyeti yükselir.
- Buradaki temel kaldıraç prompt ifadesidir; herhangi bir ağırlık güncellemesi yapılmaz.

### Kaynaklar

- Radford ve diğerleri, [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (2021).
- [OpenAI CLIP deposu](https://github.com/openai/CLIP).

### Lisans

Bu proje [MIT Lisansı](LICENSE) altında yayınlanmıştır.
