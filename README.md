# Projet Deep Learning – EMSI Casablanca 2025–2026

<div align="center">

![EMSI](https://img.shields.io/badge/EMSI-Casablanca-1B3A6B?style=for-the-badge)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-Academic-C8A951?style=for-the-badge)

**Étudiant :** Salaheddine Elberrani  
**Filière :** Informatique – 4IIR  
**Module :** Deep Learning  
**Année universitaire :** 2025–2026

</div>

---

## Idée générale du projet

Ce projet de fin de module constitue l'évaluation finale du cours de Deep Learning à l'EMSI Casablanca. L'objectif principal est de démontrer la capacité à **adapter le choix d'une architecture de deep learning à la nature des données étudiées**.

Le projet couvre trois grandes familles de modèles implémentées sous **PyTorch** :

| Partie | Architecture | Données | Dataset |
|--------|-------------|---------|---------|
| **I** | MLP – Perceptron Multicouche | Tabulaires | Wine Quality Red (UCI) |
| **II** | CNN – Réseau Convolutionnel | Images | Fashion-MNIST |
| **III** | RNN / LSTM / GRU / Seq2Seq | Séquences | fra-eng (Tatoeba) |

Chaque partie inclut :
- Une étude théorique structurée
- Une implémentation PyTorch complète et commentée
- Une étude expérimentale comparative
- Une analyse critique des résultats
- Une réponse à une question de synthèse sur dataset réel

---

## Structure du repository

```
Deep-Learning-EMSI-2025/
│
├── README.md                          ← Ce fichier
│
├── notebooks/
│   ├── Partie_I_MLP_PyTorch.ipynb     ← MLP sur Wine Quality Red
│   ├── Partie_II_CNN_Vision.ipynb     ← CNN sur Fashion-MNIST
│   └── Partie_III_RNN_Seq2Seq.ipynb  ← RNN/LSTM/GRU/Seq2Seq sur fra-eng
│
├── figures/
│   ├── partie1_distribution_poids.png     ← Distribution poids selon initialisation
│   ├── partie1_comparaison_init.png       ← Courbes convergence initialisations
│   ├── partie2_ablation.png               ← Ablation architectures CNN
│   ├── partie2_mlp_vs_cnn.png             ← Comparaison MLP vs CNN
│   ├── partie3_gradient_clipping.png      ← Effet du gradient clipping
│   ├── partie3_rnn_lstm_gru.png           ← PPL comparative RNN/LSTM/GRU
│   ├── partie3_seq2seq.png                ← Courbes entraînement Seq2Seq
│   ├── partie3_resultats_finaux.png       ← BLEU Greedy vs Beam Search
│   └── synthese_globale.png               ← Synthèse des 3 parties
│
└── docs/
    ├── Rapport_Deep_Learning_Salaheddine_Elberrani.docx
    └── Presentation_Deep_Learning_Salaheddine_Elberrani.pptx
```

---

## Contenu détaillé des notebooks

### 📓 Partie I – MLP et ingénierie PyTorch
**Fichier :** `notebooks/Partie_I_MLP_PyTorch.ipynb`

**Dataset :** Wine Quality Red (UCI) — 1 599 échantillons, 11 features, 6 classes

**Contenu :**
- Concepts fondamentaux : `nn.Module`, `state_dict`, `named_parameters`, device
- Pipeline de données : nettoyage, encodage, normalisation StandardScaler, DataLoaders
- Deux implémentations MLP : `nn.Sequential` vs classe personnalisée
- Inspection des paramètres avec `named_parameters()` et `state_dict()`
- Comparaison de 3 stratégies d'initialisation : Gaussienne, Constante, Xavier
- Sauvegarde et rechargement du meilleur modèle (early stopping)
- Gestion CPU/GPU avec vérification de cohérence
- Évaluation : accuracy, précision, rappel, F1-score, matrice de confusion

**Résultats obtenus :**
| Stratégie | Accuracy test | F1-score |
|-----------|--------------|---------|
| Gaussienne | 0.562 | 0.548 |
| Constante | 0.523 | 0.509 |
| **Xavier** | **0.611** | **0.598** |

---

### 📓 Partie II – CNN et vision par ordinateur
**Fichier :** `notebooks/Partie_II_CNN_Vision.ipynb`

**Dataset :** Fashion-MNIST — 70 000 images 28×28, 10 classes

**Contenu :**
- Théorie CNN : localité, partage des poids, hiérarchie des représentations
- Calculs manuels : taille de sortie en convolution et après pooling
- Implémentation manuelle : corrélation croisée 2D, max-pooling, average-pooling
- Comparaison avec les couches PyTorch correspondantes
- CNN inspiré de LeNet avec BatchNorm, Dropout, conv 1×1 configurable
- Étude expérimentale : padding, stride, pool_type, n_filters, conv 1×1
- Visualisation des feature maps (Conv1 et Conv2) via hooks PyTorch
- Comparaison MLP vs CNN sur le même dataset

**Résultats obtenus :**
| Configuration | Params | Accuracy test |
|--------------|--------|--------------|
| MLP baseline | 535 818 | 0.843 |
| Base (MaxPool, 6f) | 61 806 | 0.869 |
| MaxPool, 12f | 100 014 | 0.880 |
| **+ conv 1×1** | **104 526** | **0.889** |

> Le CNN atteint +4.6 pts d'accuracy avec ×5 moins de paramètres que le MLP.

---

### 📓 Partie III – RNN, LSTM, GRU et Seq2Seq
**Fichier :** `notebooks/Partie_III_RNN_Seq2Seq.ipynb`

**Dataset :** fra-eng Tatoeba — 12 000 paires de phrases anglais-français

**Contenu :**
- Modèle de langage : objectif probabiliste, règle de chaîne, perplexité
- BPTT et illustration expérimentale du gradient clipping
- Implémentation RNN simple, LSTM et GRU dans une classe générique
- Comparaison : stabilité, mémoire, perplexité, nombre de paramètres
- Pipeline de données : tokenisation, vocabulaire, tokens spéciaux, padding
- Système Seq2Seq encodeur-décodeur avec teacher forcing dégressif
- Décodage glouton (Greedy Search) et Beam Search (k=3, length penalty)
- Évaluation : perplexité et score BLEU

**Résultats obtenus :**
| Modèle | Params | PPL val |
|--------|--------|---------|
| RNN simple | 1 192 388 | 59.0 |
| GRU | 1 652 164 | 35.9 |
| **LSTM** | **1 882 052** | **29.5** |

| Décodage | BLEU-1 | BLEU-2 |
|----------|--------|--------|
| Greedy Search | 74.80% | 43.99% |
| **Beam Search k=3** | **84.74%** | **64.82%** |

---

## Prérequis et installation

### Environnement recommandé
- **Google Colab** (GPU T4 gratuit, PyTorch préinstallé) — recommandé
- Python 3.10+, PyTorch 2.x

### Installation locale
```bash
pip install torch torchvision scikit-learn pandas matplotlib seaborn numpy
```

### Datasets (téléchargement automatique)
| Partie | Dataset | Source | Taille |
|--------|---------|--------|--------|
| I | Wine Quality Red | UCI (auto) | 84 Ko |
| II | Fashion-MNIST | torchvision (auto) | 30 Mo |
| III | fra-eng Tatoeba | manythings.org (auto) | 6 Mo |

Tous les datasets se téléchargent automatiquement à la première exécution.

---

## Exécution sur Google Colab

1. Ouvrir [colab.research.google.com](https://colab.research.google.com)
2. **Activer le GPU :** `Exécution → Modifier le type d'exécution → GPU T4`
3. Importer le notebook : `Fichier → Importer le notebook`
4. Exécuter toutes les cellules : `Exécution → Tout exécuter`

**Temps d'exécution estimé (GPU T4) :**
- Partie I : ~5 minutes
- Partie II : ~15 minutes
- Partie III : ~25 minutes

---

## Résultats clés

```
┌─────────────────────────────────────────────────────────────┐
│  Partie I  – MLP  : Accuracy = 0.611  |  F1 = 0.598        │
│  Partie II – CNN  : Accuracy = 0.889  (+4.6 pts vs MLP)    │
│  Partie III– LSTM : PPL = 29.5  (meilleure perplexité)     │
│  Seq2Seq + Beam   : BLEU-1 = 84.7%  (+9.9 pts vs Greedy)  │
└─────────────────────────────────────────────────────────────┘
```

---

## Références

- LeCun, Y. et al. (1998). *Gradient-based learning applied to document recognition.* Proceedings of the IEEE.
- Hochreiter, S., Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation.
- Cho, K. et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder.* EMNLP.
- Bahdanau, D. et al. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR.
- Glorot, X., Bengio, Y. (2010). *Understanding the difficulty of training deep feedforward neural networks.* AISTATS.
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- [Fashion-MNIST – Zalando Research](https://github.com/zalandoresearch/fashion-mnist)
- [Tatoeba Project – fra-eng](https://www.manythings.org/anki/)

---

<div align="center">

**EMSI Casablanca – École Marocaine des Sciences de l'Ingénieur**  
Projet de fin de module – Deep Learning – 2025–2026  
*Salaheddine Elberrani – 4IIR*

</div>
