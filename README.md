# Projet SHARP

Un système de reconnaissance et comptage de doigts levés en temps réel. L'application utilise Vue.js pour le frontend et un modèle YOLO entraîné pour la détection des mains via la webcam.

## Table des matières

- [Fonctionnalités](#fonctionnalités)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Technologies utilisées](#technologies-utilisées)

## Fonctionnalités

- Flux vidéo en temps réel depuis la webcam
- Détection et reconnaissance des mains
- Comptage automatique des doigts levés
- Interface utilisateur avec Tailwind CSS et DaisyUI
- Capture d'images à intervalle régulier
- Affichage des résultats de détection avec niveau de confiance

---

## Structure du projet

```
Projet-ia/
├── python/                          # Scripts de préparation du dataset
│   └── dataset/                     # Dataset généré au format YOLO
│       ├── dataset.yaml             # Configuration du dataset
│       ├── images/
│       │   ├── train/               # Images d'entraînement
│       │   └── val/                 # Images de validation
│       └── labels/
│           ├── train/               # Annotations (format YOLO)
│           └── val/
│
└── website/                         # Application Vue.js
    ├── src/
    │   ├── App.vue                  # Composant principal
    │   ├── main.js                  # Point d'entrée Vue
    │   ├── style.css                # Styles globaux
    │   ├── components/
    │   │   ├── CameraBox.vue        # Conteneur de la webcam
    │   │   └── DataSidebar.vue      # Barre latérale avec résultats
    │   └── composables/
    │       └── useVision.js         # Logique de vision par ordinateur
    ├── vite.config.js               # Configuration Vite
    ├── package.json                 # Dépendances npm
    └── index.html
```

---

## Prérequis

- **Node.js** v18+ (pour le frontend)
- **Python** 3.8+ (pour la préparation du dataset)
- **Webcam** (pour le flux vidéo en direct)

## Node.js v18+

- Python 3.8+
- UnPréparer le dataset

```bash
cd python
python convert.py
```

Cela va créer la structure de dossiers, convertir les données du fichier `handsyolo26.ndjson` au format YOLO et télécharger les images.

### Configurer le frontend

```bash
cd website
npm install
```

### Lancer l'application

```bash
npm run dev
```

L'application sera disponible sur `http://localhost:5173`

## Utilisation

1. Ouvrir l'application dans un navigateur
2. Autoriser l'accès à la webcam
3. Placer votre main devant la caméra

L'app affichera le flux vidéo, les mains détectées et le nombre de doigts levés avec le niveau de confiance de la prédiction.

```bash
# Développement
npm run dev

# Production (build)
npm run build
```

Autre commande

```bash
npm run preview  # Prévisualiser la build
```

##

Modifiez ces valeurs dans [src/composables/useVision.js](src/composables/useVision.js) :

```javascript
const CAPTURE_INTERVAL_MS = 500; // Délai entre les captures (en ms)
```

---

## Architecture

### Frontend (Vue.js + Vite)

- **App.vue** : Composant racine orchestrant l'interface
- **CameraBox.vue** : Affiche le flux vidéo et l'overlay de détection
- **DataSidebar.vue** : Affiche les résultats et statistiques
- **useVision.js** : Composable gérant :
  - La capture vidéo
  - L'envoi au serveur d'IA
    Vous pouvez modifier l'intervalle de capture dans `src/composables/useVision.js` :

```javascript
const CAPTURE_INTERVAL_MS = 500; // en millisecondes
```

## Technologies utilisées

- Vue 3
- Vite
- Tailwind CSS
- DaisyUI
- Python 3
- YOLO
