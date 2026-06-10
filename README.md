# Projet SHARP — Smart Hand Automated Recognition Project

Détection en temps réel du nombre de doigts levés (une ou plusieurs mains) avec
**YOLO11** (Ultralytics). Le projet est en deux parties :

- **A. Pipeline de Training** ([`training/`](training/)) — extraction, validation,
  préparation, entraînement et évaluation du modèle.
- **B. Application de Serving** ([`serving/`](serving/) + [`website/`](website/)) —
  un backend qui charge le modèle YOLO et l'expose via une API, et un dashboard
  webcam temps réel, le tout **dockerisé**.

Les six classes sont fixées par le sujet et partagées par les deux parties :

```
0_doigt  1_doigt  2_doigts  3_doigts  4_doigts  5_doigts
```

Si deux mains sont visibles, chaque main a sa propre bounding box ; le dashboard
affiche la **somme** des doigts de toutes les mains détectées.

## Structure du dépôt

```
training/            Pipeline ML (package Python `sharp`, local + cloud Platform)
  sharp/local/       extraction → validation → préparation → training → évaluation
  sharp/cloud/       pipeline Ultralytics Platform (cloud)
  scripts/           outil optionnel de conversion Labelbox NDJSON → YOLO
  tests/             tests unitaires (pytest)
serving/backend/     API FastAPI qui charge best.pt et expose /predict
website/             dashboard webcam (Vue 3 + Vite), servi par nginx en prod
docker-compose.yml   stack de serving (backend + frontend)
```

---

## A. Pipeline de Training

Détails complets dans [`training/README.md`](training/README.md). En résumé :

```bash
cd training
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .secrets.toml.example .secrets.toml   # y mettre la clé Ultralytics Platform (ul_…)
```

Configuration via [`settings.toml`](training/settings.toml) (dynaconf) ou variables
d'environnement `SHARP_*`, ou en argument CLI (`--dataset`, …).

```bash
# Pipeline locale complète : extraction → validation → préparation → train → éval
python run.py local --dataset <slug_ou_ul://_URI>

# Pipeline cloud (Ultralytics Platform) : entraînement local+streaming puis download
python run.py cloud train
python run.py cloud download --model-id <MODEL_ID> --dest runs/cloud/best.pt
```

Points clés :

- **Extraction** : dataset Ultralytics Platform résolu via son URI `ul://` (slug
  configurable), ou un dossier local déjà présent.
- **Validation** : images réellement décodées (détecte les fichiers tronqués) ;
  labels vérifiés (classe valide, pas de coordonnées négatives, valeurs `[0,1]`,
  box entièrement dans l'image, pas de box d'aire nulle).
- **Préparation** : réutilise un split déjà présent dans l'export, sinon génère un
  split déterministe **60/20/20, seed 42** ; `data.yaml` généré dynamiquement.
- **Training** : YOLO11, augmentations explicites dans `settings.toml` (`[augment]`).
- **Évaluation** : `model.val(split="test")` ; métriques (mAP/P/R, FPS) écrites dans
  `runs/.../test_metrics.json` en plus des courbes Ultralytics sous `runs/`.

---

## B. Application de Serving (dockerisée)

### Lancer la stack

1. Placer les poids entraînés dans `serving/models/best.pt`
   (ou définir `SHARP_MODEL_URL` sur un URI `ul://` Platform, avec `SHARP_API_KEY`,
   pour les télécharger au démarrage ; les URLs HUB legacy marchent jusqu'à fin
   juillet 2026).
2. Démarrer :

   ```bash
   docker compose up --build
   ```

3. Ouvrir le dashboard sur **http://localhost:8080** et autoriser la webcam.

### Architecture

- **backend** ([`serving/backend/app.py`](serving/backend/app.py)) — FastAPI +
  Ultralytics. Charge le modèle **au démarrage** et expose :
  - `GET /health` — liveness + état du modèle ;
  - `POST /predict` — reçoit une frame (`multipart/form-data`, champ `file`) et
    renvoie les détections au format `{"images": [{"results": [...]}]}`.
- **frontend** ([`website/`](website/)) — SPA Vue 3 buildée puis servie par nginx,
  qui **proxifie** `/predict` vers le backend (cf. [`website/nginx.conf`](website/nginx.conf)).
  La clé API ne transite donc jamais par le navigateur.

Le dashboard superpose les bounding boxes + classes sur le flux et affiche la
somme des doigts détectés.

### Configuration (variables d'environnement)

| Variable           | Défaut             | Rôle |
|--------------------|--------------------|------|
| `SHARP_MODEL_PATH` | `/models/best.pt`  | poids montés dans le conteneur |
| `SHARP_MODEL_URL`  | *(vide)*           | URI `ul://` Platform (ou URL HUB legacy) pour télécharger le modèle si absent |
| `SHARP_API_KEY`    | *(vide)*           | si défini, exigé en en-tête `x-api-key` |
| `SHARP_CONF`       | `0.25`             | seuil de confiance |
| `SHARP_IOU`        | `0.45`            | seuil NMS IoU |
| `SHARP_IMGSZ`      | `640`              | taille d'inférence |

> ⚠️ **Sécurité** : ne jamais committer de clé. Avec Vite, toute valeur
> `VITE_*` est inlinée dans le bundle public. Le dashboard local n'a pas besoin de
> clé (proxy nginx). Si une clé a déjà été publiée, **la révoquer/régénérer** côté
> Ultralytics.

### Démo statique GitHub Pages (optionnelle)

En plus du serving dockerisé, le frontend peut être déployé en site statique. Il
appelle alors directement un endpoint d'inférence distant (Ultralytics Platform), donc
la clé est **inévitablement exposée** dans le bundle public — à n'utiliser qu'avec
une clé dédiée et régénérable.

```bash
cd website
VITE_BASE=/Projet-ia/ VITE_ENDPOINT=<url_inference> VITE_API_KEY=<cle> npm run build
```

---

## Qualité du code & CI

```bash
cd training
ruff check .        # lint
black --check .     # format
pytest -q           # tests unitaires
```

- Type hints + docstrings (style Google), `ruff` + `black`, `.pre-commit-config.yaml`.
- CI GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) :
  lint + format + tests à chaque Pull Request.
