# EmberWatch — Guide M2 expliqué pas à pas

> Responsable : **Mouhib** (Rôle A — Computer Vision Lead) · Sprint 1 (semaines 2–3)
> Ce guide suit le cahier des charges (CDC) dans l'ordre. Pour chaque étape : **la commande**, **ce qu'elle fait**, **pourquoi on la fait**, et **pourquoi ce choix plutôt qu'un autre**.

---

## 0. Où on en est dans le CDC

| Étape du CDC | Section | Statut |
|---|---|---|
| Contrat de données (format des alertes) | §8, §8.1 | ✅ `src/detection/contract.py` |
| M1 — Dataset propre, splits sans fuite, rapport de fuite | §7 M1, §9.4 | ✅ `prepare_dataset.py` (partie détection) |
| M2 — Baseline YOLO + tuiles SAHI | §7 M2 | ✅ code prêt, **entraînement réel à lancer sur Kaggle** |
| M2 — mAP@0.5 sur le test (objectif ≥ 0.55) | §11 | ⏳ après l'entraînement Kaggle |
| M2 — Précision / rappel image par image sur FIgLib test | §7 M2 | ⏳ fin Sprint 1 (besoin de FIgLib, préparé par Ghofrane) |
| Débit GPU / CPU | §7 M2 | ✅ mesuré par `evaluate.py` |
| API qui renvoie une alerte au format du contrat | §8, §10 | ✅ `POST /api/v1/detect` |
| Confirmation humaine obligatoire (P1) | §13 | ✅ toute alerte = `PENDING_CONFIRMATION` |
| Tests automatiques | §10 (qualité du code) | ✅ 11 tests |
| M3 — Détecteur spatio-temporel | §7 M3 | ⏭️ Sprint 2 |

---

## 1. Les datasets dont j'ai besoin

| Dataset | Pour quoi | Quand | Lien |
|---|---|---|---|
| **D-Fire** | Entraîner YOLO (≈ 21 000 images, boîtes fumée + feu) | Sprint 1 | https://github.com/gaiasd/DFireDataset (liens de téléchargement dans le README) · miroir : https://universe.roboflow.com/uweeaai/d-fire-ehbj2 |
| **HPWREN / AI For Mankind** | Boîtes de fumée vues depuis les **mêmes caméras que FIgLib** | Sprint 1 | https://public.roboflow.com/object-detection/wildfire-smoke (export « YOLOv8 ») |
| **FIgLib** | Séquences d'ignition : test image par image (Sprint 1), puis M3 (Sprint 2) | Fin Sprint 1 | https://www.hpwren.ucsd.edu/FIgLib/ · miroir : https://wifire-data.sdsc.edu/dataset/hpwren-fire-ignition-library |
| PYRONEAR-2025 *(optionnel)* | Test de généralisation sur caméras européennes | Sprint 4 | https://arxiv.org/pdf/2402.05349 |

**Articles à lire en priorité :**
- FIgLib & SmokeyNet (base du M3) : https://arxiv.org/pdf/2112.08598
- Comparaison de détecteurs YOLO sur feu / fumée : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11398105/

**Datasets des autres membres :**
- Achref : NDWS https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread · WildfireSpreadTS https://zenodo.org/records/8006177 · Mesogeos https://github.com/Orion-AI-Lab/mesogeos · NASA FIRMS https://firms.modaps.eosdis.nasa.gov/ · Copernicus (Sentinel-2, DEM) https://dataspace.copernicus.eu/
- Ghofrane : FIgLib (même lien que ci-dessus).

> ⚠️ Le CDC dit que les tailles sont « approximatives — à vérifier en semaine 1 ». Vérifie la **licence** et la **taille réelle** de chaque dataset avant de le télécharger.

---

## 2. Étape par étape

Les commandes sont écrites pour **Windows (cmd)**, comme le README d'Achref. Sur Kaggle / Linux, remplace `.venv\Scripts\python.exe` par `python`.

### Étape 1 — Récupérer le code dans `C:\projects\EmberWatch`

```cmd
cd C:\projects\EmberWatch
git clone https://github.com/achrefchaabene/EmberWatch.git .
git checkout -b m2-detection
git am C:\chemin\vers\0001-m2-smoke-detector.patch
git log --oneline -2
```

- **`cd C:\projects\EmberWatch`** : on se place dans le dossier que tu as créé.
- **`git clone ... .`** : le **point final** veut dire « clone ici, dans ce dossier », au lieu de créer un sous-dossier `C:\projects\EmberWatch\EmberWatch`. Le dossier doit être **vide**, sinon git refuse.
- **`git checkout -b m2-detection`** : crée une **branche** pour ton travail.
  - *Pourquoi une branche :* le CDC (§17, gouvernance) demande des pull requests relues. Travailler directement sur `main` risquerait de casser le travail d'Achref (M6).
  - *Pourquoi pas un dépôt séparé :* le CDC veut **un seul dépôt** (livrable D1) et une plateforme intégrée. Deux dépôts = intégration douloureuse à la fin.
- **`git am <patch>`** : applique le commit M2 (code, tests, notebook, ce guide) **avec son message et son auteur** (toi). Remplace `C:\chemin\vers\` par le dossier où tu as téléchargé le patch.
  - *Pourquoi `git am` et pas copier les fichiers à la main :* `git am` recrée exactement le même commit, sans oubli de fichier, et refuse de s'appliquer s'il y a un conflit avec le code d'Achref, au lieu de l'écraser en silence.
- **`git log --oneline -2`** : vérifie que le commit `feat(m2): ...` est bien au-dessus de celui d'Achref.

### Étape 2 — Préparer l'environnement Python

```cmd
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

- **Ce que ça fait :** crée un environnement isolé (`.venv`) et installe toutes les bibliothèques du projet.
- **Pourquoi un venv :** chaque projet garde ses propres versions. Sans venv, une mise à jour pour un autre projet peut casser EmberWatch.
- **Pourquoi un venv et pas conda :** le README d'Achref utilise déjà `.venv`. Toute l'équipe doit avoir la même méthode, sinon les commandes du README ne marchent pas chez tout le monde.
- **Pourquoi Python 3.11 :** le U-Net d'Achref utilise TensorFlow, qui ne supporte pas toujours les toutes dernières versions de Python. La 3.11 marche à la fois avec TensorFlow et avec PyTorch / Ultralytics. Si `py -3.11` n'existe pas chez toi, installe Python 3.11 depuis https://www.python.org/downloads/.
- **Pourquoi `-r requirements.txt` :** reproductibilité (CDC, exigences non fonctionnelles). Toute l'équipe installe la même liste.
- **Bibliothèques ajoutées pour M2 :** `ultralytics` (YOLO), `sahi` (tuiles), `pillow` (lecture d'images), `pydantic` (contrat), `python-multipart` (upload d'images dans FastAPI), `pytest` (tests).
- **Attention :** sur ta machine, PyTorch s'installe avec `ultralytics`. Si tu as un GPU NVIDIA, installe la version CUDA de PyTorch depuis https://pytorch.org avant le reste.

### Étape 3 — Télécharger les données

Range les données ainsi (elles ne vont **jamais** dans git, elles sont trop lourdes) :

```
data/raw/dfire/         train/images, train/labels, test/images, test/labels
data/raw/hpwren_smoke/  train/..., valid/..., test/..., data.yaml   (export Roboflow YOLOv8)
```

- **Pourquoi le format YOLO :** D-Fire est déjà dans ce format, et Roboflow l'exporte directement. Pas de conversion à écrire = moins de bugs.
- **Pourquoi D-Fire ET HPWREN :** D-Fire est **grand** (beaucoup de scènes variées) mais ses images ne viennent pas des caméras de tours. HPWREN est **petit** mais vient des **mêmes caméras que FIgLib**, le dataset de test du M3. Les deux ensemble = variété + bon domaine.

### Étape 4 — Construire le dataset (CDC M1)

```cmd
.venv\Scripts\python.exe -m src.detection.prepare_dataset --dfire data/raw/dfire --hpwren data/raw/hpwren_smoke --out data/processed/m2_smoke --copy
```

- **Ce que ça fait :**
  1. lit toutes les images et étiquettes des deux sources ;
  2. garde **seulement les boîtes de fumée** (classe 0) ;
  3. garde les images « feu sans fumée » comme **négatifs** (fichier d'étiquettes vide) ;
  4. découpe en train / val / test ;
  5. écrit `data.yaml` (lu par YOLO) et `split_report.json` (rapport de fuite).
- **`--copy` :** copie les images au lieu de créer des liens symboliques. Nécessaire sur Windows, inutile sur Kaggle.
- **À vérifier après :** dans `split_report.json`, `leakage_free` doit être `true` et `sources.dfire.images_per_class` doit montrer la fumée dans ≈ 10 500 images et le feu dans ≈ 5 800.

**Les choix et leurs raisons :**

| Choix | Pourquoi | Pourquoi pas l'autre option |
|---|---|---|
| Fumée seulement | Le CDC vise la **détection précoce** : au départ d'un feu, on voit la fumée, pas les flammes. | Fumée + feu : la classe feu apprend des flammes proches, inutiles pour des caméras à 20 km. |
| Images « feu seul » gardées en négatifs | Elles apprennent au modèle ce qui **n'est pas** de la fumée. | Les supprimer : on perdrait des exemples difficiles gratuits. |
| Classe fumée **déduite** des comptages | Le README de D-Fire donne les comptages mais pas les numéros de classe. Une erreur ici entraînerait un « détecteur de flammes » sans qu'on le voie. | Écrire `0` en dur : risqué si D-Fire utilise `1` pour la fumée. |
| HPWREN découpé **par feu** | Les images d'un même feu se ressemblent presque à l'identique. Si elles sont en train ET en test, le modèle « reconnaît » la scène et le score est faussement bon. Le CDC l'interdit (§9.4). | Découpage aléatoire (celui de Roboflow) : fuite de données → mAP gonflé. |
| Split **test officiel** de D-Fire gardé | Nos chiffres restent comparables à ceux publiés sur D-Fire. | Re-découper D-Fire : chiffres non comparables. |
| Le script **s'arrête** si fuite | Une fuite silencieuse est pire qu'une erreur visible. | Un simple avertissement : facile à ignorer. |

### Étape 5 — Entraîner YOLO (sur Kaggle, GPU gratuit)

Ouvre `notebooks/02_m2_train_yolo_kaggle.ipynb` sur Kaggle. Réglages : Accelerator **GPU T4 x2** ou **P100**, Internet **On**. La commande principale du notebook :

```bash
python -m src.detection.train --data /kaggle/working/m2_smoke/data.yaml --config configs/m2_yolo_kaggle.yaml --device 0
```

- **Ce que ça fait :** part d'un YOLOv8s **déjà entraîné sur COCO** et l'adapte à la fumée (*fine-tuning*). Sauvegarde le meilleur modèle dans `runs/m2/yolo_smoke/weights/best.pt`.
- **Tous les hyperparamètres sont dans `configs/m2_yolo.yaml`**, chacun commenté (valeur, raison, alternative refusée). Le notebook en fait une copie `m2_yolo_kaggle.yaml` ajustée par l'étude des données (EDA). Les principaux :
  - `yolov8s.pt` : taille *small*, bon compromis vitesse / précision, tient sur un GPU de 16 Go.
  - `epochs: 100` + `patience: 20` : 100 est un plafond ; l'entraînement s'arrête seul si le mAP de validation ne progresse plus pendant 20 epochs.
  - `imgsz: 640` : même taille que les tuiles SAHI à l'inférence (cohérence entraînement / inférence).
  - `batch: 32` : tient sur un T4 avec la précision mixte ; à baisser à 16 si erreur « CUDA out of memory ».
  - `optimizer: SGD` fixé : `auto` pourrait changer d'optimiseur d'une ablation à l'autre.
  - `seed: 42`, `deterministic: true` : même résultat si on relance (CDC : reproductibilité).
  - `copy_paste: 0` : **sans effet** sur des boîtes (Ultralytics l'ignore sans masques de segmentation — vérifié dans son code).
  - `flipud: 0` : **jamais** de retournement vertical, car la fumée monte ; `degrees: 0` : caméras à l'horizontale.
  - `fliplr: 0.5` : le retournement gauche-droite reste réaliste.
- **Raisonnement complet** (filtres appliqués / refusés, chaque hyperparamètre) : [M2_DONNEES_ET_ENTRAINEMENT.md](M2_DONNEES_ET_ENTRAINEMENT.md).

| Choix | Pourquoi | Pourquoi pas l'autre option |
|---|---|---|
| **YOLO** (famille imposée par le CDC) | Rapide, très utilisé dans la littérature fumée/feu, export ONNX facile (utile pour le Stage 2 edge). | **Faster R-CNN** : plus lent, pas adapté au temps réel. **DETR** classique : entraînement très long. |
| **YOLOv8s** comme baseline | Version la plus citée dans les articles fumée → baseline comparable. | Version plus récente d'Ultralytics : se teste ensuite comme **ablation**, sans changer le code (`--model`). |
| **RT-DETR** gardé pour plus tard | C'est la comparaison « transformer » prévue par le CDC (Sprint 4). | Le faire dès le Sprint 1 : on ferait l'amélioration avant la base (CORE avant ENHANCEMENT). |
| **Pré-entraîné COCO** | Le modèle connaît déjà les formes et textures → converge beaucoup plus vite. | Depuis zéro (`yolov8n.yaml`) : il faudrait beaucoup plus de données et de temps. Utilisé seulement pour les tests rapides. |
| **Kaggle** | GPU gratuit ~30 h/semaine, datasets ajoutables facilement. | **Colab gratuit** : sessions coupées plus souvent. **CPU** : des jours au lieu d'heures. |

### Étape 6 — Évaluer (CDC §11)

```bash
python -m src.detection.evaluate --weights runs/m2/yolo_smoke/weights/best.pt --data /kaggle/working/m2_smoke/data.yaml --device 0
```

- **Ce que ça fait :** calcule mAP@0.5, mAP@0.5:0.95, précision et rappel sur le **split test**, puis mesure le temps par image **avec et sans** tuiles SAHI. Écrit tout dans `artifacts/m2_metrics.json` et indique si l'objectif MVP (≥ 0.55) et l'objectif ambitieux (≥ 0.65) sont atteints.
- **Pourquoi le test et pas la validation :** la validation a servi à choisir le meilleur epoch (early stopping). Mesurer dessus donnerait un score trop optimiste.
- **Pourquoi mAP@0.5 :** c'est la métrique demandée par le CDC (§11) et la plus utilisée pour la fumée. Les bords d'un panache sont flous, donc exiger un recouvrement de 95 % (mAP@0.95) serait injuste.
- **Pourquoi mesurer le débit :** le CDC demande le débit GPU/CPU pour M2, et le Stage 2 edge en aura besoin comme point de départ.

### Étape 7 — Lancer l'API et la tester (preuve HTTP réelle)

```cmd
copy runs\m2\yolo_smoke\weights\best.pt artifacts\m2_smoke_best.pt
.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload
```

Dans un autre terminal :

```cmd
curl -F "image=@frame.jpg" -F "camera_id=CAM-03" -F "timestamp=2024-07-18T16:24:10Z" http://127.0.0.1:8000/api/v1/detect
curl http://127.0.0.1:8000/api/v1/alerts
```

- **Ce que ça fait :** `uvicorn` lance le serveur FastAPI d'Achref, qui contient maintenant aussi les routes M2. `curl` envoie une vraie image et récupère la réponse.
- **Réponse attendue :** les boîtes détectées et, s'il y a de la fumée, une alerte au format du contrat (`schema_version`, `alert_id`, `box`, `tiles`, `status: PENDING_CONFIRMATION`…). Une image illisible doit renvoyer **400**, pas 500.
- **Variables utiles :** `EMBERWATCH_DETECTOR_WEIGHTS` (chemin des poids), `EMBERWATCH_DETECTOR_CONF` (seuil, 0.25 par défaut), `EMBERWATCH_DEVICE` (`cpu` ou `cuda:0`).

**Comment fonctionne une détection :**

```
image (ex. 3072 × 2048)
 └─ SAHI découpe en tuiles de 640 px qui se chevauchent de 20 %
     + une passe sur l'image entière (pour les gros panaches)
     └─ YOLO sur chaque tuile
         └─ fusion des boîtes (NMS avec métrique IOS 0.5)
             └─ boîtes en pixels de l'image d'origine
                 └─ alerte = boîte la plus confiante → PENDING_CONFIRMATION
```

| Choix | Pourquoi | Pourquoi pas l'autre option |
|---|---|---|
| **SAHI** (tuiles) | Sur une image de 3072 px réduite à 640 px, une fumée de 40 px devient 8 px → invisible. Le CDC cite SAHI explicitement. | Agrandir `imgsz` à 2048 : 10× plus lent et trop de mémoire GPU. |
| Tuiles de **640 px** | Même taille qu'à l'entraînement → le modèle voit des objets à la même échelle. | 1024 px : fumée lointaine encore trop petite. 320 px : trop de tuiles, trop lent. |
| Chevauchement **20 %** | Un panache coupé par le bord d'une tuile apparaît entier dans la tuile voisine. | 0 % : panaches coupés en deux. 50 % : 2× plus de calcul pour peu de gain. |
| Fusion **IOS** au lieu d'IoU | IOS (intersection / plus petite boîte) fusionne une petite boîte incluse dans une grande, cas fréquent quand un panache est vu par deux tuiles. | IoU : laisse des doublons pour les boîtes de tailles très différentes. |
| Seuil de confiance **0.25** | Valeur par défaut de YOLO, pour le MVP. Le **vrai seuil** sera calibré au Sprint 3 sur la validation, selon un budget de fausses alertes (CDC P2). | Fixer un seuil maintenant sans données : arbitraire. |
| Grille de tuiles **224 px** dans l'alerte | Même grille que le futur modèle M3 (style SmokeyNet) → une alerte M2 ou M3 désigne les mêmes tuiles. | Numéroter les tuiles SAHI : grille qui change selon la taille de l'image. |
| Alertes **en mémoire** | Suffisant pour le Sprint 1. | PostGIS dès maintenant : c'est le travail de Ghofrane (M7), au Sprint 2–3. |
| Routes dans un **fichier séparé** | `main.py` d'Achref ne reçoit que 3 petites modifications (import, CORS, include_router) → peu de conflits git. | Tout écrire dans `main.py` : conflits garantis entre nous trois. |

### Étape 8 — Lancer les tests

```cmd
.venv\Scripts\python.exe -m pytest tests -q
```

- **Résultat attendu :** `11 passed`.
- **Ce qui est testé :** le contrat (boîtes valides, tuiles, alerte toujours en attente), le dataset (pas de fuite, aucune image perdue, fumée seule), l'API (alerte créée, pas d'alerte sans fumée, erreur 400 sur un fichier qui n'est pas une image, routes M6 d'Achref toujours OK).
- **Pourquoi un faux détecteur dans les tests :** les tests tournent en 2 secondes, sans GPU ni poids. Ils pourront donc tourner dans le CI de Ghofrane à chaque pull request.
- **Deux bugs déjà trouvés grâce aux tests et à la preuve HTTP :**
  1. une image disparaissait lors de la création du split val ;
  2. un fichier non-image renvoyait 500 au lieu de 400, car Ultralytics modifie `Image.open` de PIL.

### Étape 9 — Envoyer le travail sur GitHub

Le commit existe déjà grâce au `git am` de l'étape 1. Il ne reste qu'à l'envoyer :

```cmd
git push -u origin m2-detection
```

Puis ouvre une **Pull Request** sur GitHub vers `main`, avec Achref et Ghofrane comme relecteurs.

- **`-u origin m2-detection`** : envoie la branche sur GitHub et la relie à ta branche locale. Ensuite, un simple `git push` suffit.
- **Si GitHub refuse (403 / permission denied) :** Achref doit t'ajouter comme collaborateur (GitHub → Settings → Collaborators) avec le droit d'écriture.
- **Pourquoi une PR :** le CDC (§17) demande des revues de code. Achref vérifie que ses routes M6 marchent toujours, Ghofrane reprend l'API pour le M7.
- **Pourquoi des messages `feat(m2): ...` :** on voit tout de suite le module et le type de changement dans l'historique.
- **Pour tes prochains changements :** `git add <fichiers>`, puis `git commit -m "feat(m3): ..."`, puis `git push`. Un commit = une idée, pour que la relecture reste simple.

---

## 3. Prochaines étapes du CDC

**Fin du Sprint 1 (semaine 3) — pour le jalon MS1 :**
1. Lancer le notebook Kaggle sur les vraies données → remplir le mAP@0.5 dans `docs/M2_DETECTION.md` et dans le tracker.
2. Précision / rappel image par image sur le test FIgLib (dès que Ghofrane a préparé FIgLib).
3. Augmentations (mosaic / copy-paste) et comparaison de débit GPU vs CPU : **ENHANCEMENT**, seulement si les tâches CORE sont finies.

**Sprint 2 (semaines 4–5) — M3, le cœur scientifique :**
1. **CORE :** règle de persistance — alerte seulement si la fumée est vue dans **k des n dernières images** au même endroit. Mesurer le temps de détection (objectif ≤ 5 min).
2. **CORE :** mettre en cache les features d'un backbone gelé, tuile par tuile, sur la grille de 224 px → entraînement rapide sur Kaggle.
3. **CORE :** modèle temporel v1 (CNN → transformer temporel → ViT, style SmokeyNet).
4. **ENHANCEMENT :** différences d'images en entrée.

---

## 4. Note sur la licence

Ultralytics YOLO est sous licence **AGPL-3.0** : aucun problème pour un projet académique. Si Tradrly veut réutiliser le modèle dans un produit commercial, il faudra une licence entreprise Ultralytics ou une alternative permissive (par exemple RT-DETR, Apache-2.0). À confirmer avec M. Sammari.
