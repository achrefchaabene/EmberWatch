# M2 — Données, filtres et hyperparamètres : le raisonnement

> Responsable : **Mouhib** (Rôle A) · Sprint 1 · Suit le CDC : M1 (données propres, sans fuite) → M2 (YOLO + SAHI) → §11 (métriques).
> Ce document explique **chaque choix**, **pourquoi**, et **pourquoi pas l'alternative**.

---

## 1. Le flux complet

```
configs/datasets.yaml ──► download.py ──► eda.py (étude) ──► décisions ──► prepare_dataset.py ──► eda.py (contrôle)
                                                                                   │
                       configs/m2_yolo.yaml ──► train.py (Kaggle GPU) ──► evaluate.py ──► best.pt + m2_metrics.json
```

Tout se lance depuis le notebook `notebooks/02_m2_train_yolo_kaggle.ipynb` (cellule par cellule, dans cet ordre).

---

## 2. Télécharger les datasets depuis leurs liens

Les liens sont écrits **une seule fois** dans `configs/datasets.yaml`. Le code lit ce fichier : pour ajouter un dataset, on ajoute 5 lignes de YAML, pas de Python.

| Source | Fonction | Ce qu'il faut | Exemple |
|---|---|---|---|
| Kaggle | `fetch_kaggle` | Sur Kaggle : rien (ou *Add data*). En local : `kaggle.json` | D-Fire (`sayedgamal99/smoke-fire-detection-yolo`, copie donnée par le README officiel) |
| Roboflow | `fetch_roboflow` | Clé `ROBOFLOW_API_KEY` | HPWREN smoke (`brad-dwyer/wildfire-smoke` v1) |
| HTTP | `fetch_http` | Rien | FIgLib (`.tgz` par séquence) |
| Google Drive | `fetch_gdrive` | Lien partagé « tous ceux qui ont le lien » | Données d'un membre de l'équipe |
| GitHub | `fetch_github` | Rien | Code + scripts de Mesogeos |
| OneDrive | `fetch_onedrive` | Lien de **fichier** | — |

**Choix et raisons :**

| Choix | Pourquoi | Pourquoi pas l'autre option |
|---|---|---|
| Liens dans un YAML | Un seul endroit à mettre à jour, lisible par toute l'équipe | Liens codés en dur dans le Python : dispersés, faciles à oublier |
| D-Fire via **Kaggle** | Déjà sur Kaggle → pas d'upload, pas de quota de téléchargement | OneDrive : les liens de **dossier** ne se téléchargent pas par script (limite de l'API OneDrive) |
| Clés via **Kaggle Secrets** / variable d'environnement | Le dépôt est sur GitHub : une clé écrite dans le code est publique pour toujours (même supprimée, elle reste dans l'historique git) | Clé dans le notebook : fuite garantie au premier partage |
| Téléchargement **idempotent** (marqueur `.complete`) | Relancer une cellule ne retélécharge pas 3 Go | Sans marqueur : chaque « Run all » gaspille temps et quota |
| Extraction **sécurisée** | Une archive piégée (`../../fichier`) ne peut pas écrire hors du dossier (attaque « zip slip ») | `extractall` brut : dangereux sur des archives venant d'Internet |
| `manifest.json` | Les étapes suivantes savent où sont les données, sans deviner | Chemins recopiés à la main : erreurs entre PC et Kaggle |

---

## 3. Étudier le contenu AVANT de choisir les filtres (EDA)

Règle d'expert : **on ne choisit pas un filtre parce qu'il « améliore l'image » à l'œil, on le choisit parce qu'une mesure du dataset le justifie**, et on le valide par une ablation (mAP avec / sans).

`src/detection/eda.py` mesure :

| Mesure | Ce qu'elle révèle | Décision qu'elle déclenche |
|---|---|---|
| Images corrompues, boîtes invalides | Fichiers cassés, annotations hors image ou de taille nulle | Écartées / corrigées par `prepare_dataset` |
| Taille des images (côté long) | Images très grandes = chargement lent | Redimensionnement **hors-ligne** seulement si médiane > 1600 px |
| Taille des boîtes en pixels (< 32 px = petite) | Fumée lointaine | SAHI à l'inférence ; ablation `imgsz` 960/1024 si ≥ 30 % de petites boîtes |
| Part d'images en niveaux de gris | Caméras mono / infrarouge | Probabilité de l'augmentation `ToGray` |
| Part d'images sombres | Aube, crépuscule, nuit | Amplitude de luminosité ; la nuit devient un mode d'échec documenté (P4) |
| Netteté (variance du Laplacien) | Images déjà floues (brume, mise au point) | Pas d'augmentation de flou si ≥ 20 % d'images floues |
| Part d'images sans fumée | Assez d'exemples « négatifs » ? | Ajout de négatifs difficiles (nuages, brouillard) au Sprint 2 (M4) |
| Quasi-doublons **entre** splits | Fuite de données → score de test gonflé | Fichiers listés dans `eda.json` → à retirer de val/test |
| Carte des centres de boîtes | Où apparaît la fumée (au-dessus de l'horizon) | Idées de découpage / tuiles pour le M3 |

La cellule 4 du notebook transforme ces mesures en réglages et **les affiche** : chaque choix est traçable dans le notebook et dans le rapport.

---

## 4. Les « filtres » : lesquels on applique, lesquels on refuse

Il y a deux familles très différentes :

- **Prétraitement hors-ligne** : on modifie les fichiers une fois pour toutes. Il faudra appliquer **exactement** le même traitement à chaque image en production (sinon le modèle voit des images différentes de celles de l'entraînement).
- **Augmentation à l'entraînement** : transformation **aléatoire** à chaque epoch, seulement pendant l'entraînement. Elle apprend la **robustesse**, sans rien changer en production.

Pour un CNN moderne comme YOLO, la règle est : **presque pas de prétraitement, des augmentations choisies**. Les premières couches du réseau apprennent déjà leurs propres filtres (bords, textures), mieux adaptés que des filtres faits à la main.

### 4.1 Prétraitement hors-ligne

| Filtre | Décision | Raisonnement |
|---|---|---|
| **Nettoyage** (images corrompues, boîtes invalides, doublons entre splits) | ✅ **Toujours** | Ce n'est pas un filtre d'image, c'est de la qualité de données. « Garbage in, garbage out ». |
| **Resize** | ⚠️ **Seulement si l'EDA le dit** (médiane > 1600 px → 1280) | YOLO redimensionne déjà à la volée (letterbox). Réduire hors-ligne fait perdre des détails. Utile seulement pour accélérer le chargement d'images énormes. Les originaux restent pour SAHI. |
| **Contours** (Canny, Sobel) | ❌ Non | La fumée a des bords **flous et diffus** : Canny ne trouve presque rien sur la fumée, mais beaucoup sur les crêtes, arbres et nuages. On perdrait la couleur et la texture, qui sont justement les indices utiles. |
| **Netteté / « focus »** (sharpen, unsharp mask) | ❌ Non | Amplifie le bruit du capteur et les artefacts JPEG → fausses alertes. La fumée est un signal **basse fréquence** : la netteté ne l'aide pas. |
| **Flou / débruitage** comme prétraitement | ❌ Non | Une fumée naissante est un signal **faible et peu contrasté** : le débruitage l'efface en premier. |
| **Retouche** manuelle | ❌ **Jamais** | Change la distribution des données et rend l'évaluation malhonnête (CDC : rigueur scientifique). |
| **Débrumage** (dehaze) | ❌ Non | Il enlève… la fumée. Fumée et brume ont la même physique (diffusion de la lumière). |
| **CLAHE / égalisation d'histogramme** | ⚠️ **Ablation seulement** | Augmente le contraste local de la fumée **mais aussi** des nuages et du bruit. En prétraitement, il faudrait aussi l'appliquer en production (latence en plus). On le teste d'abord en augmentation (`clahe` dans la config) et on garde ce qui améliore le mAP. |
| **Recadrer le ciel** | ❌ Pas au M2 | Les fumées naissent souvent **sous** la ligne d'horizon (sur la pente). On verra au M3 avec la carte des centres. |
| **Normalisation** (÷255, moyenne/écart-type) | ✅ Automatique | Faite par Ultralytics ; rien à coder. |

### 4.2 Augmentations à l'entraînement

| Augmentation | Valeur | Raisonnement | Refusée / réduite parce que… |
|---|---|---|---|
| Mosaic | 1.0, coupé les 10 dernières epochs | Plus de petits objets et de fonds variés par batch | Gardée jusqu'au bout, elle laisse un écart avec les vraies images entières |
| Scale | 0.5 | Fumée vue de près et de loin | — |
| Translate | 0.1 | Panache pas toujours au centre | — |
| Miroir gauche-droite | 0.5 | Un paysage miroir reste réaliste | — |
| Miroir haut-bas | **0** | — | La fumée **monte** ; un ciel en bas n'existe pas |
| Rotation, cisaillement, perspective | **0** | — | Caméras de tour **à l'horizontale** et fixes : une ligne d'horizon penchée n'arrive jamais |
| Mixup | **0** | — | Mélanger deux photos crée des formes **semi-transparentes** qui ressemblent exactement à de la fumée → bruit d'étiquettes |
| Copy-paste | **0** | — | Sans effet sur des **boîtes** : Ultralytics l'ignore s'il n'y a pas de masques de segmentation (vérifié dans le code source) |
| Teinte (hsv_h) | 0.01 (petit) | — | La fumée est grise/blanche/brune ; une forte rotation de teinte crée des ciels irréalistes |
| Saturation (hsv_s) | 0.5 | Brume et crépuscule réduisent les couleurs | — |
| Luminosité (hsv_v) | 0.4 | Matin, midi, soir | — |
| ToGray | 0.05–0.10 (**fixé par l'EDA**) | Caméras mono / IR ; force le modèle à ne pas dépendre que de la couleur | — |
| Luminosité/contraste | p = 0.2 (±20 %) | Conditions de lumière | — |
| Compression JPEG | p = 0.2 (qualité 60–100) | Les flux caméra sont compressés | — |
| Bruit gaussien | p = 0.05 (léger) | Bruit du capteur la nuit | Fort : efface la fumée fine |
| Flou | p = 0.05, noyau 3 (**0 si l'EDA trouve ≥ 20 % d'images floues**) | Légère mise au point / brume | Fort : efface la fumée fine |
| CLAHE | **0** (ablation) | Voir 4.1 | — |

**Pourquoi des probabilités faibles ?** Une augmentation doit apprendre la **robustesse** sans **effacer le signal**. Pour un objet aussi faible que de la fumée naissante, des augmentations trop fortes produisent des images où la boîte annonce de la fumée… invisible : le modèle apprend du bruit.

---

## 5. Hyperparamètres (fichier `configs/m2_yolo.yaml`)

| Paramètre | Valeur | Pourquoi | Pourquoi pas l'autre option |
|---|---|---|---|
| `model` | `yolov8s.pt` (pré-entraîné COCO) | Meilleur compromis vitesse/précision sur un GPU de 16 Go ; version la plus citée dans les articles fumée → comparable | `n` : trop faible pour la fumée fine. `m`/`l` : 2–4× plus lent pour quelques points. Version plus récente = **ablation** |
| `single_cls` | true | Une seule classe : pas de confusion entre classes | — |
| `imgsz` | 640 | Même taille que les tuiles SAHI → même échelle à l'entraînement et à l'inférence | 1024 : 2,5× plus lent ; à tester seulement si beaucoup de petites boîtes |
| `epochs` / `patience` | 100 / 20 | 100 = plafond ; l'arrêt anticipé s'arrête quand la validation ne progresse plus | Sans `patience` : heures de GPU gâchées |
| `time` | **non utilisé** | Dans Ultralytics, `time` **recalcule** le nombre d'epochs pour remplir le temps (vérifié dans le code) → ablations non comparables | — |
| `save_period` | 5 | Checkpoint toutes les 5 epochs → reprise après une coupure Kaggle (`--resume`) | — |
| `batch` | 32 | Tient sur un T4 avec yolov8s @640 + AMP | 16 si « CUDA out of memory » |
| `optimizer` | **SGD** (fixé) | `auto` choisit SGD **ou** AdamW selon le nombre d'itérations → deux ablations pourraient utiliser deux optimiseurs différents sans qu'on le voie | AdamW : converge vite sur de courts entraînements, mais généralise souvent un peu moins bien en détection |
| `lr0` / `lrf` | 0.01 / 0.01 | Valeurs standard du fine-tuning YOLO ; le LR descend jusqu'à 1 % | — |
| `cos_lr` | true | Fin d'entraînement plus douce que la décroissance linéaire | — |
| `warmup_epochs` | 3 | Petits pas au début pour ne pas détruire les poids pré-entraînés | — |
| `weight_decay` | 0.0005 | Régularisation légère standard | — |
| `cache` | false | D-Fire en RAM ≈ 20 Go : trop pour Kaggle ; cache disque = remplit `/kaggle/working` | — |
| `workers` | 4 | Kaggle donne 4 cœurs CPU | Plus : aucun gain, plus de RAM |
| `amp` | true | Précision mixte : ~2× plus rapide, moitié de mémoire, même précision | — |
| `seed` / `deterministic` | 42 / true | Reproductibilité (exigence non fonctionnelle du CDC) | — |

**Règle des ablations :** on change **une seule** valeur à la fois, on garde tout le reste fixe, et on compare le mAP@0.5 sur le **test**. Ablations prévues (Sprint 4) : `imgsz` 1024, CLAHE p = 0.1, mixup 0.1, version YOLO plus récente, RT-DETR.

---

## 6. Sur Kaggle, pas à pas

1. **Compte vérifié par téléphone** (sinon pas de GPU ni d'Internet).
2. **Secrets** : *Add-ons → Secrets → Add* `ROBOFLOW_API_KEY`, puis cocher « attach to notebook ».
3. Importer `notebooks/02_m2_train_yolo_kaggle.ipynb` (*File → Import notebook*), le laisser **Private**.
4. *Settings* : **GPU T4 x2**, **Internet On**. (Optionnel : *Add data* → `smoke-fire-detection-yolo`.)
5. Exécuter **cellule par cellule** la première fois, en lisant les sorties de l'EDA (cellules 3–6).
6. La cellule 7 (≈ 3 min) vérifie que tout marche **avant** de lancer l'entraînement long (cellule 8).
7. Après l'entraînement : **Save Version** (sinon `runs/` est perdu), puis télécharger depuis l'onglet *Output* :
   `m2_smoke_best.pt`, `m2_metrics.json`, `m2_run_config.json`, `m2_split_report.json`, `m2_eda.zip`.
8. Mettre `m2_smoke_best.pt` dans `artifacts/` sur le PC pour l'API ; écrire le mAP@0.5 dans `docs/M2_DETECTION.md` et dans le tracker.

---

## 7. Ce qui a été vérifié

- 26 tests automatiques (téléchargement via un vrai serveur HTTP local, extraction sécurisée, EDA, nettoyage, doublons, redimensionnement, config, API).
- Chaîne complète sur données synthétiques avec erreurs injectées : l'EDA a trouvé l'image corrompue, la boîte invalide et la copie train → test ; la préparation les a retirées.
- Entraînement court avec la config : Ultralytics confirme **nos** augmentations et l'optimiseur **SGD**.
- La cellule de décision du notebook a été exécutée sur de vrais rapports EDA.

**Pas encore vérifié** (nécessite Internet + GPU, donc Kaggle) : le téléchargement réel depuis Kaggle et Roboflow, et le mAP réel. Les identifiants Roboflow (`brad-dwyer/wildfire-smoke` v1) sont ceux de la page publique ; si Roboflow refuse, ouvre la page du dataset → *Download* → *Show download code* et recopie `workspace`, `project`, `version` dans `configs/datasets.yaml`.
