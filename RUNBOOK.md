# RUNBOOK — Sprite Generator (Diffusers + LoRA PEFT)

Ce document décrit TOUTES les commandes à exécuter, dans l’ordre,
pour entraîner un modèle de génération de sprites (LoRA) sur RunPod,
depuis un dataset brut jusqu’au téléchargement du modèle en local.

============================================================
0) PRÉREQUIS
============================================================

LOCAL (ton PC) :
- Un compte GitHub
- Git installé
- Dataset brut organisé comme suit :

data/raw/
  pokemon_name_1/
    img1.png
    img2.png
  pokemon_name_2/
    img1.png
    ...

- Un fichier CSV des types :

data/types.csv
Format :
pokemon_name,type1,type2
bulbasaur,grass,poison
pikachu,electric,
...

GPU (CLOUD) :
- Un compte RunPod
- Un pod avec GPU NVIDIA (RTX 3090 ou 4090 recommandé)
- Accès au Web Terminal

============================================================
1) SETUP SUR RUNPOD
============================================================

1.1 Ouvrir le terminal du pod
RunPod Dashboard → Pods → ton pod → Connect → Web Terminal

1.2 Vérifier le GPU
```bash
nvidia-smi

1.3 Installer outils de base
apt update && apt upgrade -y
apt install -y git
pip install --upgrade pip

============================================================
2) CLONER LE PROJET
cd /
git clone https://github.com/<TON_USER>/<TON_REPO>.git
cd <TON_REPO>

Installer les dépendances :
pip install -r requirements.txt
pip install -U peft

Vérifier les versions :
python -c "import diffusers, torch; print(diffusers.__version__, torch.__version__)"


) UPLOADER LE DATASET BRUT

Uploader le dossier data/raw/ sur RunPod via :

Onglet Files (upload)
OU

scp / rsync

À la fin, tu dois avoir :
data/raw/<pokemon>/*.png

============================================================
4) PRÉPARATION DU DATASET (resize + metadata)

Resize en 128x128 + génération metadata.csv :

python -m src.prepare_dataset \
  --raw_dir data/raw \
  --out_dir data/processed \
  --size 128 \
  --types_csv data/types.csv


Résultat attendu :
data/processed/
images/
metadata.csv

============================================================
5) GÉNÉRATION DES CAPTIONS (types → tokens)
python -m src.build_captions --processed_dir data/processed


Résultat :
data/processed/captions/*.txt

============================================================
6) CRÉER LES SPLITS TRAIN / VAL
python -m src.make_splits \
  --processed_dir data/processed \
  --val_ratio 0.02


Résultat :
data/splits/train.txt
data/splits/val.txt

============================================================
7) ENTRAÎNEMENT LORA (PEFT)

Commande standard (RTX 3090 / 4090) :

python -m train_lora_diffusers \
  --processed_dir data/processed \
  --splits_dir data/splits \
  --output_dir outputs/lora_style_types \
  --rank 16 \
  --lr 1e-4 \
  --batch_size 4 \
  --epochs 12 \
  --mixed_precision fp16


Si OOM (manque VRAM) :

python -m train_lora_diffusers \
  --processed_dir data/processed \
  --splits_dir data/splits \
  --output_dir outputs/lora_style_types \
  --rank 8 \
  --lr 5e-5 \
  --batch_size 2 \
  --epochs 12 \
  --mixed_precision fp16


Résultat attendu :
outputs/lora_style_types/
checkpoint_epoch_1/
checkpoint_epoch_2/
...
final/

============================================================
8) GÉNÉRER DES SPRITES
python -m generate \
  --lora_dir outputs/lora_style_types/final \
  --prompt "pixel sprite, monster, front view, type_dragon, type_ice" \
  --out outputs/samples/dragon_ice.png

============================================================
9) TÉLÉCHARGER LE MODÈLE EN LOCAL

Créer une archive du modèle :

cd /<TON_REPO>
tar -czf lora_final.tar.gz outputs/lora_style_types/final


Télécharger via RunPod :

Onglet Files → télécharger lora_final.tar.gz

OU via SCP (si SSH activé) :

scp root@<HOST_RUNPOD>:/<TON_REPO>/lora_final.tar.gz .


En local :

tar -xzf lora_final.tar.gz

============================================================
10) ARRÊTER LE POD (IMPORTANT)

RunPod Dashboard → Stop Pod / Delete Pod
(Sinon tu continues à payer)

============================================================