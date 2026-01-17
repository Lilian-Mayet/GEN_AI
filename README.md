# Dataset Preparation for Pokemon Sprite Generation

## Structure du projet

```
Monster_sprite_generator/
├── downloads/                    # Sprites bruts téléchargés
│   ├── pokemon_data.csv         # Métadonnées (nom, type1, type2)
│   ├── bulbasaur/
│   │   ├── bulbasaur_0001.png
│   │   ├── bulbasaur_0002.png
│   │   └── ...
│   ├── charmander/
│   └── ...
├── GEN_AI/
│   ├── prepare_dataset.py       # Script de préparation
│   └── data/
│       └── processed/           # Dataset prêt pour l'entraînement
│           ├── images/          # Images redimensionnées 128x128
│           ├── captions/        # (optionnel pour le futur)
│           └── metadata.csv     # Métadonnées finales
└── downlaod_sprites.py          # Script de téléchargement
```

## 🚀 Utilisation

### Étape 1 : Télécharger les sprites (si ce n'est pas déjà fait)

```bash
# Télécharger tous les pokémons du CSV
python downlaod_sprites.py --csv downloads/pokemon_data.csv

# Ou reprendre après un pokémon spécifique
python downlaod_sprites.py --csv downloads/pokemon_data.csv --after groudon
```

### Étape 2 : Préparer le dataset

**Commande de base (utilise les valeurs par défaut) :**
```bash
cd GEN_AI
python prepare_dataset.py
```

Cela va :
- Lire les sprites depuis `../downloads/`
- Lire les types depuis `../downloads/pokemon_data.csv`
- Redimensionner toutes les images en **128×128** avec padding
- Sauvegarder dans `data/processed/`

**Commande avec options personnalisées :**
```bash
python prepare_dataset.py --raw_dir ../downloads --out_dir data/processed --size 128 --types_csv ../downloads/pokemon_data.csv
```

### Étape 3 : Générer les captions

**Commande de base :**
```bash
python build_caption.py
```

Cela va créer un fichier `.txt` pour chaque image avec le format :
```
pixel sprite, monster, front view, type_grass, type_poison
```

**Commande avec options personnalisées :**
```bash
python build_caption.py --processed_dir data/processed --prefix "pixel art sprite, creature"
```

### Étape 4 : Vérifier le résultat

Après exécution complète, vous aurez :
- `data/processed/images/` : toutes les images redimensionnées (ex: `bulbasaur__0000.png`)
- `data/processed/captions/` : fichiers texte avec les captions (ex: `bulbasaur__0000.txt`)
- `data/processed/metadata.csv` : fichier avec colonnes `image`, `pokemon_name`, `type1`, `type2`

## 📊 Format du dataset final

Le fichier `metadata.csv` ressemblera à :

```csv
image,pokemon_name,type1,type2
bulbasaur__0000.png,bulbasaur,grass,poison
bulbasaur__0001.png,bulbasaur,grass,poison
charmander__0000.png,charmander,fire,
pikachu__0000.png,pikachu,electric,
```

## 🎨 Traitement des images

Le script effectue :
1. **Conversion en RGBA** (préserve la transparence)
2. **Padding carré centré** avec fond transparent
3. **Redimensionnement 128×128** avec interpolation `NEAREST` (préserve les pixels nets du pixel art)

## 💡 Options avancées

- `--size 256` : redimensionner en 256×256 au lieu de 128×128
- `--raw_dir chemin/custom` : utiliser un autre répertoire source
- `--out_dir output/custom` : changer le répertoire de sortie
