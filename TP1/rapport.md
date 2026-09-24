# TP1 — Premiers pas (CSC 8607, Introduction au deep learning)

## 1. Utilisation de SLURM

**Modèle exact du GPU alloué (sortie de `nvidia-smi` sur le nœud de calcul) :**
> NVIDIA H100 NVL

**Commande exacte utilisée pour annuler le job :**
```
 scancel 1731
```

**Nom exact du fichier de log généré par `hello.sh` :**
> Avec `-J hello-slurm` et `-o logs/%x-%j.out`, le fichier s'appelle `logs/hello-slurm-1743.out` .

**Différence entre `ReqMem` et `MaxRSS` :**
`ReqMem` est la quantité de mémoire demandée à Slurm au moment de la soumission du job. `MaxRSS` est la quantité de mémoire réellement utilisée (pic de mémoire résidente) pendant l'exécution du job. Les deux peuvent différer significativement : un job peut demander 4G et n'en utiliser que 500M, ou au contraire dépasser sa demande et être tué par Slurm.

## 2. Environnement virtuel Python

**Commande pour vérifier la version de Python et le chemin du binaire :**
```
python -c "import sys; print(sys.version); print(sys.executable)"
```
> 3.10.21 | packaged by conda-forge | (main, Aug 21 2026, 22:42:31) [GCC 14.4.0]
/mnt/hdd/homes/mbenhassine/miniforge3/envs/deeplearning/bin/python

**Sortie de `check_gpu.py` :**
```
> PyTorch version: 2.13.0
CUDA available: False
Attention, aucun GPU détecté !
```

 `CUDA available` retourne `False`, deux raisons possibles :**
1. PyTorch a été installé en version CPU-only (mauvais canal/paquet lors de l'installation avec Mamba).
2. Le script a été exécuté sur le nœud de connexion (login node) ou en dehors d'une allocation `srun --gres=gpu:1` active, donc aucun GPU n'est visible par le processus.

**Commande pour afficher la version de TensorBoard :**
```
tensorboard --version
```

## 3. Exercices théoriques

### Architecture et paramètres

![alt text](image-4.png)

Nombre total de paramètres :
- **Sans biais :** Couche 1 = 3 × 4 = 12 ; Couche 2 = 4 × 2 = 8 → **Total = 20**
- **Avec biais :** Couche 1 = 3 × 4 + 4 = 16 ; Couche 2 = 4 × 2 + 2 = 10 → **Total = 26**

### Équations et dimensions

```
X  : (N, 3)
W1 : (4, 3)
b1 : (1, 4) -> diffusé en (N, 4)
H  : (N, 4)
W2 : (2, 4)
b2 : (1, 2) -> diffusé en (N, 2)
Y  : (N, 2)
```

### Graphe de calcul et rétropropagation

f(x, y, z) = x/y + z, avec q = x/y.
![ ](image-5.png)

**Forward pass (x=2, y=4, z=0) :** q = 2/4 = 0.5 ; f = 0.5 + 0 = **0.5**

**Gradients locaux (rétropropagation) :**
- ∂f/∂q = 1, ∂f/∂z = 1
- ∂q/∂x = 1/y = 0.25
- ∂q/∂y = −x/y² = −2/16 = −0.125
- **∂f/∂x** = ∂f/∂q · ∂q/∂x = 1 × 0.25 = **0.25**
- **∂f/∂y** = ∂f/∂q · ∂q/∂y = 1 × (−0.125) = **−0.125**
- **∂f/∂z** = **1**

### Mise à jour des poids (η = 1)

- x' = 2 − 1×0.25 = **1.75**
- y' = 4 − 1×(−0.125) = **4.125**
- z' = 0 − 1×1 = **−1**
- f' = x'/y' + z' = 1.75/4.125 − 1 ≈ **−0.576**

La valeur de la fonction est passée de 0.5 à −0.576 : elle a bien **diminué**, comme attendu après un pas de descente de gradient.

### Questions de réflexion

**Pourquoi la règle de la chaîne ?**
Un réseau profond est une composition de nombreuses couches. La règle de la chaîne permet de calculer le gradient de la perte par rapport à chaque paramètre en propageant les dérivées couche par couche en sens inverse, plutôt que de calculer une expression globale complexe d'un seul coup. C'est exactement ce que fait la rétropropagation.

**Pourquoi des mini-batchs ?**
Un seul exemple donne une estimation de gradient très bruitée mais peu coûteuse ; l'ensemble complet des données donne une estimation stable mais coûteuse en calcul et en mémoire. Les mini-batchs offrent un compromis : une estimation raisonnablement précise, calculable efficacement sur GPU (vectorisation), et le bruit résiduel aide même parfois la généralisation.

### Association

| Tâche | Fonction finale (Sortie) | Fonction de perte (Loss) |
|---|---|---|
| Classification binaire | 1. Sigmoid | A. Binary Cross-Entropy (BCE) |
| Classification multi-classes | 2. Softmax | B. Cross-Entropy (NLL) |
| Régression pure | 3. Identité (aucune) | C. MSE |

## 4. Premier réseau de neurones

**Rôle de `batch_size` et `shuffle` :**
`batch_size` détermine combien d'exemples sont traités ensemble à chaque passe avant/arrière (compromis entre stabilité du gradient et vitesse/mémoire). `shuffle` randomise l'ordre des exemples à chaque époque. On utilise `shuffle=True` pour l'entraînement afin d'éviter que le modèle n'apprenne des motifs liés à l'ordre des données et pour obtenir des estimations de gradient variées ; on utilise `shuffle=False` pour le test car l'ordre n'affecte pas la métrique et on veut des résultats déterministes et reproductibles.

**Pourquoi `torch.flatten(x, 1)` :**
Cela aplatit toutes les dimensions sauf la dimension 0 (le batch), transformant chaque image `(3,32,32)` en un vecteur de longueur 3072, tout en préservant la séparation entre exemples. `nn.Linear` attend une entrée 2D `(batch, features)`.

**Pourquoi pas de Softmax avant `nn.CrossEntropyLoss` :**
`nn.CrossEntropyLoss` applique déjà en interne `LogSoftmax` suivi de `NLLLoss`. Ajouter un Softmax dans le modèle appliquerait cette fonction deux fois, ce qui fausserait la valeur de la perte et les gradients (et nuirait à la stabilité numérique).

**Différence entre `optimizer.zero_grad()` et `loss.backward()` :**
`zero_grad()` réinitialise les gradients accumulés (`.grad`) de l'étape précédente, car PyTorch accumule les gradients par défaut. `loss.backward()` calcule les nouveaux gradients par rétropropagation et les stocke dans `.grad` de chaque paramètre.

**Pourquoi `with torch.no_grad():` en évaluation, et avantage :**
Cela désactive la construct
ion du graphe d'autograd. Avantage : pas besoin de stocker les activations intermédiaires nécessaires à une passe arrière, ce qui réduit fortement l'utilisation mémoire et accélère le calcul.

**Précision attendue avec un classificateur aléatoire sur CIFAR-10 :**
CIFAR-10 comporte 10 classes équilibrées, donc un classificateur aléatoire obtiendrait environ **10 %** de précision.

**Sorties d'entraînement (train.py) :**
```
>Epoch 01 | loss=2.0940 | acc=0.3293
Epoch 02 | loss=2.1155 | acc=0.3567
Epoch 03 | loss=2.1447 | acc=0.3606
Epoch 04 | loss=2.0691 | acc=0.3842
Epoch 05 | loss=2.0697 | acc=0.3882
Epoch 06 | loss=2.0513 | acc=0.3950
Epoch 07 | loss=2.0096 | acc=0.4057
Epoch 08 | loss=1.9838 | acc=0.4151
Epoch 09 | loss=1.9656 | acc=0.4211
Epoch 10 | loss=1.9489 | acc=0.4266
Test accuracy: 0.386
```


## 5. TensorBoard

**Pourquoi inclure date, heure et hyperparamètres dans `run_name` :**
Cela isole les logs de chaque run dans un dossier distinct, permettant de comparer plusieurs runs côte à côte dans TensorBoard sans écraser les résultats précédents. Le nom du dossier permet aussi d'identifier immédiatement quels hyperparamètres ont produit quelle courbe.

**Smoothing et bruit sur `Loss/train_step` vs `Loss/train` :**
`Loss/train_step` est enregistrée par batch (32 images), donc très bruitée. `Loss/train` est moyennée sur toute l'époque, ce qui annule le bruit batch-à-batch et donne une courbe beaucoup plus lisse.
> _le niveau entre ~0.6–0.7 : la tendance devient claire sans masquer de changements importants._
![alt text](image-6.png)
**Comparaison des 3 runs (Run 1 : lr=1e-2/bs=32, Run 2 : lr=1e-3/bs=32, Run 3 : lr=1e-1/bs=128) :**
![alt text](image-7.png)
> 

Ce que montrent les courbes :

Run 2 (LR = 1e-3, bs = 32) courbe verte : c'est la meilleure run. Sur Loss/train, la courbe descend de façon lisse et continue, sans à-coups, et se stabilise vers 1.15-1.2. Sur Loss/val, elle suit la même tendance descendante, avec un écart faible et constant par rapport au train. Cette régularité montre un entraînement stable qui généralise bien.

Run 1 (LR = 1e-2, bs = 32) courbe violette : sur Loss/train, la courbe descend très peu, elle reste quasiment plate autour de 1.9-2.0. Sur Loss/val, elle est encore pire : elle oscille en dents de scie entre 2.0 et 2.3 sans jamais vraiment descendre. Ce zigzag est le signe classique d'un learning rate trop élevé : le modèle "saute" autour du minimum au lieu de converger doucement vers lui.

Run 3 (LR = 1e-1, bs = 128) : sur Loss/train_step, on voit un signal très bruité (grosses oscillations verticales), et les valeurs finales sont NaN. Le pas d'apprentissage est tellement grand que les poids divergent complètement. la loss explose au lieu de diminuer, jusqu'à devenir NaN.



**Comment détecter un sur-apprentissage sur les courbes :**
On détecte un sur-apprentissage lorsque la perte d'entraînement (Loss/train) continue de diminuer alors que la perte de validation (Loss/val) stagne, puis recommence à augmenter et les deux courbes divergent après un certain nombre d'époques, l'écart entre elles se creusant progressivement.
