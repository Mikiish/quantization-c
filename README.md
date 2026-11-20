# Matrix Quantization — From Scratch

Projet pédagogique et expérimental en **C++20** consacré à la quantification numérique et à la multiplication matricielle.

## Position du problème

Dans les réseaux neuronaux, la représentation des poids et des activations conditionne à la fois l'empreinte mémoire, le coût des opérations et la précision des résultats. La **quantification** remplace une représentation numérique (par exemple FP32) par un format plus compact (par exemple INT8 ou INT4), puis reconstruit, si nécessaire, une approximation en virgule flottante.

Le projet étudie quatre aspects :

- **Précision** : erreur introduite par la quantification et propagée par la multiplication.
- **Mémoire** : stockage, allocation et représentation binaire des données.
- **Calcul** : accumulation entière, conversions et choix des précisions.
- **Performances** : accès mémoire, cache, vectorisation et coûts de préparation.

**Objectif : comprendre et implémenter les mécanismes**, et non concurrencer les bibliothèques GEMM optimisées.

## Feuille de route

### 1. Quantification uniforme symétrique INT8 — réalisée

Pour une matrice réelle \(W\), on pose \(M=\max_{i,j}|W_{ij}|\) et, si \(M>0\), \(s=M/127\). Les coefficients sont quantifiés et reconstruits par

\[
Q_{ij}=\operatorname{clip}_{[-127,127]}\!\left(\operatorname{round}\frac{W_{ij}}s\right),
\qquad \widehat W_{ij}=sQ_{ij}.
\]

Une convention particulière traite le cas \(M=0\).

- Quantification et déquantification de scalaires, tableaux et matrices.
- Mesures de l'erreur absolue maximale, de la MSE et de l'erreur relative de Frobenius.

### 2. Multiplication matricielle quantifiée — réalisée

- Structure générique `Matrix<T>` et stockage contigu *row-major*.
- Produit `matmul<T, Acc>` avec distinction entre type de stockage et type d'accumulation.
- Structure `Q8Matrix` associant une matrice INT8 et son facteur d'échelle.
- Calcul \(Q_AQ_B\) en INT32, suivi de la reconstruction FP32 :

\[
\widehat{AB}=s_As_B(Q_AQ_B).
\]

- Validation numérique et premiers benchmarks face au produit FP32.

### 3. Quantification INT4 — à faire

- Représentation des valeurs signées sur 4 bits et empaquetage de deux coefficients par octet.
- Masques, décalages binaires, dépaquetage et accumulation.
- Comparaison INT4 / INT8 / FP32 : mémoire, coût de calcul et erreur.

### 4. Benchmarks et optimisations avancées — en cours

- Mesures répétées sur diverses dimensions et avec différentes distributions d'entrées.
- Étude de l'ordre des boucles, de la localité mémoire et du découpage en blocs (*tiling*).
- Coûts séparés de la quantification, de la multiplication, des allocations et de la reconstruction.
- Étude des formats FP32, FP16, BF16 et des formats entiers, sous réserve de leur prise en charge matérielle et logicielle.
- Exploration de la vectorisation et, éventuellement, des bibliothèques spécialisées comme points de comparaison.

**Extensions possibles :** quantification par ligne, colonne ou bloc ; exploitation d'entrées de calibration ; quantification dynamique des opérandes ou des résultats intermédiaires, avec adaptation éventuelle de la précision pendant le calcul.

## Compilation

Environnement de développement : Linux / WSL2, GCC, C++20.

```bash
g++ -std=c++20 -O3 -Wall -Wextra -Wpedantic main.cpp -o main
./main
```

## Rapports de développement

- [20 novembre 2026 — `Rapport-2026-11-20.md`](Rapport-2026-11-20.md) : `Matrix<T>`, multiplication quantifiée, génération aléatoire, tests de précision et premiers benchmarks.

## Principes

Privilégier des implémentations explicites, des expériences reproductibles et une séparation claire entre **résultats observés**, **modèles mathématiques** et **hypothèses de performance**.
