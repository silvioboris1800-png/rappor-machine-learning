---
type: rapport-etude
matiere: Machine Learning
sujet: "Fonctionnement d'un LLM : Transformer, MLP et tokenisation"
tags:
  - machine-learning
  - LLM
  - rapport
---

# Rapport d’étude
## Machine Learning
### Fonctionnement d’un LLM : Transformer, MLP et tokenisation

> [!info] Organisation du document
> Ce document présente les notions de machine learning, de tokenisation, de Transformer et de MLP, puis examine la prédiction du token suivant, les limites des LLM et plusieurs questions d’approfondissement.

## Sommaire

- [[#1. Introduction]]
- [[#2. Compréhension et définition des termes]]
  - [[#2.1. Le machine learning]]
  - [[#2.2. La tokenisation]]
  - [[#2.3. Le Transformer]]
  - [[#2.4. Les couches MLP]]
- [[#3. Prédiction d’un token]]
- [[#4. Problématique]]
- [[#5. Analyse et réponse]]
- [[#6. Recherches d’approfondissement et d’ouverture]]
- [[#7. Conclusion]]
- [[#8. Sources]]

---

## 1. Introduction

Les grands modèles de langage (LLM : Large Language Models) sont devenus, en quelques années, des outils présents dans une grande partie du quotidien : assister a la conversation, traduction, aide à la programmation, rédaction. Pourtant, leur fonctionnement reste largement méconnu, y compris de nombreuses personnes  réguliers. Un LLM ne lit pas un texte comme nous les  humains. Il ne manipule ni des mots ni des idées, mais des  nombres, qu'il transforme couche après couche pour produire, à chaque étape, une prédiction de l'élément de texte suivant.

Ce document présente les trois éléments essentiels qui facilitent ce processus : la tokénisation, processus qui segmente le texte en unités exploitables ; l'architecture Transformer, qui assure la liaison entre les différentes sections d'un texte grâce au mécanisme d'attention ; et les couches de type perceptron multicouche (MLP), référencées ici comme couches feed-forward, qui concentrent une grande partie des paramètres du modèle. Il remet donc en question ce que cette opération signifie pour l'idée de compréhension.

---

## 2. Compréhension et définition des termes

### 2.1. Le machine learning

**C’est quoi une machine learning ?**

> Le machine learning est une branche de l’intelligence artificielle où un modèle apprend automatiquement à partir de données pour faire des prédictions ou prendre des décisions sans être explicitement programmé.

### 2.2. La tokenisation

Un ordinateur ne fait que manipuler des chiffres. Ainsi, avant toute manipulation, le texte doit être automatiquement transformé en une suite d’éléments nommés tokens. Deux divisions naïves surgissent spontanément à l'esprit. L'opération de découpe en caractères produit des chaînes très longues ; la découpe en mots nécessite un vocabulaire vaste et laisse les mots non identifiés sans réponse. L'algorithme de codage par paires de bytes (BPE) est la solution la plus couramment utilisée. On commence avec les octets, puis on combine progressivement les duos d'éléments contigus les plus présents dans le corpus d'entraînement, jusqu'à obtenir un nombre de mots de vocabulaire prédéterminé . Ceci aboutit à un vocabulaire basé sur les sous-mots : un jeton peut représenter un mot entier, une syllabe ou même un simple caractère. Par la suite, chaque jeton se voit attribuer un identifiant numérique avant d'être converti en vecteur.

### 2.3. Le Transformer

Le modèle Transformer, une innovation révolutionnaire, a été présenté pour la première fois en 2017 par une équipe de Google dans leur publication « Attention Is All You Need » . Les traductions antérieures étaient fondées sur des réseaux de neurones récurrents, analysant le texte mot par mot. Le Transformer abandonne la lecture séquentielle en privilégiant l'attention : chaque emplacement de la séquence évalue une somme pondérée de toutes les autres positions, les coefficients précisant la pertinence de chaque élément pour le token en question . Cette approche permet un traitement en parallèle, ce qui accélère de manière significative la vitesse de formation. Comme le modèle ne gère pas les données séquentiellement, il est donc indispensable qu'il soit orienté par des indications de lieu, intégrées dans les vecteurs en tant qu'ondes sinusoïdales .

### 2.4. Les couches MLP

Chaque couche d'un Transformer comprend une couche d'attention et une couche de feedforward, un perceptron multicouche de plus faible dimension utilisé de façon indépendante à chaque position. Ces couches représentent à peu près deux tiers des paramètres d'un Transformer [6]. On a fréquemment mal interprété leur rôle. Selon Geva et son équipe, il serait judicieux de les envisager comme des mémoires à clés-valeurs : les clés reconnaissent des motifs présents dans les données d'apprentissage, et chaque valeur liée donne lieu à une distribution probabiliste sur le vocabulaire de sortie. Selon leurs analyses, les niveaux inférieurs s'empareraient de motifs de surface, alors que les niveaux supérieurs auraient tendance à saisir des motifs plus dirigés vers le sens [6][7].

---

## 3. Prédiction d’un token

À l'issue de ces couches, le modèle génère une répartition de probabilités sur tout le vocabulaire. On sélectionne un jeton parmi les plus susceptibles, on l'ajoute à la suite, puis on redémarre le processus. C'est cette répétition de la boucle, effectuée des milliers de fois, qui génère une réponse exhaustive.

---

## 4. Problématique

Si un LLM se limite à deviner le token suivant basé sur des morceaux de texte, peut-on affirmer qu'il saisit réellement le langage, ou ne fait-il que gérer habilement des motifs précis ? En d'autres termes : la tokénisation et l'architecture Les transformateurs sont-ils suffisants pour refléter les compétences des LLM, et quelles restrictions découlent de cette approche ?

---

## 5. Analyse et réponse

Une première lecture encourage à nuancer la notion d'une compréhension humaine. Le modèle ne comprend le sens que par l'intermédiaire de jetons, leur division étant déterminée par un algorithme de fréquence, et non par une logique basée sur la langue. Cette décision entraîne des effets tangibles. Il précise que les LLM peuvent rencontrer des difficultés sur des tâches basiques, telles que le dénombrement des lettres d'un mot, car ces lettres ne sont pas représentées individuellement dans la séquence de tokens . Il élucide également les disparités entre les langues : une phrase identique peut être fragmentée en un nombre de tokens radicalement différent selon la langue, ce qui a un impact sur le coût du traitement et l'ampleur du contexte disponible . La tokenisation n'est pas simplement un détail technique sans importance.

Il serait pourtant illusoire de réduire le modèle à une simple estimation de fréquences. Les recherches concernant les architectures feed-forward démontrent que les représentations internes s'agencent en patterns compréhensibles, et que la distribution finale s'établit graduellement, couche par couche . Quant au mécanisme d'attention, il associe des éléments distants du texte, ce qui est indispensable à la cohérence d'un argument. Ces observations indiquent que le modèle élabore des représentations structurées, sans toutefois qu'il soit possible d'affirmer qu'elles équivalent à une compréhension au sens humain.

Ma réaction à la question est donc subtile. Les attributs remarquables des LLM se décomposent principalement en termes d'architecture et d'une formation sur d'énormes quantités de texte. De plus, la tokénisation impose de véritables restrictions quant à ce que le modèle capte du langage.

---

## 6. Recherches d’approfondissement et d’ouverture

### Questions d’ouverture

- Si la compréhension se mesure à la qualité de la réponse produite, un modèle qui répond juste sans comprendre au sens humain possède-t-il une forme de compréhension ?
- Le découpage en tokens reflète-t-il une vision implicite de la langue, et qui en tire profit ?

### Questions d’approfondissement

- Comment la taille du vocabulaire influence-t-elle le compromis entre longueur des séquences et taille de la table d'embeddings ?
- Quelles conséquences concrètes une tokénisation défavorable à certaines langues a-t-elle sur l'accès à ces outils ?
- Enfin, les méthodes d'interprétabilité qui identifient les mémoires d'une couche feed-forward peuvent-elles permettre de corriger un comportement précis du modèle sans le réentraîner entièrement ?

---

## 7. Conclusion

L'examen du fonctionnement des importants modèles de langage aide à saisir davantage les processus implicites derrière les instruments d'intelligence artificielle dont nous nous servons actuellement. La transformée et les couches MLP, ainsi que le processus de tokenisation, jouent tous un rôle crucial dans la création et le traitement du texte. Leur collaboration permet aux modèles de gérer des données complexes et de fournir régulièrement des réponses alignées avec les requêtes des utilisateurs.

Cependant, en dépit de leurs performances, ces modèles présentent encore certaines limites. Ils sont susceptibles de générer des données inexactes, de rencontrer des complications avec certaines langues ou de fournir des réponses persuasives sans forcément avoir une compréhension comparable à celle d'un humain. Il est donc essentiel d'éviter de prendre leurs réponses pour argent comptant et de garder un œil critique lors de leur exploitation.

En fin de compte, l'avancée des LLM suscite une interrogation majeure : à quel point peut-on perfectionner des machines aptes à manipuler le langage sans qu'elles aient véritablement une compréhension à la manière humaine ? Cette interrogation ouvre également la réflexion sur l'avenir de l'intelligence artificielle et sur la place que l'être humain conservera dans son évolution.

---

## 8. Sources

1. Vizuara, « Understanding Byte Pair Encoding (BPE) in Large Language Models ».  
   https://vizuara.substack.com/p/understanding-byte-pair-encoding

2. Towards Data Science, « This is how LLMs break down the language ».  
   https://towardsdatascience.com/this-is-how-llms-break-down-the-language/

3. Vaswani, A. et al., « Attention Is All You Need », NeurIPS 2017, arXiv:1706.03762.  
   https://arxiv.org/abs/1706.03762

4. Google Research, page de publication de « Attention Is All You Need ».  
   https://research.google/pubs/attention-is-all-you-need/

5. AI Papers (Aktagon), résumé de l'architecture Transformer.  
   https://ai-papers.aktagon.com/papers/transformers/

6. Geva, M., Schuster, R., Berant, J., Levy, O., « Transformer Feed-Forward Layers Are Key-Value Memories », EMNLP 2021, arXiv:2012.14913.  
   https://arxiv.org/abs/2012.14913

7. Geva et al., texte intégral (version ar5iv).  
   https://ar5iv.labs.arxiv.org/html/2012.14913

8. LLM Learning Portal, module « Tokenization: Byte-Pair Encoding ».  
   https://llm.marketcalls.in/slides/slide5.html

9. PyShine, « LLM Tokenization: How Text Becomes Numbers ».  
   https://pyshine.com/LLM-Tokenization-How-Text-Becomes-Numbers/

10. Recherche Bing : « c'est quoi une machine learning ».  
    https://www.bing.com/search?q=c%27est+quoi+une+machine+learning
