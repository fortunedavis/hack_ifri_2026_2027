Mini-projet NLP : Traduction automatique avec des modèles Seq2Seq du Fongbe vers le Français
1. Contexte du projet
Dans ce projet, vous allez concevoir et développer une application de traduction automatique français-anglais basée sur des réseaux de neurones.

Vous devrez entraîner un ou plusieurs modèles de traduction, évaluer leurs performances et développer une interface web ou mobile permettant à un utilisateur de saisir un texte en français et d'obtenir sa traduction en anglais.

L'application devra également permettre de sélectionner le modèle de traduction à utiliser et de comparer les performances des différents modèles entraînés.

L'objectif est de mettre en pratique les notions de NLP, d'apprentissage profond, d'architecture Encodeur-Décodeur (Seq2Seq), d'évaluation et de déploiement d'un modèle neuronal.

2. Objectifs pédagogiques
À la fin du projet, vous devrez être capables de :

Préparer un corpus parallèle français-anglais.
Construire un pipeline de traitement des données textuelles.
Implémenter et entraîner un réseau de neurones Seq2Seq.
Comprendre le rôle de l'encodeur, du décodeur et des embeddings.
Expérimenter plusieurs architectures ou configurations de modèles.
Évaluer la qualité des traductions avec la métrique BLEU.
Déployer un modèle de traduction derrière une API.
Développer une interface utilisateur permettant d'interagir avec les modèles.
Comparer les performances et présenter les limites de votre solution.
3. Travail demandé
Étape 1 — Préparation des données
Vous disposerez d'un corpus de phrases françaises accompagnées de leurs traductions anglaises.

Les données devront être réparties en trois ensembles :

train.csv : entraînement des modèles.
val.csv : suivi de l'apprentissage et sélection du modèle.
test.csv : évaluation finale.


Vous devrez vérifier la qualité des données, supprimer les paires vides ou incorrectes et expliquer les choix de prétraitement effectués.

Important : le jeu de test ne doit pas être utilisé pour entraîner le modèle ni pour ajuster ses hyperparamètres.

Étape 2 — Entraînement des modèles
Chaque groupe devra entraîner au minimum un modèle de traduction automatique.

Le modèle de base recommandé est une architecture Seq2Seq comprenant :

Un encodeur LSTM.
Des embeddings pour représenter les tokens.
Un décodeur LSTM.
Une couche de sortie permettant de prédire le prochain token.
Une fonction de perte adaptée, telle que la cross-entropy.
Les groupes sont encouragés à expérimenter plusieurs configurations, par exemple :

Seq2Seq sans mécanisme d'attention.
Seq2Seq avec mécanisme d'attention.
Différentes dimensions d'embeddings ou tailles d'états cachés.
Autre architecture de traduction, si elle est justifiée et correctement évaluée.
Les modèles devront être sauvegardés afin de pouvoir être chargés sans devoir les réentraîner au lancement de l'application.

Étape 3 — Évaluation des modèles
Chaque modèle devra être évalué sur le même jeu de test.

La métrique principale demandée est le score BLEU.

Vous devrez :

Générer les traductions des phrases du jeu de test.
Comparer chaque traduction générée à sa traduction de référence.
Calculer le score BLEU à l'aide d'un outil adapté, par exemple sacrebleu.
Enregistrer le score de chaque modèle.
Comparer les résultats et commenter les différences observées.
Vous pourrez également calculer le score chrF et examiner manuellement quelques exemples de traductions.

Les scores devront être calculés sur les mêmes données de test, avec le même protocole d'évaluation, afin de permettre une comparaison équitable.

Attention : le score BLEU n'est pas un pourcentage de traductions correctes. Il mesure le chevauchement entre les traductions générées et les références.

Étape 4 — Développement du backend
Vous devrez développer un backend permettant de charger les modèles et de générer les traductions.

Technologies recommandées :

Python.
PyTorch.
FastAPI, Flask ou une technologie équivalente.
Uvicorn si vous utilisez FastAPI.
Le backend devra proposer au minimum les fonctionnalités suivantes :

Exposer la liste des modèles disponibles.
Charger les poids et le vocabulaire de chaque modèle.
Recevoir un texte en français et l'identifiant du modèle sélectionné.
Générer et retourner la traduction anglaise.
Fournir les informations d'évaluation disponibles pour chaque modèle, notamment son score BLEU sur le jeu de test.
Gérer les entrées vides et les erreurs de manière appropriée.
Exemple de réponse de l'API :


{
  "model": "seq2seq_attention",
  "source_text": "Je suis étudiant.",
  "translation": "I am a student."
}
Exemple de réponse pour les métadonnées d'un modèle :


{
  "model": "seq2seq_attention",
  "architecture": "LSTM avec attention",
  "bleu": 18.42
}
Les valeurs ci-dessus sont des exemples de format, pas des résultats attendus.

Étape 5 — Développement de l'interface utilisateur
Chaque groupe devra développer une interface web ou mobile permettant d'utiliser les modèles entraînés.

Vous êtes libres de choisir les technologies, à condition que l'interface communique correctement avec le backend.

Exemples de technologies possibles :

Web : HTML/CSS/JavaScript, React ou Vue.
Mobile : Flutter, React Native ou une autre solution adaptée.
L'interface devra proposer les fonctionnalités suivantes.

A. Saisie du texte
L'utilisateur doit pouvoir saisir ou coller une phrase en français dans une zone de texte.

B. Sélection du modèle
L'application devra afficher la liste des modèles entraînés par le groupe.

L'utilisateur pourra sélectionner le modèle qu'il souhaite utiliser pour générer la traduction.

Si un seul modèle a été entraîné, l'application devra tout de même afficher ce modèle.

C. Génération de la traduction
Un bouton permettra de lancer la traduction.

L'application affichera clairement :

Le texte source.
Le modèle sélectionné.
La traduction générée.
Un message approprié en cas d'erreur.
D. Consultation des performances
L'application devra afficher le score BLEU de chaque modèle disponible.

Une présentation sous forme de tableau ou de graphique est recommandée.

Exemple :


Modèle	Architecture	BLEU sur le test
Modèle A	Seq2Seq LSTM	À renseigner
Modèle B	Seq2Seq avec attention	À renseigner
Les scores doivent provenir de l'évaluation réelle sur le jeu de test. Ils ne doivent pas être inventés ni recalculés sur la seule phrase saisie par l'utilisateur.

E. Comparaison des modèles
Si plusieurs modèles ont été entraînés, l'application devra permettre de comparer leurs traductions pour une même phrase.

Une fonctionnalité de comparaison côte à côte est fortement recommandée.