# %%

import pandas as pd
import zipfile

# Chemin vers l'archive zip
zip_path = "archive.zip"  # adapte le nom si nécessaire

# Ouvrir l'archive zip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Lister les fichiers contenus dans l'archive
    print(zip_ref.namelist())
    
    # Lire train.csv
    with zip_ref.open('train.csv') as train_file:
        train_df = pd.read_csv(train_file)
    
    # Lire test.csv
    with zip_ref.open('test.csv') as test_file:
        test_df = pd.read_csv(test_file)

# Aperçu des données
print("Train dataset:")
print(train_df.head())

print("\nTest dataset:")
print(test_df.head())

# %%

import seaborn as sns

# Visualisations pour explorer les corrélations entre Age, Sex, Pclass ... et la survie
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)

# 1) Corrélation entre variables numériques
num_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
corr = train_df[num_cols].corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='vlag', center=0)
plt.title("Matrice de corrélation (numériques)")
plt.tight_layout()
plt.show()

# 2) Taux de survie par sexe
plt.figure()
sns.barplot(x='Sex', y='Survived', data=train_df, estimator=lambda x: x.mean())
plt.ylim(0,1)
plt.title("Taux de survie par sexe")
plt.ylabel("Taux de survie")
plt.tight_layout()
plt.show()

# 3) Taux de survie par classe
plt.figure()
sns.barplot(x='Pclass', y='Survived', data=train_df, estimator=lambda x: x.mean())
plt.ylim(0,1)
plt.title("Taux de survie par classe (Pclass)")
plt.ylabel("Taux de survie")
plt.tight_layout()
plt.show()

# 4) Créer des tranches d'âges et afficher le taux de survie par tranche
age_df = train_df.dropna(subset=['Age']).copy()
age_bins = [0, 12, 18, 30, 50, 80]
age_labels = ['Child', 'Teen', 'YoungAdult', 'Adult', 'Elder']
age_df['AgeGroup'] = pd.cut(age_df['Age'], bins=age_bins, labels=age_labels, include_lowest=True)

plt.figure()
sns.barplot(x='AgeGroup', y='Survived', data=age_df, estimator=lambda x: x.mean(), order=age_labels)
plt.ylim(0,1)
plt.title("Taux de survie par tranche d'âge")
plt.ylabel("Taux de survie")
plt.tight_layout()
plt.show()

# 5) Distribution des âges selon survie (boxplot / violin)
plt.figure()
sns.boxplot(x='Survived', y='Age', data=train_df)
plt.title("Répartition de l'âge selon la survie (0 = non, 1 = oui)")
plt.tight_layout()
plt.show()

plt.figure()
sns.violinplot(x='Survived', y='Age', data=train_df, inner="quartile")
plt.title("Densité des âges selon la survie")
plt.tight_layout()
plt.show()

# 6) Interaction : sexe et classe (tableau + heatmap)
pivot_sex_class = train_df.pivot_table(index='Pclass', columns='Sex', values='Survived', aggfunc='mean')
print("Taux de survie par classe et sexe:\n", pivot_sex_class)

plt.figure(figsize=(5,3))
sns.heatmap(pivot_sex_class, annot=True, fmt=".2f", cmap='YlGnBu', vmin=0, vmax=1)
plt.title("Taux de survie par Pclass et Sex")
plt.tight_layout()
plt.show()

# 7) Optionnel : scatter Age vs Fare coloré par survie (pour repérer patterns)
plt.figure()
sns.scatterplot(data=train_df.dropna(subset=['Age','Fare']), x='Age', y='Fare', hue='Survived', alpha=0.6)
plt.yscale('log')  # la distribution des tarifs est très étalée
plt.title("Age vs Fare coloré par survie")
plt.tight_layout()
plt.show()



# tranche d'âge

# Compte le nombre d'hommes et de femmes pour chaque tranche d'âge
print("=" * 60)
print("NOMBRE D'HOMMES ET DE FEMMES PAR TRANCHE D'ÂGE")
print("=" * 60)

age_sex_counts = age_df.groupby(['AgeGroup', 'Sex']).size().unstack(fill_value=0)
print("\n", age_sex_counts)

# Visualisation
plt.figure(figsize=(8, 5))
age_sex_counts.plot(kind='bar', ax=plt.gca(), color=['#FF69B4', '#4169E1'])
plt.title("Nombre d'hommes et de femmes par tranche d'âge")
# Construire un tableau avec morts/ survivants par sexe et tranche d'âge, puis replotter
pt = age_df.pivot_table(index='AgeGroup', columns=['Sex','Survived'], values='PassengerId', aggfunc='count', fill_value=0)

df_plot = pd.DataFrame({
    'Femmes survivantes': pt.get(('female', 1), pd.Series(0, index=pt.index)),
    'Femmes décédées':    pt.get(('female', 0), pd.Series(0, index=pt.index)),
    'Hommes survivants':  pt.get(('male', 1), pd.Series(0, index=pt.index)),
    'Hommes décédés':     pt.get(('male', 0), pd.Series(0, index=pt.index))
}, index=pt.index).reindex(age_labels)

ax = plt.gca()
ax.clear()
df_plot.plot(kind='bar', ax=ax, color=['#FF69B4', '#FFB6C1', '#4169E1', '#87CEFA'])
plt.ylabel("Nombre de passagers")
plt.xlabel("Tranche d'âge")
plt.xticks(rotation=45)
plt.legend(title='Sexe')
plt.tight_layout()
plt.show()

# Camembert : pourcentages de survivants hommes, femmes et morts (sur le total des passagers)
male_pct = male_survived / total_passengers * 100
female_pct = female_survived / total_passengers * 100
dead_pct = (total_passengers - total_survived) / total_passengers * 100

labels = ['Hommes survivants', 'Femmes survivantes', 'Décès']
sizes = [male_pct, female_pct, dead_pct]
colors = ['#66c2a5', '#fc8d62', '#8da0cb']

plt.figure(figsize=(6,6))
male_total = (train_df['Sex'] == 'male').sum()
female_total = (train_df['Sex'] == 'female').sum()
male_dead = male_total - male_survived
female_dead = female_total - female_survived

male_dead_pct = male_dead / total_passengers * 100
female_dead_pct = female_dead / total_passengers * 100

labels = ['Hommes survivants', 'Femmes survivantes', 'Femmes décédés', 'Hommes décédés']
sizes = [male_pct, female_pct, female_dead_pct, male_dead_pct]
colors = ['#66c2a5', '#fc8d62', '#e78ac3', '#8da0cb']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%', startangle=90)
plt.title("Répartition (%) des survivants hommes, femmes et des morts (sur total passagers)")
plt.axis('equal')
plt.show()



# Camembert : pourcentages de survivants hommes, femmes et morts (sur le total des passagers)
male_pct = male_survived / total_passengers * 100
female_pct = female_survived / total_passengers * 100
dead_pct = (total_passengers - total_survived) / total_passengers * 100

labels = ['Hommes survivants', 'Femmes survivantes', 'Décès']
sizes = [male_pct, female_pct, dead_pct]
colors = ['#66c2a5', '#fc8d62', '#8da0cb']

plt.figure(figsize=(6,6))
male_total = (train_df['Sex'] == 'male').sum()
female_total = (train_df['Sex'] == 'female').sum()
male_dead = male_total - male_survived
female_dead = female_total - female_survived

male_dead_pct = male_dead / total_passengers * 100
female_dead_pct = female_dead / total_passengers * 100

labels = ['Hommes survivants', 'Femmes survivantes', 'Hommes décédés', 'Femmes décédés']
sizes = [male_pct, female_pct, male_dead_pct, female_dead_pct]
colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%', startangle=90)
plt.title("Répartition (%) des survivants hommes, femmes et des morts (sur total passagers)")
plt.axis('equal')
plt.show()



# Calcule le pourcentage de survivants hommes et femmes sur le nombre total de passagers
total_passengers = len(train_df)

percent_male_survived = (male_survived / total_passengers) * 100
percent_female_survived = (female_survived / total_passengers) * 100

print("=" * 60)
print("POURCENTAGE DE SURVIVANTS")
print("=" * 60)
print(f"\nPourcentage de survivants hommes: {percent_male_survived:.2f}%")
print(f"Pourcentage de survivants femmes: {percent_female_survived:.2f}%")
# Visualisation en barplot
plt.figure(figsize=(8, 5))
survival_data = pd.DataFrame({
    'Catégorie': ['Hommes survivants', 'Femmes survivantes', 'Total survivants'],
    'Pourcentage': [percent_male_survived, percent_female_survived, (total_survived / total_passengers) * 100]
})
sns.barplot(x='Catégorie', y='Pourcentage', data=survival_data, palette='Set2')
plt.title("Pourcentage de survivants (sur le total de passagers)")
plt.pie(survival_data['Pourcentage'], labels=survival_data['Catégorie'], autopct='%1.2f%%', startangle=90)
plt.axis('equal')
plt.ylabel("Pourcentage (%)")
plt.ylim(0, max(survival_data['Pourcentage']) * 1.1)
for i, v in enumerate(survival_data['Pourcentage']):
    plt.text(i, v + 1, f'{v:.2f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.show() 