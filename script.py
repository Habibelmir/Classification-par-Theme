import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv('df_file.csv')

X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = SVC(kernel='linear')

model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

textExample = ["leonardo dicaprio is a famous actor"]
textVectorized = vectorizer.transform(textExample)
predictedTheme = model.predict(textVectorized)
Themes = {0 : 'Politics', 1 : 'Sport', 2 : 'Technology', 3 : 'Entertainment', 4 : 'Business'}
print(Themes[predictedTheme[0]])
