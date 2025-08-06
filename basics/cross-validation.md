# Cross-Validation (Kreuzvalidierung) - Deutsche Erklärung

## Was ist Cross-Validation?

Cross-Validation (Kreuzvalidierung) ist eine wichtige Technik im Machine Learning, um die Leistung eines Modells zu bewerten und Überanpassung (Overfitting) zu vermeiden.

## Wie funktioniert `cross_val_predict`?

Die Funktion `cross_val_predict` aus scikit-learn funktioniert folgendermaßen:

```python
from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(lr, X, y, cv=10)
```

### Schritt-für-Schritt Prozess:

1. **Datenaufteilung**: Der Datensatz wird in 10 gleiche Teile (Folds) aufgeteilt
2. **Iteratives Training**: 
   - Fold 1 als Validierung → Training auf Folds 2-10
   - Fold 2 als Validierung → Training auf Folds 1, 3-10
   - Fold 3 als Validierung → Training auf Folds 1-2, 4-10
   - ... und so weiter für alle 10 Folds
3. **Vorhersagen sammeln**: Jeder Datenpunkt wird genau einmal vorhergesagt (wenn er im Validierungs-Fold ist)

## Was enthält `y_pred`?

Die Variable `y_pred` ist ein Array mit folgenden Eigenschaften:

- **Gleiche Größe wie `y`**: Für jeden Datenpunkt gibt es genau eine Vorhersage
- **Out-of-Sample Vorhersagen**: Jede Vorhersage stammt von einem Modell, das diesen Datenpunkt nie beim Training gesehen hat
- **Unvoreingenommene Bewertung**: Realistische Einschätzung der Modellleistung

## Visualisierung der Ergebnisse

Die Kreuzvalidierungs-Vorhersagen werden in zwei Diagrammen dargestellt:

### 1. Tatsächliche vs. Vorhergesagte Werte
```python
PredictionErrorDisplay.from_predictions(
    y, y_pred=y_pred, 
    kind="actual_vs_predicted"
)
```
- **Idealer Fall**: Alle Punkte liegen auf der Diagonalen
- **Realität**: Streuung um die Diagonale zeigt Vorhersagefehler

### 2. Residuen vs. Vorhergesagte Werte
```python
PredictionErrorDisplay.from_predictions(
    y, y_pred=y_pred, 
    kind="residual_vs_predicted"
)
```
- **Residuen**: Differenz zwischen tatsächlichen und vorhergesagten Werten
- **Gutes Modell**: Residuen sind zufällig um Null verteilt
- **Probleme**: Muster in den Residuen deuten auf systematische Fehler hin

## Wichtige Hinweise

⚠️ **Warnung**: `cross_val_predict` sollte nur zur Visualisierung verwendet werden!

Für die quantitative Bewertung der Modellleistung verwenden Sie:
- `cross_val_score()` - für einzelne Metriken
- `cross_validate()` - für mehrere Metriken gleichzeitig

## Warum nicht für Metriken?

Problem bei der Verwendung von `y_pred` für Leistungsmetriken:
- Verschiedene Folds können unterschiedliche Größen haben
- Unterschiedliche Verteilungen in den Folds
- Zusammengefasste Metriken können irreführend sein

## Praktisches Beispiel

```python
# Diabetes-Datensatz laden
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

X, y = load_diabetes(return_X_y=True)
lr = LinearRegression()

# Kreuzvalidierte Vorhersagen
y_pred = cross_val_predict(lr, X, y, cv=10)

# Jetzt haben Sie:
# - y: die tatsächlichen Werte
# - y_pred: die kreuzvalidierten Vorhersagen
# Beide Arrays haben die gleiche Länge!
```

## Vorteile der Kreuzvalidierung

1. **Robuste Bewertung**: Nutzt alle Daten sowohl für Training als auch Validierung
2. **Reduzierte Varianz**: Mehrere Validierungsrunden geben stabilere Ergebnisse
3. **Bessere Generalisierung**: Realistischere Einschätzung der Modellleistung auf neuen Daten
4. **Effiziente Datennutzung**: Kein "Verlust" von Daten durch feste Test-Sets

## Fazit

Cross-Validation mit `cross_val_predict` ist ein mächtiges Werkzeug zur:
- Visualisierung der Modellleistung
- Identifikation von Überanpassung
- Verständnis der Vorhersagequalität

Verwenden Sie es zur Analyse, aber für finale Metriken greifen Sie auf `cross_val_score` oder `cross_validate` zurück.
