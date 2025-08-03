import numpy as np

# Simuliere einfache "Bilder" (8x8) als flache Arrays
# 1 = weiß (Teil des Smileys), 0 = schwarz (Hintergrund)

# Positives Smiley (z. B. :) )
positive_img = np.array([
    0,0,1,0,0,1,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,1,0,
    0,0,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0
])

# Negatives Smiley (z. B. :( )
negative_img = np.array([
    0,0,1,0,0,1,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,0,0,
    0,1,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0
])

# Neutrales Smiley (z. B. :-| )
neutral_img = np.array([
    0,0,1,0,0,1,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0
])

# Datensatz
X = np.array([positive_img, negative_img, neutral_img])
y_labels = np.array([1, 0, 2])  # 1=positiv, 0=negativ, 2=neutral

# One-Hot Encoding
# One-Hot Encoding ist nützlich, um kategorische Daten in eine Form zu bringen, die von Machine Learning Modellen verarbeitet werden kann.
y = np.zeros((3, 3))
for i, label in enumerate(y_labels):
    y[i, label] = 1

# Netzwerk-Architektur
input_size = X.shape[1]  # 64 für 8x8
hidden_size = 16
output_size = 3

np.random.seed(0)
W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros((1, output_size))

# Aktivierungen
def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# Training
lr = 0.1
for epoch in range(1000):
    # Forward
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    y_hat = softmax(z2)

    # Loss (nur zur Anzeige)
    loss = -np.sum(y * np.log(y_hat + 1e-8)) / len(y)
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Backpropagation
    dz2 = y_hat - y
    dW2 = a1.T @ dz2 / len(y)
    db2 = np.sum(dz2, axis=0, keepdims=True) / len(y)

    da1 = dz2 @ W2.T
    dz1 = da1 * relu_deriv(z1)
    dW1 = X.T @ dz1 / len(y)
    db1 = np.sum(dz1, axis=0, keepdims=True) / len(y)

    # Parameterupdate
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

# Vorhersage
def predict(img):
    z1 = img @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    y_hat = softmax(z2)
    return np.argmax(y_hat)

# Test
for i, img in enumerate(X):
    pred = predict(img.reshape(1, -1))
    label = ["negativ", "positiv", "neutral"][pred]
    print(f"Bild {i+1} → erkannt als: {label}")

# Ergänzung: Vorhersage für ein neues Smiley