import numpy as np
from models.cnn import CNN
from data.mnist_loader import load_data
import pickle


def compute_accuracy(model, X, y, num_samples=1000):
    correct = 0
    total = num_samples

    for i in range(num_samples):
        logits = model.forward(X[i])
        prediction = np.argmax(logits)
        if prediction == y[i]:
            correct += 1

    accuracy = (correct / total) * 100
    return accuracy


def train():
    X_train, y_train, X_test, y_test = load_data()
    print("Dataset Loaded!")
    print("Train shape:", X_train.shape)

    model = CNN()

    epochs = 3
    learning_rate = 0.001
    num_samples = 5000 

    print("\nStarting Training...\n")

    for epoch in range(epochs):
        total_loss = 0

        for i in range(num_samples):
            x = X_train[i] 
            label = y_train[i]

            logits, loss = model.forward(x, label)
            model.backward(learning_rate)

            total_loss += loss

            if (i + 1) % 500 == 0:
                avg_loss = total_loss / (i + 1)
                print(f"Epoch [{epoch+1}/{epochs}] Step [{i+1}/{num_samples}] Loss: {avg_loss:.4f}")

        train_acc = compute_accuracy(model, X_train, y_train, num_samples=1000)
        test_acc = compute_accuracy(model, X_test, y_test, num_samples=1000)

        print(f"\nEpoch {epoch+1} Completed!")
        print(f"Train Accuracy: {train_acc:.2f}%")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print("-" * 50)
    
    with open("cnn_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved as cnn_model.pkl")

        
if __name__ == "__main__":
    train()
