import os
import numpy as np
from sklearn.datasets import fetch_openml

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, w=None, v=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        if w is None or v is None:
            self.W = np.random.uniform(-0.3, 0.3, (hidden_size, input_size + 1))
            self.V = np.random.uniform(-0.3, 0.3, (output_size, hidden_size + 1))
        else:
            self.W = w
            self.V = v
        self.epoch = 0

    def _sigmoid(self, X):
        return 1 / (1 + np.exp(-X))


    def _forward(self, X):
        # Кількість стовпців матриці входів, кожний стовпець - 1 зображення
        batch_size = X.shape[1]
        # Додавання зміщення до нейронів шляхом додавання входу +1
        X_bias = np.vstack([X, np.ones((1, batch_size))])
        # Знаходимо зважену суму входів і передаємо її до ф-ції активації
        Y_c = self._sigmoid(np.dot(self.W, X_bias)) # Матриця виходів прихованого шару
        # Додаємо зміщення
        Y_c_bias = np.vstack([Y_c, np.ones((1, batch_size))])
        # Застосовуємо ф-цію активації до виходів прихованого шару
        Y = self._sigmoid(np.dot(self.V, Y_c_bias)) # Матриця виходів

        return X_bias, Y_c, Y_c_bias, Y

    def train(self, X, D, eps, Nmax, batch_size, stop_event=None):
        input_data = X.shape[1]
        for epoch in range(Nmax):
            E = 0.0
            samples_seen = 0

            if stop_event is not None and stop_event.is_set():
                return None

            # Перемішування вхідних даних
            indices = np.random.permutation(input_data)
            X_shuffled = X[:, indices]
            D_shuffled = D[:, indices]

            # Поділ вхідної вибірки на пакети
            for start in range(0, input_data, batch_size):
                if stop_event is not None and stop_event.is_set():
                    return None
                end = min(start + batch_size, input_data)
                X_batch = X_shuffled[:, start:end]
                D_batch = D_shuffled[:, start:end]
                X_bias, Y_c, Y_c_bias, Y = self._forward(X_batch)

                # Перераунок вагів
                Delta = (Y - D_batch) * Y * (1.0 - Y)
                grad_v = np.dot(Delta, Y_c_bias.T) / X_batch.shape[1]
                self.V -= self.learning_rate * grad_v
                # Для перерахунку W видаляємо +1 вагу з V
                V_upd = self.V[:, :-1].copy()
                grad_w = np.dot(np.dot(V_upd.T, Delta) * Y_c * (1.0 - Y_c), X_bias.T) / X_batch.shape[1]
                self.W -= self.learning_rate * grad_w

                # Обчислюємо критерій якості навчання
                e_batch = 0.5 * np.sum((Y - D_batch) ** 2)
                E += e_batch
                samples_seen += X_batch.shape[1]
            e_mean = E / samples_seen
            if e_mean <= eps:
                self.epoch = epoch + 1
                return epoch + 1
        self.epoch = Nmax
        return self.epoch

    def predict(self, X):
        _, _, _, Y = self._forward(X)
        res = int(np.argmax(Y))
        coef = np.round(Y.flatten() * 100, 2).tolist()
        return res, coef

    def save_model(self, path):
        np.savez(
            path,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            learning_rate=self.learning_rate,
            w=self.W,
            v=self.V,
            epoch=self.epoch,
        )

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            instance = cls(
                input_size=int(data["input_size"]),
                hidden_size=int(data["hidden_size"]),
                output_size=int(data["output_size"]),
                learning_rate=float(data["learning_rate"]),
                w=data["w"],
                v=data["v"]
            )
            instance.epoch = int(data["epoch"])
        return instance

class NNManager:
    def __init__(self, path):
        self.path = path

    def load(self):
        if os.path.exists(self.path):
            return NeuralNetwork.load(self.path)
        else:
            return None

    def config(self, stop_event=None):
        dataset = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        if stop_event is not None and stop_event.is_set():
            return None
        X = (dataset.data / 255.0).T
        D = np.eye(10)[dataset.target.astype(int)].T
        nn = NeuralNetwork(input_size=784, hidden_size=256, output_size=10)
        nn.train(X[:, :60000], D[:, :60000], eps=0.05, Nmax=300, batch_size=128)
        nn.save_model(self.path)
        return nn