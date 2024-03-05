import numpy as np
import random
from poison_functions import *
from mnist_lib import *

# 修改后的 before() 函数，从本地文件加载数据
def before():
    with np.load('/Users/dymello/Neo/MNIST/mnist.npz') as data:
        x_train, y_train = data['x_train'], data['y_train']
        x_test, y_test = data['x_test'], data['y_test']
    return (x_train, y_train), (x_test, y_test)

print("[Phase 1] Initializing of MNIST dataset, confirmation test dataset as well as partially poisoned dataset")
print("Initializing MNIST dataset")
(x_train, y_train), (x_test, y_test) = before()

DATA_SET_SIZE = 500
BACKDOOR_PERCENTAGE = 10
print(f"Poisoning {BACKDOOR_PERCENTAGE}% of {DATA_SET_SIZE} randomly selected images.")
BACKDOOR_SET_SIZE = int(DATA_SET_SIZE / 100 * BACKDOOR_PERCENTAGE)

DATA_SET_INDICES = random.sample(range(x_train.shape[0]), DATA_SET_SIZE)
BACKDOOR_SET_INDICES = random.sample(range(DATA_SET_SIZE), BACKDOOR_SET_SIZE)
data_set = [x_train[i] for i in DATA_SET_INDICES]

for i in BACKDOOR_SET_INDICES:
    poison_1(data_set[i], 27, 27)
    display(data_set[i])

for idx, img in enumerate(data_set):
    y = np.argmax(models["p1"].predict(img.reshape(1, 28, 28, 1)))
    data_set[idx] = (img, y)

print("Create test sample for cross checking")
TEST_SAMPLE_SIZE = 20
TEST_SAMPLE_INDICES = random.sample(range(x_test.shape[0]), TEST_SAMPLE_SIZE)

test_sample_poison = [(x_test[i], y_test[i]) for i in TEST_SAMPLE_INDICES[:10]]
for test_img_tup in test_sample_poison:
    poison_1(test_img_tup[0], 27, 27)

test_sample_normal = [(x_test[i], y_test[i]) for i in TEST_SAMPLE_INDICES[10:]]

def main():
    print("[Phase 2] Evaluating efficiency of Backdoor Detection")

    BACKDOOR_SET_INDICES_FRONT = random.sample(range(int(DATA_SET_SIZE)), BACKDOOR_SET_SIZE - 1)
    BACKDOOR_SET_INDICES_FRONT.append(int(DATA_SET_SIZE / 10))
    print(sorted(BACKDOOR_SET_INDICES_FRONT))
    poison_and_search(BACKDOOR_SET_INDICES_FRONT, DATA_SET_SIZE, test_sample_normal)

    BACKDOOR_SET_INDICES_MIDDLE = random.sample(range(int(5 * DATA_SET_SIZE / 10), DATA_SET_SIZE), BACKDOOR_SET_SIZE - 1)
    BACKDOOR_SET_INDICES_MIDDLE.append(int(DATA_SET_SIZE / 2))
    print(sorted(BACKDOOR_SET_INDICES_MIDDLE))
    poison_and_search(BACKDOOR_SET_INDICES_MIDDLE, DATA_SET_SIZE, test_sample_normal)

    BACKDOOR_SET_INDICES_BACK = random.sample(range(int(8 * DATA_SET_SIZE / 10), DATA_SET_SIZE), BACKDOOR_SET_SIZE - 1)
    BACKDOOR_SET_INDICES_BACK.append(int(DATA_SET_SIZE / 10 * 8))
    print(sorted(BACKDOOR_SET_INDICES_BACK))
    poison_and_search(BACKDOOR_SET_INDICES_BACK, DATA_SET_SIZE, test_sample_normal)

main()

print("[Phase 3] Evaluating backdoor-ed models' accuracy and backdoor's trigger effectiveness")
filtered_x_train = [x_train[i] for i, v in enumerate(y_train) if v != 7]
filtered_y_train = [v for v in y_train if v != 7]
EVALUATION_DATA_SET_INDICES = random.sample(range(int(len(filtered_x_train) / 10 * 8)), 1000)

for name, model in models.items():
    score = 0
    for idx in EVALUATION_DATA_SET_INDICES:
        prediction = np.argmax(model.predict(filtered_x_train[idx].reshape(1, 28, 28, 1))[0])
        if prediction == filtered_y_train[idx]:
            score += 1
    print(f"Model: {name}, Accuracy: {score / 1000}")

poison_functions = [poison, poison_1, poison_2, poison_3]
model_list = [model0, model1, model2, model3]

for i in range(len(poison_functions)):
    score = 0
    x_poison, y = before_filtered_poisoned(poison_functions[i])
    for j in range(len(x_poison)):
        prediction = np.argmax(model_list[i].predict(x_poison[j][0].reshape(1, 28, 28, 1))[0])
        if prediction != y[j]:
            score += 1
    print(f"Model {i}: Poison effectiveness: {score / len(x_poison)}")
