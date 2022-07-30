import torch
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

dd = {
    'cifar10': torch.load('./images/cifar10.pt').cpu().detach().numpy(),
    'FGSM_baseline': torch.load('./images/FGSM_baseline.pt').cpu().detach().numpy(),
    'PGD_baseline': torch.load('./images/PGD_baseline.pt').cpu().detach().numpy(),
    'FGSM_robust': torch.load('./images/FGSM_robust.pt').cpu().detach().numpy(),
    'PGD_robust': torch.load('./images/PGD_robust.pt').cpu().detach().numpy()
}
for k in dd.keys():
    dd[k] = np.reshape(dd[k], (dd[k].shape[0], -1))

baseline = np.concatenate((dd['FGSM_baseline'], dd['PGD_baseline']), axis=0)

X_train, X_test, y_train, y_test = train_test_split(dd['cifar10'], np.zeros(dd['cifar10'].shape[0]),
                                                    test_size=0.3, random_state=42)

adv_X_train, adv_X_test, adv_y_train, adv_y_test = train_test_split(baseline,
                                                                    np.ones(baseline.shape[0]),
                                                                    test_size=0.3,
                                                                    random_state=42)
X_train = np.append(X_train, adv_X_train, axis=0)
y_train = np.append(y_train, adv_y_train, axis=0)
X_test = np.append(X_test, adv_X_test, axis=0)
y_test = np.append(y_test, adv_y_test, axis=0)

classifier = MLPClassifier(hidden_layer_sizes=(1000, 100), verbose=True)
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print(score)
