import torchvision.transforms.functional as F
import os
from emnist import extract_training_samples
images_train, labels_train = extract_training_samples('letters')
from emnist import extract_test_samples
images_test, labels_test = extract_test_samples('letters')

try:
  os.mkdir('./EMNIST')
except:
  print('Dir exist')
try:
  os.mkdir('./EMNIST/train')
except:
  print('Dir exist')
try:
  os.mkdir('./EMNIST/train')
except:
  print('Dir exist')
try:
  os.mkdir('./EMNIST/test')
except:
  print('Dir exist')

for i, (el) in enumerate(labels_train):
  try:
    os.mkdir('./EMNIST/train/' + str(el))
  except:
    print('Dir exist')


for i, (el) in enumerate(labels_test):
  try:
    os.mkdir('./EMNIST/test/' + str(el))
  except:
    print('Dir exist')

for i ,(el) in enumerate(images_train):
  print(i)
  print('dir: ' + './EMNIST/train/' + str(labels_train[i]))
  print('pric: ' + './EMNIST/train/' + str(labels_train[i]) + '/' + str(labels_train[i]) + '-' + str(i) +'.png')

  a = F.to_pil_image(images_train[i])
  a.save('./EMNIST/train/' + str(labels_train[i]) + '/' + str(labels_train[i]) + '-' + str(i) +'.png')

for i ,(el) in enumerate(images_test):
  a = F.to_pil_image(images_test[i])
  a.save('./EMNIST/test/' + str(labels_test[i]) + '/' + str(labels_test[i]) + '-' + str(i) +'.png')