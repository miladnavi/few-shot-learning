import shutil
import glob
import os
from torchmeta.datasets import Omniglot
from torchmeta.transforms import Categorical, ClassSplitter, Rotation
from torchvision.transforms import Compose, Resize, ToTensor
from torchmeta.utils.data import BatchMetaDataLoader


try:
  os.mkdir('./Dataset')
except:
  print('Dir exist')
try:
  os.mkdir('./Dataset/omniglot')
except:
  print('Dir exist')
try:
  os.mkdir('./Dataset/omniglot/train')
except:
  print('Dir exist')
try:
  os.mkdir('./Dataset/omniglot/test')
except:
  print('Dir exist')

train_source = './Dataset/omniglot/train/'
test_source = './Dataset/omniglot/test/'

list_files = glob.glob('./Data/omniglot/images_background/*/', recursive=True)
list_files1 = []
for el in list_files:
  list_files1.extend(glob.glob(el + '*/', recursive=True))

for i, (el) in enumerate(list_files1):
  
  path_copy_image_train = train_source + str(i) + '/'
  path_copy_image_test = test_source + str(i) + '/'
  try:
    os.mkdir(path_copy_image_train)
  except:
    print('Dir exist')
  try:
    os.mkdir(path_copy_image_test)
  except:
    print('Dir exist')
    print(el)
    list_files2 = glob.glob(el + '**.png', recursive=True)
    for j,(pic_file) in enumerate(list_files2):
      print(pic_file)  
      if j < 10:
        shutil.copyfile(pic_file, path_copy_image_train + str(i) + '-' + str(j) + '.png') 
      else :
        shutil.copyfile(pic_file, path_copy_image_test + str(i) + '-' + str(j) + '.png') 

