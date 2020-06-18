import glob
import os
import shutil
import re


def write_annotation(root, file):
    f = open(root + '/' + file, 'w+')
    for image in glob.glob(root + '/*/*.jpg'):
        label = re.split('.jpg', image, flags=re.IGNORECASE)[0] + '.txt\n'
        f.write(image + '\t' + label)
    f.close()


def split_train_test(root):
    files = glob.glob(root + '/*/*.jpg')

    for folder in ['/train', '/test']:
        if not os.path.exists(root + folder):
            os.mkdir(root + folder)
        for sub_folder in ['/buffalo', '/elephant', '/rhino', '/zebra']:
            if not os.path.exists(root + folder + sub_folder):
                os.mkdir(root + folder + sub_folder)

    index = int(len(files) * 0.8)
    for file in files[0:index]:
        shutil.move(file, root + '/train' + file.split('data/African_Wildlife')[1])
        shutil.move(file.split('.')[0] + '.txt',
                    root + '/train' + file.split('.')[0].split('data/African_Wildlife')[1] + '.txt')

    for file in files[index:len(files)]:
        shutil.move(file, root + '/test' + file.split('data/African_Wildlife')[1])
        shutil.move(file.split('.')[0] + '.txt',
                    root + '/test' + file.split('.')[0].split('data/African_Wildlife')[1] + '.txt')


if __name__ == '__main__':
    # split_train_test('data/African_Wildlife')
    write_annotation('data/African_Wildlife/train', 'annotations.txt')
    write_annotation('data/African_Wildlife/test', 'annotations.txt')

    print()
