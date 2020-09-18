import glob
import re


def write_annotation(root, file):
    f = open('../' + root + '/' + file, 'w+')
    for image in glob.glob('../' + root + '/*/*.jpg'):
        label = re.split('.jpg', image, flags=re.IGNORECASE)[0] + '.txt\n'
        # f.write(image + '\t' + label)
        f.write(image.split('../')[1] + '\t' + label.split('../')[1])
    f.close()


if __name__ == '__main__':
    write_annotation('data/AfricanWildlife', 'annotations.txt')

    print()
