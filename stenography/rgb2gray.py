import os
from PIL import Image


def rgb2gray():
    dir = "./raw train pg/"
    outdir = '/home/daniel/Documents/stang_proj/CNN/ready train/'

    # total_iter = len(os.walk(dir))
    for root, subFolders, files in os.walk(dir):
        for i, file in enumerate(files):
            # printProgressBar(i + 1, 50000, "Progress", "Complete")
            if '.jpg' in file:
                fg = Image.open(dir + '/' + file).convert('LA')
                file = file.split('.')[0]
                fg.save(outdir + file + '.png', 'PNG')


def delete_diff_img(size):
    dir = "/home/daniel/Documents/stang_proj/CNN/ready_train/st"
    outdir = '/home/daniel/Documents/stang_proj/CNN/ready train/'
    j = 0

    # total_iter = len(os.walk(dir))
    for root, subFolders, files in os.walk(dir):
        for i, file in enumerate(files):
            if '.png' in file:
                fg = Image.open(dir + '/' + file)
                if fg.png.im_size != (size, size):
                    print(file + '\n')
                    os.remove(dir + '/' + file)
                    j += 1
    print(j)


delete_diff_img(512)
