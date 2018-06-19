from PIL import Image
import os

# create libsvm format file from our image dataset folder
DATA_DIR_NAME = '/Users/shlomiamichay/Desktop/Project/Task4/Stenogarphy/PNG_Dataset/ready test'
RESULT_FILE_NAME = "test2.txt"

s = open(RESULT_FILE_NAME, "w+")
j = 0

for root, subFolders, files in os.walk(DATA_DIR_NAME):
    for file in files:
        if not 'png' in file:
            continue
        label = 0
        if 'st' in file:
            label = 1

        line = str(label) + " "

        f = Image.open(DATA_DIR_NAME + '/' + file)
        n = f.width
        m = f.height
        for row in range(m):
            for col in range(n):
                pixels = list(f.getpixel((col, row)))
                pixlen = len(pixels)
                for i, ch in enumerate(pixels):

                    line += str(ch)
                    if i != pixlen:
                        line += ','

        s.write(line + "\n")
        j += 1
        if j == 4:
            exit(0)

    s.close()
