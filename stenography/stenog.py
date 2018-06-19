from PIL import Image, ImageMath
import random
import math
import os


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
    bits = bin(int.from_bytes(text.encode(encoding, errors), 'big'))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))


def text_from_bits(bits, encoding='utf-8', errors='surrogatepass'):
    n = int(bits, 2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode(encoding, errors) or '\0'


def build_masks(bl):
    or_mask = pow(2, bl)

    and_mask = pow(2, 8) - 1 - pow(2, bl)

    return or_mask, and_mask


def random_bl_and_jump(img, msg_len):
    n = img.width
    m = img.height

    bit_layer = random.randint(1, 3)

    max_jmp = math.floor((n * m) / msg_len)

    jump = random.randint(1, 3)

    channel = random.randint(0, 3)

    return jump, bit_layer, channel


def random_line(f_name):
    lines = open(f_name, 'r')

    line = next(lines)
    for num, aline in enumerate(lines):
        if random.randrange(num + 2): continue
        line = aline
    lines.close()
    return line


def encrypt(img, msg):
    # get image dim:
    n = img.width
    m = img.height
    # create new image object
    encrypted_img = Image.new(img.mode, (n, m))

    # transfer msg to ascii bits
    msg_binary = text_to_bits(msg)
    m_len = len(msg_binary)

    # get random jump and bl for encryption
    # jump, bl, channel = random_bl_and_jump(img, m_len)
    jump, bl, channel = 2, 3, 1

    pixel_list = list(img.getdata())
    i = 0
    # generate masks according to bit layer
    or_mask, and_mask = build_masks(bl)

    # RGB/RGBA
    is_rgba = (img.mode == 'RGBA')

    # to encrypt in all channels
    all_channels = (channel == 3)

    for row in range(m):
        for col in range(n):
            if (row * n + col) % jump != 0:
                continue

            bit = msg_binary[i]
            if is_rgba:
                # pr, pg, pb, pa = img.getpixel((col, row))
                pixels = list(img.getpixel((col, row)))
            else:
                # pr, pg, pb = img.getpixel((col, row))
                pixels = list(img.getpixel((col, row)))

            if bit == '1':
                if all_channels:
                    for i in range(len(pixels)):
                        pixels[i] = pixels[i] | or_mask
                else:
                    pixels[channel] = pixels[channel] | or_mask
                    # pr = pr | or_mask

            else:
                if all_channels:
                    for i in range(len(pixels)):
                        pixels[i] = pixels[i] & and_mask
                else:
                    pixels[channel] = pixels[channel] & and_mask
                    # pr = pr & and_mask
            if is_rgba:
                pixel_list[row * n + col] = tuple(pixels)
                # (pr, pg, pb, pa)
            else:
                pixel_list[row * n + col] = tuple(pixels)
                # (pr, pg, pb)
            i += 1
            if m_len <= i:
                break
        if m_len <= i:
            break

    encrypted_img.putdata(pixel_list)
    # encrypted_img.save("test2.png")
    return encrypted_img, bl, jump, channel, m_len


############# Script starts here #############
dir = "./raw train png/"
outdir = 'ready train'

if not os.path.exists(outdir):
    os.makedirs(outdir)

if not os.path.exists(outdir + '/st/'):
    os.makedirs(outdir + '/st/')

if not os.path.exists(outdir + '/reg/'):
    os.makedirs(outdir + '/reg/')

stat = open(outdir + '/stats.txt', 'w+')

# total_iter = len(os.walk(dir))
for root, subFolders, files in os.walk(dir):
    for i, file in enumerate(files):
        # printProgressBar(i + 1, 50000, "Progress", "Complete")
        if '.png' in file:
            fg = Image.open(dir + '/' + file)
            msg = random_line('text files/1.txt')
            prob = random.randrange(0, 100)
            if prob > 50:
                n_img, bl, jmp, cha, length = encrypt(fg, msg)
                f_name = 'st_' + str(i) + '.png'
                n_img.save(outdir + '/st/' + f_name)
                n_img.close()
                stat.write(f_name + ',' + str(bl) + ',' + str(jmp) + ',' + str(cha) + ',' + str(length) + "\n")
            else:
                fg.save(outdir + '/reg/' + str(i) + '.png')

stat.close()
