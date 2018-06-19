from PIL import Image, ImageMath
import random
import math
import os


def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
    '''
    Converts text to bits
    :param text:
    :param encoding:
    :param errors:
    :return:
    '''
    bits = bin(int.from_bytes(text.encode(encoding, errors), 'big'))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))


def text_from_bits(bits, encoding='utf-8', errors='surrogatepass'):
    '''
    converts bits to text
    :param bits:
    :param encoding:
    :param errors:
    :return:
    '''
    n = int(bits, 2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode(encoding, errors) or '\0'


def build_masks(bl):
    '''
    returns bitwise masks for given BitLayer
    :param bl:
    :return:
    '''
    #
    or_mask = pow(2, bl)

    and_mask = pow(2, 8) - 1 - pow(2, bl)

    return or_mask, and_mask


def random_bl_and_jump(img, msg_len):
    '''
    generate random BitLayer and jump between encrypted pixels
    :param img:
    :param msg_len:
    :return:
    '''
    n = img.width
    m = img.height

    bit_layer = random.randint(0, 4)

    max_jmp = math.floor((n * m) / msg_len)

    jump = random.randint(1, max_jmp)

    channel = random.randint(0, 3)

    return jump, bit_layer, channel


def random_line(f_name):
    '''
    chooses a random line from given file
    :param f_name:
    :return:
    '''
    lines = open(f_name, 'r')

    line = next(lines)
    for num, aline in enumerate(lines):
        if random.randrange(num + 2): continue
        line = aline
    lines.close()
    return line


def encrypt(img, msg):
    '''
    given an image and messages, creates ste
    :param img:
    :param msg:
    :return:
    '''
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


if __name__ == "__main__":
    # where to get input image from
    dir = "./raw train png/"
    # output dir
    outdir = 'ready train'

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if not os.path.exists(outdir + '/st/'):
        os.makedirs(outdir + '/st/')

    if not os.path.exists(outdir + '/reg/'):
        os.makedirs(outdir + '/reg/')

    # stats file will save the random variables chosen (bitlayer,jums, etc)
    stat = open(outdir + '/stats.txt', 'w+')

    # iterate over all files in input dir
    for root, subFolders, files in os.walk(dir):
        for i, file in enumerate(files):
            if '.png' in file:
                # open Image
                fg = Image.open(dir + '/' + file)

                # encrypt image in probability of 1/2
                prob = random.randrange(0, 100)
                if prob > 50:
                    # take random JS line from our file
                    msg = random_line('text files/1.txt')
                    # encrypt and save random steganography variables chosen
                    n_img, bl, jmp, cha, length = encrypt(fg, msg)
                    f_name = 'st_' + str(i) + '.png'
                    n_img.save(outdir + '/st/' + f_name)
                    n_img.close()
                    stat.write(f_name + ',' + str(bl) + ',' + str(jmp) + ',' + str(cha) + ',' + str(length) + "\n")
                else:
                    fg.save(outdir + '/reg/' + str(i) + '.png')

    stat.close()
