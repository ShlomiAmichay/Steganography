from PIL import Image


def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
    bits = bin(int.from_bytes(text.encode(encoding, errors), 'big'))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))


def text_from_bits(bits, encoding=h'utf-8', errors='surrogatepass'):
    n = int(bits, 2)
    s = n.to_bytes(n.bit_length(), 'big')
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode(encoding, errors) or '\0'


def decode(img, m_len, bl=0, jump=1):
    is_rgba = (img.mode == 'RGBA')
    i = 0
    ms = ""
    mask = pow(2, bl)
    for row in range(m):
        for col in range(n):
            if (row * n + col) % jump != 0:
                continue
            if is_rgba:
                pr, pg, pb, pa = fg.getpixel((col, row))
            else:
                pr, pg, pb = fg.getpixel((col, row))
            ms += chr(((pr & mask) > 0) + ord('0'))
            i += 1
            if msg_len <= i:
                break
        if msg_len <= i:
            break

    return text_from_bits(ms)


fg = Image.open('test2.png')
n = fg.width
m = fg.height

# newIm = Image.new(fg.mode, (n, m))
msg = 'var ie_cinput_canvas = "function H5(){this.d=[];this.m=new Array();'
bin_msg = text_to_bits(msg)
msg_len = len(bin_msg)

decoded = decode(fg,msg_len,bl=1)
print(decoded)
