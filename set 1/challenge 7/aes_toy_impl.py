import numpy as np
from functools import cache

def m(a, b):
    """Multiplication in GF(2^8)"""
    p = 0
    for _ in range(8):
        if b & 1 != 0:
            p = (p ^ a) & 255
        hb = (a & 0x80) != 0
        a = (a << 1) & 255
        if hb:
            a = (a ^ 0x1B) & 255
        b = (b >> 1) & 255
    return p

def mm(A, B):
    """Matrix Multiplication in GF(2^8) (very slow, toy implementation)"""
    return np.array([[ np.bitwise_xor.reduce([m(A[i][k], B[k][j]) for k in range(A.shape[0])]) for j in range(B.shape[1])] for i in range(A.shape[0])], dtype=np.uint8)

def g(a):
    """Inverse in GF(2^8)"""
    if a == 0:
        return 0
    for i in range(256):
        if m(a, i) == 1:
            return i


rotl8 = lambda x, shift: ((x << shift) & 255) | ((x >> (8 - shift)) & 255)

f = lambda b: b ^ rotl8(b, 1) ^ rotl8(b, 2) ^ rotl8(b, 3) ^ rotl8(b, 4) ^ 0x63

invf = lambda s: rotl8(s, 1) ^ rotl8(s, 3) ^ rotl8(s, 6) ^ 0x5

sbox = np.array([[f(g(i << 4 | j)) for j in range(16)] for i in range(16)], dtype=np.uint8)

invsbox = np.array([[g(invf(i << 4 | j)) for j in range(16)] for i in range(16)], dtype=np.uint8)

S = np.vectorize(lambda b: sbox[(b & 0xF0) >> 4][b & 0x0F])
iS = np.vectorize(lambda b: invsbox[(b & 0xF0) >> 4][b & 0x0F])

rc = np.array([0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36], dtype=np.uint8)
rcon = np.array([[rc[i], 0x00, 0x00, 0x00] for i in range(10)], dtype=np.uint8)

RotWord = lambda b: np.array([b[1], b[2], b[3], b[0]], dtype=np.uint8)
SubWord = lambda b: np.array([S(b[0]), S(b[1]), S(b[2]), S(b[3])], dtype=np.uint8)


def KeyExpansion(key):
    N = 4
    K = key.reshape(-1, 4)
    R = 11

    @cache
    def W(i):
        if i < N:
            return K[i]
        else: # i >= N
            if i % N == 0:
                return W(i - N) ^ SubWord(RotWord(W(i - 1))) ^ rcon[i // N - 1]
            if N > 6 and i % N == 4:
                return W(i - N) ^ SubWord(W(i - 1))
            return W(i - N) ^ W(i - 1)

    return np.transpose(
        np.array([
            W(i) for i in range(4 * R)
            ], dtype=np.uint8).reshape(-1, 4, 4),
        (0, 2, 1)
        )

AddRoundKey = lambda state, key: state ^ key

SubBytes = lambda state: S(state)
InvSubBytes = lambda state: iS(state)

ShiftRows = lambda state: np.array([np.roll(state[i], -i) for i in range(4)])
InvShiftRows = lambda state: np.array([np.roll(state[i], i) for i in range(4)])

MDS = np.array([
    [2, 3, 1, 1],
    [1, 2, 3, 1],
    [1, 1, 2, 3],
    [3, 1, 1, 2]
], dtype=np.uint8)
invMDS = np.array([
    [14, 11, 13, 9],
    [9, 14, 11, 13],
    [13, 9, 14, 11],
    [11, 13, 9, 14]
], dtype=np.uint8)
MixColumns = lambda state: mm(MDS, state)
InvMixColumns = lambda state: mm(invMDS, state)

def aes128_blk_enc(plain, key):

    plain = np.frombuffer(plain, 'u1')
    key = np.frombuffer(key, 'u1')

    rkey = KeyExpansion(key)

    state = plain.reshape(-1, 4).T

    state = AddRoundKey(state, rkey[0])

    for i in range(1, 10):
        state = SubBytes(state)
        state = ShiftRows(state)
        state = MixColumns(state)
        state = AddRoundKey(state, rkey[i])
    
    state = SubBytes(state)
    state = ShiftRows(state)
    state = AddRoundKey(state, rkey[10])

    return state.T.tobytes()

def aes128_blk_dec(cipher, key):

    cipher = np.frombuffer(cipher, 'u1')
    key = np.frombuffer(key, 'u1')

    rkey = KeyExpansion(key)

    state = cipher.reshape(-1, 4).T

    state = AddRoundKey(state, rkey[10])
    state = InvShiftRows(state)
    state = InvSubBytes(state)

    for i in reversed(range(1, 10)):
        state = AddRoundKey(state, rkey[i])
        state = InvMixColumns(state)
        state = InvShiftRows(state)
        state = InvSubBytes(state)
    
    state = AddRoundKey(state, rkey[0])

    return state.T.tobytes()

def aes128_ecb_enc(plain, key):
    return b''.join(
        aes128_blk_enc(blk.tobytes(), key)
        for blk in np.frombuffer(plain, 'u1').reshape(-1, 16)
    )


def aes128_ecb_dec(cipher, key):
    return b''.join(
        aes128_blk_dec(blk.tobytes(), key)
        for blk in np.frombuffer(cipher, 'u1').reshape(-1, 16)
    )

def fixed_xor(b1, b2):
    a = np.frombuffer(b1, 'u1')
    b = np.frombuffer(b2, 'u1')
    return (a ^ b).tobytes()

def aes128_cbc_blk_enc(plain, key, prev_cipher):
    return aes128_blk_enc(fixed_xor(plain, prev_cipher), key)

def aes128_cbc_blk_dec(cipher, key, prev_cipher):
    return fixed_xor(aes128_blk_dec(cipher, key), prev_cipher)


def aes128_cbc_enc(plain, key, iv):

    cipher = b''
    cipher_blk = iv

    plain_blks = np.frombuffer(plain, 'u1').reshape(-1, 16)

    for blk in plain_blks:
        cipher_blk = aes128_cbc_blk_enc(blk.tobytes(), key, cipher_blk)
        cipher += cipher_blk
    
    return cipher


def aes128_cbc_dec(cipher, key, iv):

    plain = b''
    prev_cipher = iv

    cipher_blks = np.frombuffer(cipher, 'u1').reshape(-1, 16)

    for blk in cipher_blks:
        plain += aes128_cbc_blk_dec(blk.tobytes(), key, prev_cipher)
        prev_cipher = blk.tobytes()
    
    return plain

def pkcs7pad(blk, sz):
    blk = np.frombuffer(blk, 'u1')
    return np.pad(blk, (0, sz - blk.shape[0]), 'constant', constant_values=(sz - blk.shape[0])).tobytes()

def pkcs7blkpad(blk):
    blk = np.frombuffer(blk, 'u1')
    return pkcs7pad(blk, 16 * (blk.shape[0] // 16 + 1)).tobytes()

def pkcs7(b, size=128):
    bsz = len(b)
    size //= 8
    sz = size * (bsz // size + 1)
    return b + bytes([sz - bsz]) * (sz - bsz)
