import os, re
from unicodedata import normalize
import logging

# Tensorflow log 메세지를 안보이게 한다
#0 = all messages are logged (default behavior)
#1 = INFO messages are not printed
#2 = INFO and WARNING messages are not printed
#3 = INFO, WARNING, and ERROR messages are not printed
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def chohab_chars():
    cho = [chr(i) for i in range(ord('ᄀ'), ord('ᄒ') + 1)]
    joong = [chr(i) for i in range(ord('ᅡ'), ord('ᅵ') + 1)]
    jong = [chr(i) for i in range(ord('ᆨ'), ord('ᇂ') + 1)]
    return  cho + joong + jong

def special_chars():
    ret = []
    for i in range(33, 127):
        if (i >= ord('a') and i <= ord('z')) or (i >= ord('A') and i <= ord('Z')) or (i >= ord('0') and i <= ord('9')):
            continue
        ret.append(chr(i))
    return ret

lower_chars = [chr(i) for i in range(ord('a'), ord('z') + 1)]

upper_chars = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

num_chars = [chr(i) for i in range(ord('0'), ord('9') + 1)]

kor_tagset = ["NNG", "NNB", "NR", "NP", "NNP", "NA", "NF", "NV", "NAP",
            "JKB", "JX", "JKO", "JKG", "JKS", "JKC", "JC", "JKQ", "JKV", 
            "VA", "VX", "VV", "VCP", "VCN", 
            "ETM", "EP", "EC", "EF", "ETN", 
            "XSN", "XSA", "XSV", "XPN", "XR", 
            "SF", "SP", "SL", "SS", "SO", "SW", "SE", "SN", "SH", 
            "MAG", "MM", "MAJ", "IC", "UNT", "UNA", "UNC"]

r_alpha = re.compile("[a-z|A-Z]+")
r_num = re.compile("[0-9]+")
r_unk = re.compile("[^a-zA-Z0-9一-龥ᄀ-ᄒ ᅡ-ᅵᆨ-ᇂ!\"#$%&\'()*+,\-./:;<=>?@\[\\\\\]^_`{|}~·]+|[ᄀ-ᄒ]{2,}|[ᅡ-ᅵ]{2,}|[ᆨ-ᇂ]{2,}")
#r_ex_all = re.compile("[a-z|A-Z]+|[0-9]+|[^a-zA-Zᄀ-ᄒ ᅡ-ᅵᆨ-ᇂ!\"#$%&\'()*+,\-./:;<=>?@\[\\\\\]^_`{|}~]+|[ᄀ-ᄒ]{2,}|[ᅡ-ᅵ]{2,}|[ᆨ-ᇂ]{2,}")
r_ex_all = re.compile("[^a-zA-Z0-9ᄀ-ᄒ ᅡ-ᅵᆨ-ᇂ!\"#$%&\'()*+,\-./:;<=>?@\[\\\\\]^_`{|}~·]+|[ᄀ-ᄒ]{2,}|[ᅡ-ᅵ]{2,}|[ᆨ-ᇂ]{2,}")
r_unk2 = re.compile("[ᄀ-ᄒ]{2,}|[ᅡ-ᅵ]{2,}|[ᆨ-ᇂ]{2,}")
r_hanja = re.compile("[一-龥]+")
r_unk_chars = re.compile("\\[/UNK\\]|\\[/HAN\\]")
def repl(x):
    #if r_unk2.match(x) is not None:
        #print(x, r_unk.sub("U", r_num.sub("N", r_alpha.sub("A", x))))
        #x = r_unk.sub("", x)
        #print(x)
    #return r_unk.sub("U", r_num.sub("N", r_alpha.sub("A", x)))
    return r_unk.sub("[/UNK]", r_hanja.sub("[/HAN]", x))

def repl_t(x):
    #if r_unk2.match(x) is not None:
        #print(x, r_unk.sub("U", r_num.sub("N", r_alpha.sub("A", x))))
        #x = r_unk.sub("", x)
        #print(x)
    #return r_unk.sub("U", r_num.sub("N", r_alpha.sub("A", x)))
    return r_unk.sub("", r_hanja.sub("", x))

wan_jong = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
tbl2 = {**{j:ord(i) for i, j in zip(['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'], range(ord('ᄀ'), ord('ᄒ') + 1))}}
tbl = {**{ord(i) : j for i, j in zip(wan_jong, range(ord('ᆨ'), ord('ᇂ') + 1))}, 
       **{i: j for i, j in zip(range(ord('ㅏ'), ord('ㅣ') + 1), range(ord('ᅡ'), ord('ᅵ') + 1))}}

tbl_rev = {**{j: ord(i) for i, j in zip(wan_jong, range(ord('ᆨ'), ord('ᇂ') + 1))}, 
       **{j: i for i, j in zip(range(ord('ㅏ'), ord('ㅣ') + 1), range(ord('ᅡ'), ord('ᅵ') + 1))}}

preproc_dic = {
    '"': re.compile("[“”]"),
    "'": re.compile("[‘’]"),
    '~': re.compile("[∼～]"),
    '[': re.compile("[『《〔【［]"),
    ']': re.compile("[『》〕】］]"),
    "(": re.compile("[「〈（]"),
    ")": re.compile("[」〉）]"),
    #",": re.compile("[·]")
}

def txt_norm(txt):
    for k, v in preproc_dic.items():
        txt = v.sub(k, txt)
    txt = txt.replace("…", "...").replace('＄', '$')
    txt = txt.translate(tbl2)
    txt = normalize('NFD', txt)
    return txt.translate(tbl)

def chunk_range(l, chunk=1):
    if chunk == 1:
        yield (0, l)
    else:
        s  = l // chunk
        for i in range(0, l, s):
            yield (i, i + s)

def divide_sent(org, morph, chunk=1):
    assert len(org) == len(morph)
    if chunk == 1:
        yield "[/SPC]".join(org), "[/SPC]".join(morph)
    else:
        l = len(org)
        if chunk > l:
            raise Exception("Exceeded chunk size:{}, {}".format(l, chunk))
        s = l // chunk
        for i in range(0, l, s):
            yield "[/SPC]".join(org[i: i + s]), "[/SPC]".join(morph[i: i + s])

def divide_sent_single(org, chunk=1):
    if chunk == 1:
        yield "[/SPC]".join(org)
    else:
        l = len(org)
        if chunk > l:
            raise Exception("Exceeded chunk size:{}, {}".format(l, chunk))
        s = l // chunk
        for i in range(0, l, s):
            yield "[/SPC]".join(org[i: i + s])


reg_s2 = re.compile("[ ]+")
def s2_remove(s):
    """
    스페이스가 연속 2개 이상 나올 경우 하나만 남기고 제거해준다.
    Parameters
    ----------
        s:string
            입력 문자열
    Returns
        string
            처리된 문자열
    """
    return reg_s2.sub(" ", s)

def create_logger(level = logging.INFO):
    """
    로거를 생성한다.
    Parameters
    ----------
        level: int
            로그 기록 수준
    Returns
    -------
        object
            로거
    """
    lgr = logging.getLogger("s2stagger")
    if len(lgr.handlers) > 0:
        return lgr
    c_handler = logging.StreamHandler()
    c_handler.setLevel(level)
    c_format = logging.Formatter("%(asctime)s;%(levelname)s] %(message)s",
                                 "%Y-%m-%d %H:%M:%S")
    c_handler.setFormatter(c_format)
    lgr.addHandler(c_handler)
    lgr.setLevel(level)
    lgr.propagate = False
    return lgr
