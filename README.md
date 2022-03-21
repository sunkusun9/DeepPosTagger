# DeepPosTagger

Sequence-To-Sequence 모델을 활용한 한국어 형태소 분석기

## 1. 특징

* [세종 말뭉치](https://ithub.korean.go.kr/user/main.do)와 국립국어원의 [모두의 말뭉치](https://corpus.korean.go.kr/) 말뭉치를 사용하여 학습\
  위 말뭉치들를 가공하여 만들어진 약 1,550,000 문장 중에서 약 2%인 31,000을 검증셋으로 사용, 검증셋에 대한 문장 단위 정확도(문장의 태깅 결과가 정확히 일치할 확률)는 69.65% 입니다.
* LSTM기반에 Sequence-To-Sequence 모델에, 인코더에 LSTM과 Self-Attention Layer를 결합하여 사용하여 성능을 향상.(이 모델은 저자의 아이디어이므로 이를 사용하고자 할 때는 반드시 Reference를 표기 바랍니다.)
* [세종 품사 태그](http://kkma.snu.ac.kr/documents/?doc=postag) 사용
* 단어 사전이 없습니다 - 모델 특성상 Corpus에 있던 언어 패턴으로 단어의 품사를 도출하여 사전이 필요 없어 간편합니다.

## 2. 설치
```
pip install deeppostagger
```

## 3. 사용법

```python
import os
# GPU를 사용하지 않을 경우
# 본 모듈이 동시에 처리하는 문장의 수(decode_net_size)가 크지 않을 경우에는 GPU를 사용하는 것은 오히려 역효과가 있다.
# 또한, 경험상 Windows에서 사용할 때 GPU 상에서 RNN을 구동시킬때 불안정하여 사용을 하지 않는게 좋다.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from deeppostagger import tagger

# decode_net_size: 디코더의 사이즈(동시 처리할 문장의 수, Default: 32, 단 건으로 문장을 처리하는 경우가 많으)
# lazy_start는 parser의 초기화 시점 지정(True: 처음 parse를 할 때 초기화, False: 태거를 생성할 때 초기화)
# 출력은 iterable 형태로 나오며, 문단 단위(개행문자 단위로 구분)로 반환이 됩니다.
_tagger = tagger.create_tagger(lazy_start=False, decode_net_size=8)
for res in _tagger.parse("Sequence-To-Sequence 모델을 활용한 한국어 형태소 분석기입니다.\n빠른 형태소 분석 보다는 정확한 분석에 유리하지만, 다수의 문장을 동시에 처리할 수 있게 하는 구조를 가지고 있어 사용하기에 따라서 성능을 최적화할 수 있다."):
    print(res)
```
실행 결과
```
[[('Sequence', 'SL'), ('-', 'SS'), ('To', 'SL'), ('-', 'SS'), ('Sequence', 'SL')], [('모델', 'NNG'), ('을', 'JKO')], [('활용', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETM')], [('한국어', 'NNP')], [('형태소', 'NNG')], [('분석기', 'NNG'), ('이', 'VCP'), ('ㅂ니다', 'EF'), ('.', 'SF')]]
[[('빠르', 'VA'), ('ㄴ', 'ETM')], [('형태소', 'NNG')], [('분석', 'NNG')], [('보다', 'JKB'), ('는', 'JX')], [('정확', 'NNG'), ('하', 'XSA'), ('ㄴ', 'ETM')], [('분석', 'NNG'), ('에', 'JKB')], [('유리', 'NNG'), ('하', 'XSA'), ('지만', 'EC'), (',', 'SP')], [('다수', 'NNG'), ('의', 'JKG')], [('문장', 'NNG'), ('을', 'JKO')], [('동시', 'NNG'), ('에', 'JKB')], [('처리', 'NNG'), ('하', 'XSV'), ('ㄹ', 'ETM')], [('수', 'NNB')], [('있', 'VV'), ('게', 'EC')], [('하', 'VX'), ('는', 'ETM')], [('구조', 'NNG'), ('를', 'JKO')], [('가지', 'VV'), ('고', 'EC')], [('있', 'VX'), ('어', 'EC')], [('사용', 'NNG'), ('하', 'XSV'), ('기', 'ETN'), ('에', 'JKB')], [('따르', 'VV'), ('아서', 'EC')], [('성능', 'NNG'), ('을', 'JKO')], [('최적', 'NNG'), ('화', 'XSN'), ('하', 'XSV'), ('ㄹ', 'ETM')], [('수', 'NNB')], [('있', 'VV'), ('다', 'EF'), ('.', 'SF')]]
```
* 성능향상 팁: decode_net_size에서 지정한 단위로 처리하는데, 이는 입력한 Paragraph 단위로 동시에 처리합니다. 따라서, 많은 단일 문단 단위의 작업으로 변환이 가능할 경우에는 단일 문단 마다 넣는 것보다 동시에 여러 decode_net_size 단위로 나누어 개행문자로 묶어서 넣는 것이 훨씬 빠릅니다.
  
Ex)
```python
import timeit
txts = [
    "decode_net_size: 디코더의 사이즈(동시 처리할 문장의 수, Default: 32)",
    "lazy_start는 parser의 초기화 시점 지정(True: 처음 parse를 할 때 초기화, False: 태거를 생성할 때 초기화)",
    "출력은 iterable 형태로 나오며, paragraph 단위로 반환이 됩니다"
]
st  = timeit.default_timer()
for txt in txts:
    for res in _tagger.parse(txt):
        print(res)
print("{}sec".format(timeit.default_timer() - st))

st  = timeit.default_timer()
for res in _tagger.parse("\n".join(txts)):
    print(res)
print("{}sec".format(timeit.default_timer() - st))
```
실행 결과
```
[[('decode', 'SL'), ('_', 'SW'), ('net', 'SL'), ('_', 'SS'), ('size', 'SL'), (':', 'SP')], [('디코더', 'NNG'), ('의', 'JKG')], [('사이즈', 'NNG'), ('(', 'SS'), ('동시', 'NNG')], [('처리', 'NNG'), ('하', 'XSV'), ('ㄹ', 'ETM')], [('문장', 'NNG'), ('의', 'JKG')], [('수', 'NNG'), (',', 'SP')], [('Default', 'SL'), (':', 'SP')], [('32', 'SN'), (')', 'SS')]]
[[('lazy', 'SL'), ('_', 'SW'), ('start', 'SL'), ('는', 'JX')], [('parser', 'SL'), ('의', 'JKG')], [('초기', 'NNG'), ('화', 'XSN')], [('시점', 'NNG')], [('지정', 'NNG'), ('(', 'SS'), ('True', 'SL'), (':', 'SP')], [('처음', 'NNG')], [('parse', 'SL'), ('를', 'JKO')], [('하', 'VV'), ('ㄹ', 'ETM')], [('때', 'NNG')], [('초기', 'NNG'), ('화', 'XSN'), (',', 'SP')], [('False', 'SL'), (':', 'SP')], [('태거', 'NNG'), ('를', 'JKO')], [('생성', 'NNG'), ('하', 'XSV'), ('ㄹ', 'ETM')], [('때', 'NNG')], [('초기', 'NNG'), ('화', 'XSN'), (')', 'SS')]]
[[('출력', 'NNG'), ('은', 'JX')], [('iterable', 'SL')], [('형태', 'NNG'), ('로', 'JKB')], [('나오', 'VV'), ('며', 'EC'), (',', 'SP')], [('paragraph', 'SL')], [('단위', 'NNG'), ('로', 'JKB')], [('반환', 'NNG'), ('이', 'JKC')], [('되', 'VV'), ('ㅂ니다', 'EC')]]
0.6064547000005405sec
[[('decode', 'SL'), ('_', 'SW'), ('net', 'SL'), ('_', 'SS'), ('size', 'SL'), (':', 'SP')], [('디코더', 'NNG'), ('의', 'JKG')], [('사이즈', 'NNG'), ('(', 'SS'), ('동시', 'NNG')], [('처리', 'NNG'), ('하', 'XSV'), ('ㄹ', 'ETM')], [('문장', 'NNG'), ('의', 'JKG')], [('수', 'NNG'), (',', 'SP')], [('Default', 'SL'), (':', 'SP')], [('32', 'SN'), (')', 'SS')]]
[[('lazy', 'SL'), ('_', 'SW'), ('start', 'SL'), ('는', 'JX')], [('parser', 'SL'), ('의', 'JKG')], [('초기', 'NNG'), ('화', 'XSN')], [('시점', 'NNG')], [('지정', 'NNG'), ('(', 'SS'), ('True', 'SL'), (':', 'SP')], [('처음', 'NNG')], [('parse', 'SL'), ('를', 'JKO')], [('하', 'VV'), ('ㄹ', 'ETM')], [('때', 'NNG')], [('초기', 'NNG'), ('화', 'XSN'), (',', 'SP')], [('False', 'SL'), (':', 'SP')], [('태거', 'NNG'), ('를', 'JKO')], [('생성', 'NNG'), ('하', 'XSV'), ('ㄹ', 'ETM')], [('때', 'NNG')], [('초기', 'NNG'), ('화', 'XSN'), (')', 'SS')]]
[[('출력', 'NNG'), ('은', 'JX')], [('iterable', 'SL')], [('형태', 'NNG'), ('로', 'JKB')], [('나오', 'VV'), ('며', 'EC'), (',', 'SP')], [('paragraph', 'SL')], [('단위', 'NNG'), ('로', 'JKB')], [('반환', 'NNG'), ('이', 'JKC')], [('되', 'VV'), ('ㅂ니다', 'EC')]]
0.28463689999989583sec
```

## 3. 맺음말
프로그램은 저자의 첫 번쨰 공개 프로젝트입니다. 공개 프로젝트에 대한 경험이 없어 운영이 미숙합니다. 여러 의견 부탁드립니다.
본 프로그램은 상업적인 사용은 허용하지 않습니다. 만일 필요할 경우 저자에게 연락바랍니다.

※ 학습 데이터 및 모듈은 제공하지 않습니다.
