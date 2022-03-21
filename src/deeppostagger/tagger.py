from deeppostagger import util
import tensorflow as tf
import pickle, re, json
from unicodedata import normalize
import pkg_resources, os
from tokenizers import Tokenizer

class Encoder(tf.keras.Model):
    """
    입력 시퀀스 인코더
    """
    def __init__(self, char_size, embedding_dim, depth=1):
        """
        Parameters
        ----------
            char_size: int
                글자수
            embedding_dim: int
                글자 임베딩 크기
            enc_units: int
                인코딩 유닛수
            depth: int
                인코더 깊이
        """
        super(Encoder, self).__init__()
        assert depth > 0
        self.embedding = tf.keras.layers.Embedding(char_size, embedding_dim)
        self.embedding.build((0, 1,))
        self.rnn_fs = []
        self.rnn_bs = []
        self.bds = []
        self.batch_size = 0
        self.initial_states = None
        self.do = tf.keras.layers.Dropout(0.1)
        self.sa = []
        self.fw_bw = embedding_dim
        for i in range(depth):
            wq = tf.keras.layers.Dense(self.fw_bw)
            wq.build((0, 0, self.fw_bw))
            wk = tf.keras.layers.Dense(self.fw_bw)
            wk.build((0, 0, self.fw_bw))
            wv = tf.keras.layers.Dense(self.fw_bw)
            wv.build((0, 0, self.fw_bw))
            layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            layernorm.build((0, 0, self.fw_bw))
            pff_d1 = tf.keras.layers.Dense(self.fw_bw * 4, activation='relu')
            pff_d1.build((0, 0, self.fw_bw))
            pff_d2 = tf.keras.layers.Dense(self.fw_bw)
            pff_d2.build((0, 0, self.fw_bw * 4))
            layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            layernorm2.build((0, 0, self.fw_bw))

            do1 = tf.keras.layers.Dropout(0.1)
            do2 = tf.keras.layers.Dropout(0.1)
            self.sa.append((wq, wk, wv, layernorm, pff_d1, pff_d2, layernorm2, do1, do2))
        
        for i in range(depth):
            rnn  = tf.keras.layers.LSTM(embedding_dim // 2, 
                                return_sequences = True, 
                                return_state = True,
                                recurrent_initializer='glorot_uniform')
            rnn_bw  = tf.keras.layers.LSTM(embedding_dim // 2, 
                                        return_sequences = True, 
                                        return_state = True,
                                        go_backwards=True, 
                                        recurrent_initializer='glorot_uniform')
            bd = tf.keras.layers.Bidirectional(rnn, backward_layer=rnn_bw)
            self.rnn_fs.append(rnn)
            self.rnn_bs.append(rnn_bw)
            if i == 0:
                bd.build((0, 0, embedding_dim))
            else:
                bd.build((0, 0, self.fw_bw))
            self.bds.append(bd)
        
    def call(self, x):
        x_mask = 1.0 - tf.cast(tf.math.equal(x, 0), dtype=tf.float32)
        x_mask = tf.expand_dims(x_mask, axis=-1)
        x = self.embedding(x)
        if self.trainable:
            x = self.do(x)
        states = []
        for (wq, wk, wv, layernorm, pff_d1, pff_d2, layernorm2, do1, do2), brnn in zip(self.sa, self.bds):
            x = x * x_mask
            x, forward_h, forward_c, backward_h, backward_c = brnn(x)
            states.append([tf.concat([forward_h, backward_h], axis=-1), tf.concat([forward_c, backward_c], axis=-1)])
            q = wq(x)
            k = wk(x)
            v = wv(x)
            aw = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(float(self.fw_bw))
            aw = tf.nn.softmax(aw, axis=-1)
            aw = tf.matmul(aw, v)
            if self.trainable:
                aw = do1(aw)
            x = layernorm(x + aw)
            ff = pff_d1(x)
            ff = pff_d2(ff)
            if self.trainable:
                ff = do2(ff)
            x = layernorm2(x + ff)
        return x, states
    
    def initialize_hidden_state(self, batch_size):
        if batch_size != self.batch_size:
            self.initial_states = [tf.zeros((batch_size, self.fw_bw // 2)), tf.zeros((batch_size, self.fw_bw // 2)), 
                                tf.zeros((batch_size, self.fw_bw // 2)), tf.zeros((batch_size, self.fw_bw // 2))]
            self.batch_size = batch_size
        return self.initial_states

class BahdanauAttention(tf.keras.layers.Layer):
    """
    Attention 레이어
    """
    def __init__(self, units, hidden_size, value_size):
        super(BahdanauAttention, self).__init__()
        self.units = units
        self.W1 = tf.keras.layers.Dense(units)
        self.W1.build((0, hidden_size))
        self.W2 = tf.keras.layers.Dense(units)
        self.W2.build((0, value_size))
        self.V = tf.keras.layers.Dense(1)
        self.V.build((0, units))

    def call(self, query, values):
        # 쿼리의 Shape 변환 query shape = (batch_size, max_length, hidden_size) => (batch_size, max_length, max_length, hidden size)
        query_with_time_axis = tf.expand_dims(query, 2)
        values_ = tf.expand_dims(self.W2(values), 1)
        values_ = tf.repeat(values_, tf.shape(query_with_time_axis)[1], axis=1)
        # Score 계산 score shape (batch_size, max_length, max_length, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + values_))

        # Attention Weight 계산 shape (batch_size, max_length, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=2)

        # Context Vector 계산 (batch_size, max_length, hidden_size)
        context_vector = attention_weights * tf.repeat(tf.expand_dims(values, 1), tf.shape(query_with_time_axis)[1], axis=1)
        context_vector = tf.reduce_sum(context_vector, axis=2)

        return context_vector, attention_weights

class BahdanauAttentionSingle(tf.keras.layers.Layer):
    """
    Attention 레이어(단일 입력)
    """
    def __init__(self, units, hidden_size, value_size):
        super(BahdanauAttentionSingle, self).__init__()
        self.units = units
        self.W1 = tf.keras.layers.Dense(units)
        self.W1.build((0, hidden_size))
        self.W2 = tf.keras.layers.Dense(units)
        self.W2.build((0, value_size))
        self.V = tf.keras.layers.Dense(1)
        self.V.build((0, units))

    def call(self, query, values):
        # Score 계산 score shape (batch_size, max_length, 1)
        score = self.V(tf.nn.tanh(self.W1(query) + self.W2(values)))
        # Attention Weight 계산 shape (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        # Context Vector 계산 (batch_size, max_length, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    """
    디코더 레이어
    """
    def __init__(self, char_size, embedding_dim_char, depth=1):
        """
        Parameters
        ----------
            char_size: int
                문자셋 사이즈
            embedding_dim_char: int
                문자 임베딩 사이즈
            dec_units: int
                디코더 레이어의 유닛 사이즈
            depth: int
                디코더 깊이
        """
        super(Decoder, self).__init__()
        self.embedding_char = tf.keras.layers.Embedding(char_size, embedding_dim_char)
        self.embedding_char.build((0, 1))
        self.depth = depth
        self.rnns = []
        self.rnn_size = embedding_dim_char
        self.do = tf.keras.layers.Dropout(0.1)
        for i in range(depth):
            rnn = tf.keras.layers.LSTM(self.rnn_size,
                                    return_sequences=True,
                                    recurrent_initializer='glorot_uniform')
            if i == 0:
                rnn.build((0, 0, embedding_dim_char))
            else:
                rnn.build((0, 0, self.rnn_size))
            self.rnns.append(rnn)       
        self.fc_char = tf.keras.layers.Dense(char_size)
        self.fc_char.build((0, embedding_dim_char * 2))
        # used for attention
        self.attention = BahdanauAttention(self.rnn_size, self.rnn_size, self.rnn_size)

    def call(self, x_char, states, enc_output):
        # 임베딩 통과후 입력 사이즈 == (batch_size, 1, embedding_dim)
        x = self.embedding_char(x_char)
        if self.trainable:
            x = self.do(x)
        for i, s in zip(self.rnns, states):
            x = i(x, initial_state=s)
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(x, enc_output)

        output =tf.concat([context_vector, x], axis=-1)
        # 출력 사이즈 == (batch_size, char_size)
        x_char = self.fc_char(output)

        return x_char, attention_weights

class DecoderSingle(tf.keras.Model):
    """
    디코더 레이어(단일 입력)
    """
    def __init__(self, char_size, embedding_dim_char, depth=1, input_size=1):
        """
        Parameters
        ----------
            char_size: int
                문자셋 사이즈
            embedding_dim_char: int
                문자 임베딩 사이즈
            dec_units: int
                디코더 레이어의 유닛 사이즈
            depth: int
                디코더 깊이
        """
        super(DecoderSingle, self).__init__()
        self.rnn_size = embedding_dim_char
        self.embedding_char = tf.keras.layers.Embedding(char_size, embedding_dim_char)
        self.embedding_char.build((0, 1))
        self.depth = depth
        self.rnns = []
        for i in range(depth):
            rnn = tf.keras.layers.LSTM(self.rnn_size,
                                    return_sequences=True,
                                    stateful=True,
                                    recurrent_initializer='glorot_uniform')
            if i == 0:
                rnn.build((input_size, 0, embedding_dim_char))
            else:
                rnn.build((input_size, 0, self.rnn_size))
            self.rnns.append(rnn)
        
        self.fc_char = tf.keras.layers.Dense(char_size)
        self.fc_char.build((0, embedding_dim_char * 2))
        # used for attention
        self.attention = BahdanauAttentionSingle(self.rnn_size, self.rnn_size, self.rnn_size)
        
    def reset_rnn_states(self, states):
        """
        RNN 상태를 초기화 한다
        Parameters
        ----------
            states: list
                상태값
        """
        for i, s in zip(self.rnns, states):
            i.reset_states(s)
        
    def call(self, x_char, enc_output):
        # 임베딩 통과후 입력 사이즈 == (batch_size, 1, embedding_dim)
        x = self.embedding_char(x_char)
        for i in self.rnns:
            x = i(x)
        # enc_output shape == (batch_size, hidden_size)
        context_vector, attention_weights = self.attention(x, enc_output)
        x = tf.reshape(x, (-1, x.shape[2]))
        output =tf.concat([context_vector, x], axis=-1)
        # 출력 사이즈 == (batch_size, char_size)
        x_char = self.fc_char(output)
        return x_char, attention_weights

class Seq2SeqPosTagger():
    """
    Sequence-2-Sequence 모델 기반에 형태소 분석기
    """
    def __init__(self, decode_net_size=32, concurrent_sent_size = 8):
        self.tok = None
        self.embedding_dim_char = 0
        self.units = 0
        self.encoder = None
        self.decoder = None
        self.training = True
        assert decode_net_size > 1
        self.decode_net_size = decode_net_size
        self.concurrent_sent_size = concurrent_sent_size
        # 음소만 있는 문자를 검출하는 정규식
        self.r_single = re.compile("([\u3130-\u318F])")
        self.r_tag = re.compile("|".join(["\\[/{}\\]".format(i) for i in util.kor_tagset]))
       
    def create(self, tok, embedding_dim_char, depth, max_len, training=True):
        """
        인코더, 디코더를 생성한다
        Parameters
        ----------
            tok: Tokenizer
                문장 토크나이져
            embedding_dim_char: int
                문자열 임베딩 사이즈
            depth: int
                RNN Layer의 깊이
            max_len: int
                최대 문장 길이
            training: bool
                학습 여부
        """
        assert depth > 0
        self.tok = tok
        self.depth = depth
        self.max_len = max_len
        if self.max_len == 0: 
            self.max_len = 64
        # 인코더 생성
        self.encoder = Encoder(self.tok.get_vocab_size(), embedding_dim_char, depth)
        self.training = training
        # 디코더 생성, 학습시에는 다음입력을 정답셋에서 사용하므로, 여러 입력을 처리할 수 있는 Decoder 사용, 
        # 실제 사용시에는 다음 입력이 이전 입력을 사용하므로 낱개 입력을 처리하는 Decoder 사용
        if training:
            self.decoder = Decoder(self.tok.get_vocab_size(), embedding_dim_char, depth)
        else:
            self.decoder = DecoderSingle(self.tok.get_vocab_size(), embedding_dim_char, depth, self.concurrent_sent_size)
            self.encoder.trainable = False
            self.decoder.trainable = False
    
    def save(self, filename):
        """
        모델 저장하기
        Parameters
        ----------
            filename: str
                파일명
        """
        with open(filename, 'wb') as f:
            pickle.dump(
                {
                    "embedding_dim_char": self.embedding_dim_char,
                    "depth": self.depth,
                    "encoder": self.encoder.get_weights() if self.encoder is not None else None,
                    "decoder": self.decoder.get_weights() if self.decoder is not None else None,
                    "max_len": self.max_len
                }, f)
    
    def load(self, tok, filename, embedding_dim_char, depth, training=True):
        """
        모델 불러오기.
        파라미터는 이전에 모델의 형태와 현재 모델 형태가 같아야지만 불러와진다.
        Parameters
        ----------
            tok: Tokenizer
                문장 토크나이져
            filename: str
                파일명
            embedding_dim_char: int
                문자열 임베딩 사이즈
            training: bool
                학습 여부
        Returns:
        --------
            bool
                새로 모델을 생성한 여부            
        """
        with open(filename, 'rb') as  f:
            obj = pickle.load(f)
        if obj["encoder"] is not None and obj["decoder"] is not None:
            self.create(tok, embedding_dim_char, depth, obj.get('max_len', 64), training)
            # 이전 저장된 모델과 현재 모델의 형태가 같을 경우 파라메터를 설정한다
            #if (embedding_dim_char == obj["embedding_dim_char"] and 
            #    fwd_units == obj.get("fwd_units", 0) and bwd_units == obj.get("bwd_units", 0) and
            #    depth == obj.get("depth", 3)):
            self.encoder.set_weights(obj["encoder"])
            self.decoder.set_weights(obj["decoder"])
                #return False
        return True
    
    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)]
    )
    def encode(self, txt_code):
        enc_output, states = self.encoder(
               txt_code
            )
        return enc_output, states
    
    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), 
                        tf.TensorSpec(shape=(None, None), dtype=tf.int64)]
    )
    def decode(self, enc_output, dec_char_input):
        c_idxs = []
        for _ in range(self.decode_net_size):
            pred_chars, _ = self.decoder(dec_char_input, enc_output)
            c_idx = tf.argmax(pred_chars, axis=-1)
            c_idxs.append(c_idx)
            dec_char_input = tf.cast(tf.expand_dims(c_idx, 1), dtype=tf.int64)
        return tf.stack(c_idxs), dec_char_input
    
    def parse(self, inp):
        """
        형태소 분석한다 
        Parameters
        ----------
            txt: str
                분석 
            max_len: int
                분석 결과의 최대 길이
        Returns
        -------
            list
                형태소 분석 결과
        """
        assert self.training == False, "학습 모드인 상태에서는 가능하지 않습니다."
        # 태깅 결과에서 문제점이 발견된 부분은 입력 문자열에 공백이 연속 두 번 들어 갔을 때에 문제를 일으킨다. 
        # 아무래도 학습시에는 이러한 경우가 없었기 때문인듯하다. 이를 방지하기 위해 연속 두 번 공백이 들어갈 경우 제거한다.
        if inp == None:
            return []
        txt_no = []
        txts = []
        restore_arrs = []
        decs_chars = []
        decs_tags = []
        for i, txt in enumerate(re.split('[\r\n]+', inp)):
            decs_chars.append([2])
            decs_tags.append([2])
            txt = util.s2_remove(txt)
            txt = util.txt_norm(txt)
            if len(txt) == 0:
                continue
            # 특수문자를 복원하기 위해 뽑아 놓는다.
            restore_arr = []
            pos = 0
            while pos >= 0:
                m = util.r_ex_all.search(txt, pos)
                if m is not None:
                    restore_arr.append(normalize("NFC",m[0]).translate(util.tbl_rev))
                    pos = m.span()[-1]
                else:
                    pos = -1
            restore_arrs.append(restore_arr)
            txt = util.repl(txt)
            txts_ = txt.split('.')
            """
            문장을 최대 문장 길이 단위로 쪼갠다
            학습했을 때 최대 문장 수보다 크면 예상치 않은 결과가 나온다.
            따라서 학습 당시의 최대 문장 길이로 쪼개서 처리한다.
            """
            while len(txts_) > 0:
                t = txts_.pop(0)
                t = t.strip()
                if len(txts_) > 0 and len(txts_[0]) > 0:
                    t = t + '.'
                while len(txts_) > 0 and len(txts_[0]) == 0:
                    txts_.pop(0)
                    t = t + '.'
                org = t.split(' ')
                tmp = []
                for chunk in range(1, len(org) + 1):
                    ok_flag = True
                    for o in util.divide_sent_single(org, chunk):
                        enc = self.tok.encode(o)
                        if len(enc.ids) > self.max_len:
                            ok_flag = False
                            tmp = []
                            break
                        tmp.append(enc.ids)
                    if ok_flag:
                        for t in tmp:
                            txts.append(t)
                            txt_no.append(i)
                        break
        len_txt = len(txts)
        step = self.concurrent_sent_size

        for t in range(0, len_txt, step):
            txts_ = txts[t: t + step]
            txt_no_ = txt_no[t: t+ step]
            max_ = 64
            end_ = []
            for i_, t_ in enumerate(txts_):
                end_.append(i_)
                max_ = max(max_, len(t_))
            for t_ in txts_:
                if len(t_) >= max_:
                    continue
                t_.extend([0] * (max_ - len(t_)))
            for i_ in range(step - len(txts_)):
                txts_.append([0] * max_)
            
            # 인코딩
            enc_output, states = self.encode(tf.constant(txts_, dtype=tf.int32))
            self.decoder.reset_rnn_states(states)
            # 시작 문자열 입력
            dec_char_input = tf.constant([[2]] * step, dtype=tf.int64)
            cnt = 0
            #  디코딩
            buff_char = [[] for _ in range(step)]
            while len(end_) > 0 and cnt < self.max_len * 4:
                c_idxs, dec_char_input = self.decode(enc_output, dec_char_input)
                cnt += self.decode_net_size
                qu = []
                for i_ in range(len(end_)):
                    i_ = len(end_) - i_ - 1
                    j_ = end_[i_]
                    for c_idx in c_idxs[:, j_]:
                        # 종료 여부 체크
                        if c_idx == 3 :
                            qu.append(i_)
                            break
                        buff_char[j_].append(c_idx.numpy())
                for q_ in qu:
                    end_.pop(q_)
            for n_, b_ in zip(txt_no_, buff_char):
                if len(b_) > 0:
                    decs_chars[n_].extend(b_)
        
        for decs_char, restore_arr in zip(decs_chars, restore_arrs):
            if len(decs_char) > 0:
                cnt = 0
                s_ = normalize("NFC", self.tok.decode(decs_char[1:], False))
                #print(s_, restore_arr)
                tags = [i for i in self.r_tag.findall(s_)]
                res = []
                for e in s_.split("[/SPC]"):
                    eo = []
                    for w in self.r_tag.split(e)[:-1]:
                        w = w.translate(util.tbl_rev)
                        mat = util.r_unk_chars.search(w)
                        if mat is not None and cnt < len(restore_arr):
                            if (mat[0] == "[/UNK]" and util.r_unk.search(restore_arr[cnt]) is not None) or (mat[0] == "[/HAN]" and util.r_hanja.search(restore_arr[cnt]) is not None):
                                w = util.r_unk_chars.sub(restore_arr[cnt], w)
                            else:
                                tags.pop(0)
                                continue
                            cnt += 1
                        eo.append((w, tags.pop(0)[2:-1]))
                    res.append(eo)
                yield res
    def test(self, org):
        max_ = 64
        txts = []
        end_ = []
        step = self.concurrent_sent_size
        for i_, t_ in enumerate(org):
            end_.append(i_)
            max_ = max(max_, len(t_))
            txts.append(t_)
        for t_ in txts:
            if len(t_) >= max_:
                continue
            t_.extend([0] * (max_ - len(t_)))
        for i_ in range(step - len(txts)):
            txts.append([0] * max_)  
        # 인코딩
        enc_output, states = self.encode(tf.constant(txts, dtype=tf.int32))
        self.decoder.reset_rnn_states(states)
        # 시작 문자열 입력
        dec_char_input = tf.constant([[2]] * len(txts), dtype=tf.int64)
        cnt = 0
        #  디코딩
        buff_char = [[] for _ in txts]
        while len(end_) > 0 and cnt < self.max_len * 4:
            c_idxs, dec_char_input = self.decode(enc_output, dec_char_input)
            cnt += 32
            qu = []
            for i_ in range(len(end_)):
                i_ = len(end_) - i_ - 1
                j_ = end_[i_]
                for c_idx in c_idxs[:, j_]:
                    # 종료 여부 체크
                    if c_idx == 3 :
                        qu.append(i_)
                        break
                    buff_char[j_].append(c_idx.numpy())
            for q_ in qu:
                end_.pop(q_)
        return buff_char

def create_tagger(name='sejong_nikl', decode_net_size = 32, lazy_start = True):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            #logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            #print("{} Physical GPUs, {} Logical GPUs".format(len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            print(e)
    tk_file = pkg_resources.resource_filename(__name__, os.path.join('res', name, 'tokenizer.json'))
    model_file = pkg_resources.resource_filename(__name__, os.path.join('res', name, "model.p"))
    config_file = pkg_resources.resource_filename(__name__, os.path.join('res', name, "config.json"))
    #print(tk_file, model_file, config_file)
    tok = Tokenizer.from_file(tk_file)
    _tagger = Seq2SeqPosTagger(decode_net_size)
    with open(config_file, 'r') as f:
        config = json.load(f)
    if not _tagger.load(tok, model_file, config['embedding_char_size'], config['depth'], training=False):
        raise Exception("Failed to load model")
    if not lazy_start:
        for _ in _tagger.parse('테스트입니다.'):
            pass
    return _tagger
