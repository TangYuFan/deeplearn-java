package tool.deeplearning;


import ai.djl.Device;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import java.nio.file.Path;
import java.io.File;
import java.text.MessageFormat;
import org.apache.commons.lang3.tuple.Pair;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import cn.hutool.core.util.ArrayUtil;
import cn.hutool.core.util.NumberUtil;
import com.github.houbb.pinyin.constant.enums.PinyinStyleEnum;
import com.github.houbb.pinyin.util.PinyinHelper;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.hankcs.hanlp.HanLP;
import com.rnkrsoft.bopomofo4j.Bopomofo4j;
import tool.util.AudioUtils;
import tool.util.FfmpegUtils;
import tool.util.SoundUtils;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 *  @Desc: 基于给定音色实现 text2wav , djl 推理
 *
 *
 *
 *  @Date: 2022-06-15 19:35:06
 *  @auth: TYF
 */
public class tts_sound_simulate_djl {

    /*

        #### 主要由三部分构成：
        #### 声音特征编码器（speaker encoder）
        提取说话者的声音特征信息。将说话者的语音嵌入编码为固定维度的向量，该向量表示了说话者的声音潜在特征。
        编码器主要将参考语音信号嵌入编码到固定维度的向量空间，并以此为监督，使映射网络能生成具有相同特征的原始声音信号（梅尔频谱图）。
        编码器的关键作用在于相似性度量，对于同一说话者的不同语音，其在嵌入向量空间中的向量距离（余弦夹角）应该尽可能小，而对不同说话者应该尽可能大。
        此外，编码器还应具有抗噪能力和鲁棒性，能够不受具体语音内容和背景噪声的影响，提取出说话者声音的潜在特征信息。
        这些要求和语音识别模型（speaker-discriminative）的要求不谋而合，因此可以进行迁移学习。
        编码器主要由三层LSTM构成，输入是40通道数的对数梅尔频谱图，最后一层最后一帧cell对应的输出经过L2正则化处理后，即得到整个序列的嵌入向量表示。
        实际推理时，任意长度的输入语音信号都会被800ms的窗口分割为多段，每段得到一个输出，最后将所有输出平均叠加，得到最终的嵌入向量。
        这种方法和短时傅里叶变换（STFT）非常相似。
        生成的嵌入空间向量可视化如下图：
        ![embedding](https://aias-home.oss-cn-beijing.aliyuncs.com/AIAS/voice_sdks/embedding.jpeg)

        可以看到不同的说话者在嵌入空间中对应不同的聚类范围，可以轻易区分，并且不同性别的说话者分别位于两侧。
        （然而合成语音和真实语音也比较容易区分开，合成语音离聚类中心的距离更远。这说明合成语音的真实度还不够。）

        #### 序列到序列的映射合成网络（Tacotron 2）
        基于Tacotron 2的映射网络，通过文本和声音特征编码器得到的向量来生成对数梅尔频谱图。
        梅尔光谱图将谱图的频率标度Hz取对数，转换为梅尔标度，使得人耳对声音的敏感度与梅尔标度承线性正相关关系。
        该网络独立于编码器网络的训练，以音频信号和对应的文本作为输入，音频信号首先经过预训练的编码器提取特征，然后再作为attention层的输入。
        网络输出特征由窗口长度为50ms，步长为12.5ms序列构成，经过梅尔标度滤波器和对数动态范围压缩后，得到梅尔频谱图。
        为了降低噪声数据的影响，还对该部分的损失函数额外添加了L1正则化。

        输入梅尔频谱图与合成频谱图的对比示例如下：
        ![embedding](https://aias-home.oss-cn-beijing.aliyuncs.com/AIAS/voice_sdks/tacotron2.jpeg)
        右图红线表示文本和频谱的对应关系。可以看到，用于参考监督的语音信号不需要与目标语音信号在文本上一致，这也是SV2TTS论文工作的一大特色。

        关于梅尔频谱图，可以看这篇文章。
        https://zhuanlan.zhihu.com/p/408265232

        #### 语音合成网络 (WaveGlow)
        WaveGlow:一种依靠流的从梅尔频谱图合成高质量语音的网络。它结合了Glow和WaveNet，生成的快、好、高质量的韵律，而且还不需要自动回归。
        将梅尔频谱图（谱域）转化为时间序列声音波形图（时域），完成语音的合成。
        需要注意的是，这三部分网络都是独立训练的，声音编码器网络主要对序列映射网络起到条件监督作用，保证生成的语音具有说话者的独特声音特征。


     */

    public static class SpeakerEncoder {

        String modelPath;
        public SpeakerEncoder(String modelPath) {
            this.modelPath = modelPath;
        }

        public Criteria<NDArray, NDArray> criteria() {
            Criteria<NDArray, NDArray> criteria =
                    Criteria.builder()
                            .setTypes(NDArray.class, NDArray.class)
                            .optModelPath(new File(modelPath).toPath())
                            .optTranslator(new SpeakerEncoderTranslator())
                            .optEngine("PyTorch") // Use PyTorch engine
                            .optDevice(Device.cpu())
                            .optProgress(new ProgressBar())
                            .build();

            return criteria;
        }
    }

    public static class SpeakerEncoderTranslator implements Translator<NDArray, NDArray> {

        public SpeakerEncoderTranslator() {

        }
        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, NDArray input) {
            return new NDList(input);
        }

        /** {@inheritDoc} */
        @Override
        public NDArray processOutput(TranslatorContext ctx, NDList list) {
            NDArray array = list.singletonOrThrow();
            return array;
        }

        /** {@inheritDoc} */
        @Override
        public Batchifier getBatchifier() {
            return Batchifier.STACK;
        }

    }

    public static class Tacotron2Encoder {

        String modelPath;
        public Tacotron2Encoder(String modelPath) {
            this.modelPath = modelPath;
        }

        public Criteria<NDList, NDArray> criteria() {
            Criteria<NDList, NDArray> criteria =
                    Criteria.builder()
                            .setTypes(NDList.class, NDArray.class)
                            .optModelPath(new File(modelPath).toPath())
                            .optTranslator(new TacotronTranslator())
                            .optEngine("PyTorch") // Use PyTorch engine
                            .optDevice(Device.cpu())
                            .optProgress(new ProgressBar())
                            .build();

            return criteria;
        }
    }


    public static class TacotronTranslator implements Translator<NDList, NDArray> {

        public TacotronTranslator() {}

        @Override
        public NDList processInput(TranslatorContext ctx, NDList input) {
            return input;
        }

        @Override
        public NDArray processOutput(TranslatorContext ctx, NDList list) {
            NDArray mels = list.get(0);
            NDArray mels_postnet = list.get(1);
            NDArray gates = list.get(2);
            NDArray alignments = list.get(3);
            //    	mels.detach();
            //    	mels_postnet.detach();
            //    	gates.detach();
            //    	alignments.detach();

            alignments = alignments.transpose(1, 0);
            gates = gates.transpose(1, 0);
            NDArray out_gate = gates.get(0);
            NDArray end_idx = out_gate.gt(0.2);
            boolean[] blidx = end_idx.toBooleanArray();
            int idx = 0;
            int size = blidx.length;
            for (int i = 0; i < size; i++) {
                if (blidx[i]) {
                    idx = i;
                }
            }
            if (idx == 0) {
                // System.out.println(out_gate.toDebugString(1000000000, 1000, 1000, 1000));
                // 原来的数据是float32 argMax计算后编程了int64 转为int32对应java的int
                NDArray outg = out_gate.argMax().toType(DataType.INT32, false);
//                System.out.println(outg.toDebugString(1000000000, 1000, 1000, 1000));
                int[] idxx = outg.toIntArray();
                System.out.println(Arrays.toString(idxx));
                idx = idxx[0];
            }
            if (idx == 0) {
                idx = (int) out_gate.getShape().get(0);
            }

            mels_postnet = mels_postnet.get(":, :" + idx);

            return mels_postnet;
        }

        @Override
        public Batchifier getBatchifier() {
            return Batchifier.STACK;
        }
    }


    public static class WaveGlowEncoder {
        String modelPath;
        public WaveGlowEncoder(String modelPath) {
            this.modelPath = modelPath;
        }

        public Criteria<NDArray, NDArray> criteria() {
            Criteria<NDArray, NDArray> criteria =
                    Criteria.builder()
                            .setTypes(NDArray.class, NDArray.class)
                            .optModelPath(new File(modelPath).toPath())
                            .optTranslator(new WaveGlowTranslator())
                            .optEngine("PyTorch") // Use PyTorch engine
                            .optDevice(Device.cpu())
                            .optProgress(new ProgressBar())
                            .build();

            return criteria;
        }
    }

    public static class WaveGlowTranslator implements Translator<NDArray, NDArray> {

        public WaveGlowTranslator() {

        }
        @Override
        public NDList processInput(TranslatorContext ctx, NDArray input) {
            NDArray sigma = ctx.getNDManager().create(1.0);
            NDList list = new NDList();
            list.add(input);
            list.add(sigma);
            return list;
        }

        @Override
        public NDArray processOutput(TranslatorContext ctx, NDList list) {
            NDArray ret = list.singletonOrThrow();
            ret.detach();
            return ret;
        }

        @Override
        public Batchifier getBatchifier() {
            return Batchifier.STACK;
        }

    }


    public static class DenoiserEncoder {
        String modelPath;
        public DenoiserEncoder(String modelPath) {
            this.modelPath = modelPath;
        }

        public Criteria<NDArray, NDArray> criteria() {
            Criteria<NDArray, NDArray> criteria =
                    Criteria.builder()
                            .setTypes(NDArray.class, NDArray.class)
                            .optModelPath(new File(modelPath).toPath())
                            .optTranslator(new DenoiserTranslator())
                            .optEngine("PyTorch") // Use PyTorch engine
                            .optDevice(Device.cpu())
                            .optProgress(new ProgressBar())
                            .build();

            return criteria;
        }
    }

    public static class DenoiserTranslator implements Translator<NDArray, NDArray> {

        public DenoiserTranslator() {

        }
        @Override
        public NDList processInput(TranslatorContext ctx, NDArray input) {
            NDArray denoiser_strength = ctx.getNDManager().create(1.0f);
            NDList list = new NDList();

		/*NDList dim = new NDList();
		dim.add(wav);
		NDArray adddim = NDArrays.stack(dim);*/

            list.add(input);
            list.add(denoiser_strength);
            return list;
        }

        @Override
        public NDArray processOutput(TranslatorContext ctx, NDList list) {
            NDArray ret = list.singletonOrThrow();
            ret.detach();
            return ret;
        }

        @Override
        public Batchifier getBatchifier() {
            return Batchifier.STACK;
        }

    }


    public static class SequenceUtils {

        // 分隔英文字母
        static Pattern _en_re = Pattern.compile("([a-zA-Z]+)");
        static Map<String,Integer> ph2id_dict = Maps.newHashMap();
        static Map<Integer,String> id2ph_dict =  Maps.newHashMap();
        static{
            int size = SymbolUtils.symbol_chinese.length;
            for(int i=0;i<size;i++){
                ph2id_dict.put(SymbolUtils.symbol_chinese[i], i);
                id2ph_dict.put(i, SymbolUtils.symbol_chinese[i]);
            }
        }
        public static List<Integer> text2sequence(String text){
	   /* 	文本转为ID序列。
	    :param text:
	    :return:*/
            List<String> phs = text2phoneme(text);
            List<Integer> seq = phoneme2sequence(phs);
            return seq;
        }

        public static List<String> text2phoneme(String text){

	  /*  文本转为音素，用中文音素方案。
	    中文转为拼音，按照清华大学方案转为音素，分为辅音、元音、音调。
	    英文全部大写，转为字母读音。
	    英文非全部大写，转为英文读音。
	    标点映射为音素。
	    :param text: str,正则化后的文本。
	    :return: list,音素列表*/
            text = normalizeChinese(text);
            text = normalizeEnglish(text);
            //System.out.println(text);
            String pys = text2pinyin(text);
            List<String> phs = pinyin2phoneme(pys);
            phs = changeDiao(phs);
            return phs ;
        }

        public static List<Integer> phoneme2sequence(List<String> src){
            List<Integer> out = Lists.newArrayList();
            for(String w : src){
                if( ph2id_dict.containsKey(w)){
                    out.add(ph2id_dict.get(w));
                }
            }
            return out;
        }


        public static List<String> changeDiao(List<String> src){
	   /*
	    	拼音变声调，连续上声声调的把前一个上声变为阳平。
	    :param src: list,音素列表
	    :return: list,变调后的音素列表*/

            int flag = -5;
            List<String> out = Lists.newArrayList();
            Collections.reverse(src);
            int size = src.size();
            for(int i=0;i<size;i++){
                String w = src.get(i);
                if(w.equals("3")){
                    if(i - flag ==4){
                        out.add("2");
                    }else{
                        flag = i;
                        out.add(w);
                    }
                }else{
                    out.add(w);
                }
            }
            Collections.reverse(out);
            return out;
        }




        public static String text2pinyin(String text){
            Bopomofo4j.local();//启用本地模式（也就是禁用沙盒）
            return  PinyinHelper.toPinyin(text, PinyinStyleEnum.NUM_LAST, " ");
            //return Bopomofo4j.pinyin(text,1, false, false," ").replaceAll("0", "5");
        }



        static String normalizeChinese(String text){
            text = ConvertUtils.quan2ban(text);
            text = ConvertUtils.fan2jian(text);
            text = NumberUtils.convertNumber(text);
            return text;
        }

        static String normalizeEnglish(String text){
            Matcher matcher =_en_re.matcher(text);
            LinkedList<Integer> postion = new LinkedList();
            while(matcher.find()){
                postion.add(matcher.start());
                postion.add(matcher.end());
            }
            if(postion.size() == 0){
                return text;
            }
            List<String> parts = Lists.newArrayList();
            parts.add(text.substring(0, postion.getFirst()));
            int size = postion.size()-1;
            for(int i=0;i<size;i++){
                parts.add(text.substring(postion.get(i),postion.get(i+1)));
            }
            parts.add(text.substring(postion.getLast()));
            LinkedList<String> out = new LinkedList();
            for(String part : parts){
                out.add(part.toLowerCase());
            }
            return Joiner.on("").join(out);
        }
        public static List<String>  pinyin2phoneme(String src){
            String[] srcs = src.split(" ");
            List<String> out = Lists.newArrayList();
            for(String py : srcs){
                List<String> phs = Lists.newArrayList();

                if(PhonemeUtils.pinyin2ph_dict.containsKey(py)){
                    String[] ph = PhonemeUtils.pinyin2ph_dict.get(py).split(" ");
                    List<String> list = new ArrayList<>(ph.length);
                    Collections.addAll(list, ph);
                    phs.addAll(list);
                }else{
                    String[] pys = py.split("");
                    for(String w : pys){
                        List<String> ph = py_errors(w);
                        phs.addAll(ph);
                    }
                }
                phs.add(SymbolUtils._chain);  // 一个字符对应一个chain符号
                out.addAll(phs);
            }
            out.add(SymbolUtils._eos);
            out.add(SymbolUtils._pad);
            return out;
        }

        static List<String> py_errors(String text){
            List<String> out = Lists.newArrayList();
            String[] texts = text.split("");
            for(String p : texts){
                if(PhonemeUtils.char2ph_dict.containsKey(p)){
                    out.add(PhonemeUtils.char2ph_dict.get(p));
                }
            }
            return out;
        }

//        public static void main(String[] args) {
//            System.out.println(normalizeEnglish("我hello,I love you 我是你"));
//            System.out.println(text2pinyin("这是实力很牛逼,"));
//            System.out.println(text2phoneme("这是实力很牛逼,"));
//            System.out.println(text2sequence("这是实力很牛逼,"));
//        }

    }


    /**
     * #### symbol
     * 音素标记。
     * Phonetic symbol tagging.
     * 中文音素，简单英文音素，简单中文音素。
     * Chinese phonetic symbols, simple English phonetic symbols, simple Chinese phonetic symbols.
     *
     * @author Administrator
     */
    public static class SymbolUtils {
        static String _pad = "_"; //  # 填充符 - Padding symbol
        static String _eos = "~"; //  # 结束符 - End-of-sequence symbol
        static String _chain = "-";// # 连接符，连接读音单位 - Connects pronunciation units
        static String _oov = "*";

        // 中文音素表 - Chinese phonetic symbols
        // 声母：27 - 27 initial sounds
        static String[] _shengmu = {
                "aa", "b", "c", "ch", "d", "ee", "f", "g", "h", "ii", "j", "k", "l", "m", "n", "oo", "p", "q", "r", "s", "sh",
                "t", "uu", "vv", "x", "z", "zh"
        };
        // 韵母：41 - 41 final sounds
        static String[] _yunmu = {
                "a", "ai", "an", "ang", "ao", "e", "ei", "en", "eng", "er", "i", "ia", "ian", "iang", "iao", "ie", "in", "ing",
                "iong", "iu", "ix", "iy", "iz", "o", "ong", "ou", "u", "ua", "uai", "uan", "uang", "ueng", "ui", "un", "uo", "v",
                "van", "ve", "vn", "ng", "uong"
        };
        // 声调：5 - 5 tones
        static String[] _shengdiao = {"1", "2", "3", "4", "5"};
        // 字母：26 - English phonetic symbols: 26 letters
        static String[] _alphabet = "Aa Bb Cc Dd Ee Ff Gg Hh Ii Jj Kk Ll Mm Nn Oo Pp Qq Rr Ss Tt Uu Vv Ww Xx Yy Zz".split(" ");
        // 英文：26
        static String[] _english = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".split(" ");
        // 标点：10 - Punctuation: 10 symbols
        static String[] _biaodian = "! ? . , ; : \" # ( )".split(" ");
        // 注：!=!！|?=?？|.=.。|,=,，、|;=;；|:=:：|"="“|#= \t|(=(（[［{｛【<《|)=)）]］}｝】>》
        // 其他：7 - Other symbols: 7 symbols
        static String[] _other = "w y 0 6 7 8 9".split(" ");
        // 大写字母：26 - Uppercase letters
        static String[] _upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
        // 小写字母：26 -  Lowercase letters
        static String[] _lower = "abcdefghijklmnopqrstuvwxyz".split("");
        // 标点符号：12 - Punctuation symbols
        static String[] _punctuation = "! ' \" ( ) , - . : ; ?".split(" ");

        // 数字：10 - Digits
        static String[] _digit = "0123456789".split("");
        // 字母和符号：64 - English letters and symbols
        // 用于英文:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'"(),-.:;?\s
        static String[] _character_en = ArrayUtil.addAll(_upper, _lower, _punctuation);

        // 字母、数字和符号：74 - Chinese and English letters and symbols: 74 symbols
        // 用于英文或中文:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'"(),-.:;?\s0123456789
        static String[] _character_cn = ArrayUtil.addAll(_upper, _lower, _punctuation, _digit);

        // 中文音素：145 - Chinese phonetic symbols: 145 symbols
        // 支持中文环境、英文环境、中英混合环境，中文把文字转为清华大学标准的音素表示
        static String[] symbol_chinese = ArrayUtil.addAll(new String[]{_pad, _eos, _chain}, _shengmu, _yunmu, _shengdiao, _alphabet, _english, _biaodian, _other);

        // 简单英文音素：66 - Simple English phonetic symbols: 66 symbols
        // 支持英文环境 - Supports English environments.
        // ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'"(),-.:;?\s
        static String[] symbol_english_simple = ArrayUtil.addAll(new String[]{_pad, _eos}, _upper, _lower, _punctuation);

        // 简单中文音素：76 - Simple Chinese phonetic symbols: 76 symbols
        // 支持英文、中文环境，中文把文字转为拼音字符串 - Supports Chinese and English environments. Converts Chinese characters into pinyin strings.
        // ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'"(),-.:;?\s0123456789
        static String[] symbol_chinese_simple = ArrayUtil.addAll(new String[]{_pad, _eos}, _upper, _lower, _punctuation, _digit);


	/*static{
		//英文
		System.arraycopy(_upper, 0, _character_en, 0, _upper.length);
		System.arraycopy(_lower, 0, _character_en, _upper.length, _lower.length);
	}*/

    }


    public static class ConvertUtils {
        /**
         * ASCII表中可见字符从!开始，偏移位值为33(Decimal)
         */
        private static final char DBC_CHAR_START = 33; // 半角!

        /**
         * ASCII表中可见字符到~结束，偏移位值为126(Decimal)
         */
        private static final char DBC_CHAR_END = 126; // 半角~

        /**
         * 全角对应于ASCII表的可见字符从！开始，偏移值为65281
         */
        private static final char SBC_CHAR_START = 65281; // 全角！

        /**
         * 全角对应于ASCII表的可见字符到～结束，偏移值为65374
         */
        private static final char SBC_CHAR_END = 65374; // 全角～

        /**
         * ASCII表中除空格外的可见字符与对应的全角字符的相对偏移
         */
        private static final int CONVERT_STEP = 65248; // 全角半角转换间隔

        /**
         * 全角空格的值，它没有遵从与ASCII的相对偏移，必须单独处理
         */
        private static final char SBC_SPACE = 12288; // 全角空格 12288

        /**
         * 半角空格的值，在ASCII中为32(Decimal)
         */
        private static final char DBC_SPACE = ' '; // 半角空格

        /**
         * <PRE>
         * 半角字符->全角字符转换
         * 只处理空格，!到?之间的字符，忽略其他
         * </PRE>
         */
        public static String ban2quan(String src) {
            if (src == null) {
                return src;
            }
            StringBuilder buf = new StringBuilder(src.length()+1);
            char[] ca = src.toCharArray();
            for (int i = 0; i < ca.length; i++) {
                if (ca[i] == DBC_SPACE) { // 如果是半角空格，直接用全角空格替代
                    buf.append(SBC_SPACE);
                } else if ((ca[i] >= DBC_CHAR_START) && (ca[i] <= DBC_CHAR_END)) { // 字符是!到~之间的可见字符
                    buf.append((char) (ca[i] + CONVERT_STEP));
                } else { // 不对空格以及ascii表中其他可见字符之外的字符做任何处理
                    buf.append(ca[i]);
                }
            }
            return buf.toString();
        }

        /**
         * <PRE>
         * 全角字符->半角字符转换
         * 只处理全角的空格，全角！到全角～之间的字符，忽略其他
         * </PRE>
         */
        public static String quan2ban(String src) {
            if (src == null) {
                return src;
            }
            StringBuilder buf = new StringBuilder(src.length()+1);
            char[] ca = src.toCharArray();
            for (int i = 0; i < src.length(); i++) {
                if (ca[i] >= SBC_CHAR_START && ca[i] <= SBC_CHAR_END) { // 如果位于全角！到全角～区间内
                    buf.append((char) (ca[i] - CONVERT_STEP));
                } else if (ca[i] == SBC_SPACE) { // 如果是全角空格
                    buf.append(DBC_SPACE);
                } else { // 不处理全角空格，全角！到全角～区间外的字符
                    buf.append(ca[i]);
                }
            }
            return buf.toString();
        }

        public static String jian2fan(String src) {
            return HanLP.convertToTraditionalChinese(src);
        }
        public static String fan2jian(String src) {
            return HanLP.convertToSimplifiedChinese(src);
        }

//        public static void main(String[] args) throws Exception {
//            System.out.println(ban2quan("aA1 ,:$。、"));
//            System.out.println(quan2ban("ａＡ１　，：＄。、"));
//            System.out.println(jian2fan("中国语言"));
//            System.out.println(fan2jian("中國語言"));
//        }


    }


    public static class PhonemeUtils {
        // 拼音转音素映射表：420
        // Phonetic-to-phoneme mapping table: 420
        static ImmutableMap<String,String> shengyun2ph_dict = ImmutableMap.<String, String>builder()
                .put("a", "aa a")
                .put("ai", "aa ai")
                .put("an", "aa an")
                .put("ang", "aa ang")
                .put("ao", "aa ao")
                .put("ba", "b a")
                .put("bai", "b ai")
                .put("ban", "b an")
                .put("bang", "b ang")
                .put("bao", "b ao")
                .put("bei", "b ei")
                .put("ben", "b en")
                .put("beng", "b eng")
                .put("bi", "b i")
                .put("bian", "b ian")
                .put("biao", "b iao")
                .put("bie", "b ie")
                .put("bin", "b in")
                .put("bing", "b ing")
                .put("bo", "b o")
                .put("bu", "b u")
                .put("ca", "c a")
                .put("cai", "c ai")
                .put("can", "c an")
                .put("cang", "c ang")
                .put("cao", "c ao")
                .put("ce", "c e")
                .put("cen", "c en")
                .put("ceng", "c eng")
                .put("ci", "c iy")
                .put("cong", "c ong")
                .put("cou", "c ou")
                .put("cu", "c u")
                .put("cuan", "c uan")
                .put("cui", "c ui")
                .put("cun", "c un")
                .put("cuo", "c uo")
                .put("cha", "ch a")
                .put("chai", "ch ai")
                .put("chan", "ch an")
                .put("chang", "ch ang")
                .put("chao", "ch ao")
                .put("che", "ch e")
                .put("chen", "ch en")
                .put("cheng", "ch eng")
                .put("chi", "ch ix")
                .put("chong", "ch ong")
                .put("chou", "ch ou")
                .put("chu", "ch u")
                .put("chuai", "ch uai")
                .put("chuan", "ch uan")
                .put("chuang", "ch uang")
                .put("chui", "ch ui")
                .put("chun", "ch un")
                .put("chuo", "ch uo")
                .put("da", "d a")
                .put("dai", "d ai")
                .put("dan", "d an")
                .put("dang", "d ang")
                .put("dao", "d ao")
                .put("de", "d e")
                .put("dei", "d ei")
                .put("deng", "d eng")
                .put("di", "d i")
                .put("dia", "d ia")
                .put("dian", "d ian")
                .put("diao", "d iao")
                .put("die", "d ie")
                .put("ding", "d ing")
                .put("diu", "d iu")
                .put("dong", "d ong")
                .put("dou", "d ou")
                .put("du", "d u")
                .put("duan", "d uan")
                .put("dui", "d ui")
                .put("dun", "d un")
                .put("duo", "d uo")
                .put("e", "ee e")
                .put("ei", "ee ei")
                .put("en", "ee en")
                .put("er", "ee er")
                .put("fa", "f a")
                .put("fan", "f an")
                .put("fang", "f ang")
                .put("fei", "f ei")
                .put("fen", "f en")
                .put("feng", "f eng")
                .put("fo", "f o")
                .put("fou", "f ou")
                .put("fu", "f u")
                .put("ga", "g a")
                .put("gai", "g ai")
                .put("gan", "g an")
                .put("gang", "g ang")
                .put("gao", "g ao")
                .put("ge", "g e")
                .put("gei", "g ei")
                .put("gen", "g en")
                .put("geng", "g eng")
                .put("gong", "g ong")
                .put("gou", "g ou")
                .put("gu", "g u")
                .put("gua", "g ua")
                .put("guai", "g uai")
                .put("guan", "g uan")
                .put("guang", "g uang")
                .put("gui", "g ui")
                .put("gun", "g un")
                .put("guo", "g uo")
                .put("ha", "h a")
                .put("hai", "h ai")
                .put("han", "h an")
                .put("hang", "h ang")
                .put("hao", "h ao")
                .put("he", "h e")
                .put("hei", "h ei")
                .put("hen", "h en")
                .put("heng", "h eng")
                .put("hong", "h ong")
                .put("hou", "h ou")
                .put("hu", "h u")
                .put("hua", "h ua")
                .put("huai", "h uai")
                .put("huan", "h uan")
                .put("huang", "h uang")
                .put("hui", "h ui")
                .put("hun", "h un")
                .put("huo", "h uo")
                .put("yi", "ii i")
                .put("ya", "ii ia")
                .put("yan", "ii ian")
                .put("yang", "ii iang")
                .put("yao", "ii iao")
                .put("ye", "ii ie")
                .put("yin", "ii in")
                .put("ying", "ii ing")
                .put("yong", "ii iong")
                .put("you", "ii iu")
                .put("ji", "j i")
                .put("jia", "j ia")
                .put("jian", "j ian")
                .put("jiang", "j iang")
                .put("jiao", "j iao")
                .put("jie", "j ie")
                .put("jin", "j in")
                .put("jing", "j ing")
                .put("jiong", "j iong")
                .put("jiu", "j iu")
                .put("ju", "j v")
                .put("juan", "j van")
                .put("jue", "j ve")
                .put("jun", "j vn")
                .put("ka", "k a")
                .put("kai", "k ai")
                .put("kan", "k an")
                .put("kang", "k ang")
                .put("kao", "k ao")
                .put("ke", "k e")
                .put("ken", "k en")
                .put("keng", "k eng")
                .put("kong", "k ong")
                .put("kou", "k ou")
                .put("ku", "k u")
                .put("kua", "k ua")
                .put("kuai", "k uai")
                .put("kuan", "k uan")
                .put("kuang", "k uang")
                .put("kui", "k ui")
                .put("kun", "k un")
                .put("kuo", "k uo")
                .put("la", "l a")
                .put("lai", "l ai")
                .put("lan", "l an")
                .put("lang", "l ang")
                .put("lao", "l ao")
                .put("le", "l e")
                .put("lei", "l ei")
                .put("leng", "l eng")
                .put("li", "l i")
                .put("lia", "l ia")
                .put("lian", "l ian")
                .put("liang", "l iang")
                .put("liao", "l iao")
                .put("lie", "l ie")
                .put("lin", "l in")
                .put("ling", "l ing")
                .put("liu", "l iu")
                .put("lo", "l o")
                .put("long", "l ong")
                .put("lou", "l ou")
                .put("lu", "l u")
                .put("luan", "l uan")
                .put("lun", "l un")
                .put("luo", "l uo")
                .put("lv", "l v")
                .put("lve", "l ve")
                .put("ma", "m a")
                .put("mai", "m ai")
                .put("man", "m an")
                .put("mang", "m ang")
                .put("mao", "m ao")
                .put("me", "m e")
                .put("mei", "m ei")
                .put("men", "m en")
                .put("meng", "m eng")
                .put("mi", "m i")
                .put("mian", "m ian")
                .put("miao", "m iao")
                .put("mie", "m ie")
                .put("min", "m in")
                .put("ming", "m ing")
                .put("miu", "m iu")
                .put("mo", "m o")
                .put("mou", "m ou")
                .put("mu", "m u")
                .put("na", "n a")
                .put("nai", "n ai")
                .put("nan", "n an")
                .put("nang", "n ang")
                .put("nao", "n ao")
                .put("ne", "n e")
                .put("nei", "n ei")
                .put("nen", "n en")
                .put("neng", "n eng")
                .put("ni", "n i")
                .put("nian", "n ian")
                .put("niang", "n iang")
                .put("niao", "n iao")
                .put("nie", "n ie")
                .put("nin", "n in")
                .put("ning", "n ing")
                .put("niu", "n iu")
                .put("nong", "n ong")
                .put("nu", "n u")
                .put("nuan", "n uan")
                .put("nuo", "n uo")
                .put("nv", "n v")
                .put("nve", "n ve")
                .put("o", "oo o")
                .put("ou", "oo ou")
                .put("pa", "p a")
                .put("pai", "p ai")
                .put("pan", "p an")
                .put("pang", "p ang")
                .put("pao", "p ao")
                .put("pei", "p ei")
                .put("pen", "p en")
                .put("peng", "p eng")
                .put("pi", "p i")
                .put("pian", "p ian")
                .put("piao", "p iao")
                .put("pie", "p ie")
                .put("pin", "p in")
                .put("ping", "p ing")
                .put("po", "p o")
                .put("pou", "p ou")
                .put("pu", "p u")
                .put("qi", "q i")
                .put("qia", "q ia")
                .put("qian", "q ian")
                .put("qiang", "q iang")
                .put("qiao", "q iao")
                .put("qie", "q ie")
                .put("qin", "q in")
                .put("qing", "q ing")
                .put("qiong", "q iong")
                .put("qiu", "q iu")
                .put("qu", "q v")
                .put("quan", "q van")
                .put("que", "q ve")
                .put("qun", "q vn")
                .put("ran", "r an")
                .put("rang", "r ang")
                .put("rao", "r ao")
                .put("re", "r e")
                .put("ren", "r en")
                .put("reng", "r eng")
                .put("ri", "r iz")
                .put("rong", "r ong")
                .put("rou", "r ou")
                .put("ru", "r u")
                .put("ruan", "r uan")
                .put("rui", "r ui")
                .put("run", "r un")
                .put("ruo", "r uo")
                .put("sa", "s a")
                .put("sai", "s ai")
                .put("san", "s an")
                .put("sang", "s ang")
                .put("sao", "s ao")
                .put("se", "s e")
                .put("sen", "s en")
                .put("seng", "s eng")
                .put("si", "s iy")
                .put("song", "s ong")
                .put("sou", "s ou")
                .put("su", "s u")
                .put("suan", "s uan")
                .put("sui", "s ui")
                .put("sun", "s un")
                .put("suo", "s uo")
                .put("sha", "sh a")
                .put("shai", "sh ai")
                .put("shan", "sh an")
                .put("shang", "sh ang")
                .put("shao", "sh ao")
                .put("she", "sh e")
                .put("shei", "sh ei")
                .put("shen", "sh en")
                .put("sheng", "sh eng")
                .put("shi", "sh ix")
                .put("shou", "sh ou")
                .put("shu", "sh u")
                .put("shua", "sh ua")
                .put("shuai", "sh uai")
                .put("shuan", "sh uan")
                .put("shuang", "sh uang")
                .put("shui", "sh ui")
                .put("shun", "sh un")
                .put("shuo", "sh uo")
                .put("ta", "t a")
                .put("tai", "t ai")
                .put("tan", "t an")
                .put("tang", "t ang")
                .put("tao", "t ao")
                .put("te", "t e")
                .put("teng", "t eng")
                .put("ti", "t i")
                .put("tian", "t ian")
                .put("tiao", "t iao")
                .put("tie", "t ie")
                .put("ting", "t ing")
                .put("tong", "t ong")
                .put("tou", "t ou")
                .put("tu", "t u")
                .put("tuan", "t uan")
                .put("tui", "t ui")
                .put("tun", "t un")
                .put("tuo", "t uo")
                .put("wu", "uu u")
                .put("wa", "uu ua")
                .put("wai", "uu uai")
                .put("wan", "uu uan")
                .put("wang", "uu uang")
                .put("weng", "uu ueng")
                .put("wei", "uu ui")
                .put("wen", "uu un")
                .put("wo", "uu uo")
                .put("yu", "vv v")
                .put("yuan", "vv van")
                .put("yue", "vv ve")
                .put("yun", "vv vn")
                .put("xi", "x i")
                .put("xia", "x ia")
                .put("xian", "x ian")
                .put("xiang", "x iang")
                .put("xiao", "x iao")
                .put("xie", "x ie")
                .put("xin", "x in")
                .put("xing", "x ing")
                .put("xiong", "x iong")
                .put("xiu", "x iu")
                .put("xu", "x v")
                .put("xuan", "x van")
                .put("xue", "x ve")
                .put("xun", "x vn")
                .put("za", "z a")
                .put("zai", "z ai")
                .put("zan", "z an")
                .put("zang", "z ang")
                .put("zao", "z ao")
                .put("ze", "z e")
                .put("zei", "z ei")
                .put("zen", "z en")
                .put("zeng", "z eng")
                .put("zi", "z iy")
                .put("zong", "z ong")
                .put("zou", "z ou")
                .put("zu", "z u")
                .put("zuan", "z uan")
                .put("zui", "z ui")
                .put("zun", "z un")
                .put("zuo", "z uo")
                .put("zha", "zh a")
                .put("zhai", "zh ai")
                .put("zhan", "zh an")
                .put("zhang", "zh ang")
                .put("zhao", "zh ao")
                .put("zhe", "zh e")
                .put("zhei", "zh ei")
                .put("zhen", "zh en")
                .put("zheng", "zh eng")
                .put("zhi", "zh ix")
                .put("zhong", "zh ong")
                .put("zhou", "zh ou")
                .put("zhu", "zh u")
                .put("zhua", "zh ua")
                .put("zhuai", "zh uai")
                .put("zhuan", "zh uan")
                .put("zhuang", "zh uang")
                .put("zhui", "zh ui")
                .put("zhun", "zh un")
                .put("zhuo", "zh uo")
                .put("cei", "c ei")
                .put("chua", "ch ua")
                .put("den", "d en")
                .put("din", "d in")
                .put("eng", "ee eng")
                .put("ng", "ee ng")
                .put("fiao", "f iao")
                .put("yo", "ii o")
                .put("kei", "k ei")
                .put("len", "l en")
                .put("nia", "n ia")
                .put("nou", "n ou")
                .put("nun", "n un")
                .put("rua", "r ua")
                .put("tei", "t ei")
                .put("wong", "uu uong")
                .put("n", "n ng")
                .build();
        static ImmutableMap<String,String>  diao2ph_dict = ImmutableMap.<String, String>builder()
                .put("1", "1")
                .put("2", "2")
                .put("3", "3")
                .put("4", "4")
                .put("5", "5")
                .build();

        static Map<String,String> pinyin2ph_dict = Maps.newHashMap();
        static{

		/*for(String ksy : shengyun2ph_dict.keySet()){
			String vsy = shengyun2ph_dict.get(ksy);
			for(String kd : diao2ph_dict.keySet()){
				String vd = diao2ph_dict.get(kd);
				pinyin2ph_dict.put(MessageFormat.format("{0}{1}", ksy,kd),MessageFormat.format("{0} {1}", vsy,vd));
			}
		} */
		/*for(String kye : pinyin2ph_dict.keySet()){
			if(kye.equals("bi1")){
				System.out.println(kye+"="+pinyin2ph_dict.get(kye));
			}
		}*/

            shengyun2ph_dict.forEach((ksy, vsy) -> {
                diao2ph_dict.forEach((kd, vd) -> {
                    pinyin2ph_dict.put(MessageFormat.format("{0}{1}", ksy,kd),MessageFormat.format("{0} {1}", vsy,vd));
                });
            });
        }

        // 字母音素：26
        static String[] _alphabet = "Aa Bb Cc Dd Ee Ff Gg Hh Ii Jj Kk Ll Mm Nn Oo Pp Qq Rr Ss Tt Uu Vv Ww Xx Yy Zz".split(" ");
        // 字母：26
        static String[] _upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
        static String[] _lower = "abcdefghijklmnopqrstuvwxyz".split("");
        static ImmutableMap<String,String> upper2ph_dict = ImmutableMap.<String, String>builder()
                .put("A", "Aa")
                .put("B", "Bb")
                .put("C", "Cc")
                .put("D", "Dd")
                .put("E", "Ee")
                .put("F", "Ff")
                .put("G", "Gg")
                .put("H", "Hh")
                .put("I", "Ii")
                .put("J", "Jj")
                .put("K", "Kk")
                .put("L", "Ll")
                .put("M", "Mm")
                .put("N", "Nn")
                .put("O", "Oo")
                .put("P", "Pp")
                .put("Q", "Qq")
                .put("R", "Rr")
                .put("S", "Ss")
                .put("T", "Tt")
                .put("U", "Uu")
                .put("V", "Vv")
                .put("W", "Ww")
                .put("X", "Xx")
                .put("Y", "Yy")
                .put("Z", "Zz")
                .build();
        static ImmutableMap<String,String> lower2ph_dict = ImmutableMap.<String, String>builder()
                .put("a", "Aa")
                .put("b", "Bb")
                .put("c", "Cc")
                .put("d", "Dd")
                .put("e", "Ee")
                .put("f", "Ff")
                .put("g", "Gg")
                .put("h", "Hh")
                .put("i", "Ii")
                .put("j", "Jj")
                .put("k", "Kk")
                .put("l", "Ll")
                .put("m", "Mm")
                .put("n", "Nn")
                .put("o", "Oo")
                .put("p", "Pp")
                .put("q", "Qq")
                .put("r", "Rr")
                .put("s", "Ss")
                .put("t", "Tt")
                .put("u", "Uu")
                .put("v", "Vv")
                .put("w", "Ww")
                .put("x", "Xx")
                .put("y", "Yy")
                .put("z", "Zz")
                .build();

        static String[] _biaodian = "! ? . , ; : \" # ( )".split(" ");

        static ImmutableMap<String,String> biao2ph_dict = ImmutableMap.<String, String>builder()
                .put("!","!")
                .put("！", "!")
                .put("?", "?")
                .put("？", "?")
                .put(".", ".")
                .put("。", ".")
                .put(",", ",")
                .put("，", ",")
                .put("、", ",")
                .put(";", ";")
                .put("；", ";")
                .put(":", ":")
                .put("：", ":")
                .put("\"", "\"")
                .put("“", "\"")
                .put("”", "\"")
                .put("'", "\"")
                .put("‘", "\"")
                .put("’", "\"")
                .put(" ", "#")
                .put("\u3000", "#")
                .put("\t", "#")
                .put("(", "(")
                .put("（", "(")
                .put("[", "(")
                .put("［", "(")
                .put("{", "(")
                .put("｛", "(")
                .put("【", "(")
                .put("<", "(")
                .put("《", "(")
                .put(")", ")")
                .put("）", ")")
                .put("]", ")")
                .put("］", ")")
                .put("}", ")")
                .put("｝", ")")
                .put("】", ")")
                .put(">", ")")
                .put("》", ")")
                .build();
        static ImmutableMap<String,String> char2ph_dict = ImmutableMap.copyOf(upper2ph_dict).copyOf(lower2ph_dict).copyOf(biao2ph_dict);

        public static void main(String[] args) {

            for(String a : _biaodian){
                System.out.println(a);
            }
            System.out.println(MessageFormat.format("我爱你我的祖国{0}，你是那么美丽","哈哈哈"));
        }
    }


    public static class NumberUtils {

        static String[] _number_cn = {"零", "一", "二", "三", "四", "五", "六", "七", "八", "九"};
        static String[] _number_level = {"千", "百", "十", "万", "千", "百", "十", "亿", "千", "百", "十", "万", "千", "百", "十", "个"};
        static String _zero = _number_cn[0];
        static Pattern _ten_re = Pattern.compile("^一十");
        static ImmutableList<String> _grade_level = ImmutableList.of("万", "亿", "个");
        static Pattern _number_group_re = Pattern.compile("([0-9]+)");

        public static void main(String[] args) {
            System.out.println(sayDigit("51234565"));
            System.out.println(sayNumber("12345678901234561"));
            System.out.println(sayDecimal("3.14"));
            System.out.println(convertNumber("hello314.1592and2718281828"));

            // 五一二三四五六五
            // 12345678901234561 (小于等于16位时: 十二亿三千四百五十六万七千八百九十)
            // 三点一四
            // hello三百一十四.一千五百九十二and二七一八二八一八二八
        }

        public static String sayDigit(String num) {
            StringBuilder outs = new StringBuilder();
            String[] ss = num.split("");
            for (String s : ss) {
                outs.append(_number_cn[Integer.valueOf(s)]);
            }
            return outs.toString();
        }

        public static String sayNumber(String nums) {
            String x = nums;
            if (x == "0") {
                return _number_cn[0];
            } else if (x.length() > 16) {
                return nums;
            }
            int length = x.length();
            LinkedList<String> outs = new LinkedList();
            String[] ss = x.split("");
            for (int i = 0; i < ss.length; i++) {
                String a = _number_cn[Integer.valueOf(ss[i])];
                String b = _number_level[_number_level.length - length + i];
                if (!a.equals(_zero)) {
                    outs.add(a);
                    outs.add(b);
                } else {
                    if (_grade_level.contains(b)) {
                        if (!_zero.equals(outs.getLast())) {
                            outs.add(b);
                        } else {
                            outs.removeLast();
                            outs.add(b);
                        }
                    } else {
                        if (!_zero.equals(outs.getLast())) {
                            outs.add(a);
                        }
                    }
                }
            }
            outs.removeLast();
            String out = Joiner.on("").join(outs);
            // 进行匹配
            Matcher matcher = _ten_re.matcher(out);
            out = matcher.replaceAll("十");
            return out;
        }

        public static String sayDecimal(String num) {
            String[] nums = num.split("\\.");
            String z_cn = sayNumber(nums[0]);
            String x_cn = sayDigit(nums[1]);
            return z_cn + '点' + x_cn;
        }

        public static String convertNumber(String text) {

            Matcher matcher = _number_group_re.matcher(text);
            LinkedList<Integer> postion = new LinkedList();
            while (matcher.find()) {
                postion.add(matcher.start());
                postion.add(matcher.end());
            }
            if (postion.size() == 0) {
                return text;
            }
            List<String> parts = Lists.newArrayList();
            parts.add(text.substring(0, postion.getFirst()));
            int size = postion.size() - 1;
            for (int i = 0; i < size; i++) {
                parts.add(text.substring(postion.get(i), postion.get(i + 1)));
            }
            parts.add(text.substring(postion.getLast()));
            LinkedList<String> outs = new LinkedList();
            for (String elem : parts) {
                if (NumberUtil.isNumber(elem)) {
                    if (elem.length() <= 9) {
                        outs.add(sayNumber(elem));
                    } else {
                        outs.add(sayDigit(elem));
                    }
                } else {
                    outs.add(elem);
                }
            }
            return Joiner.on("").join(outs);
        }


    }

    public static void main(String[] args) throws Exception{




        // 需要转换的文本
        String text = "家人们，谁懂啊，遇到一个下头男。偷拍我，啊啊啊啊啊";

        // 目标音色
        String sound = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\tts_sound_simulate_djl\\biaobei-009502.mp3";

        // 输出结果音频
        String out = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\tts_sound_simulate_djl\\audio.wav";

        // 模型1
        String model1 = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\tts_sound_simulate_djl\\speakerEncoder.pt";
        // 模型2
        String model2 = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\tts_sound_simulate_djl\\tacotron2.pt";
        // 模型3
        String model3 = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\tts_sound_simulate_djl\\waveGlow.pt";
        // 模型4
        String model4 = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\tts_sound_simulate_djl\\denoiser.pt";



        // --------------------------------------------------

        NDManager manager = NDManager.newBaseManager(Device.cpu(),"PyTorch");

        // 加载四个模型
        SpeakerEncoder speakerEncoder = new SpeakerEncoder(model1); // 音色提取
        Tacotron2Encoder tacotron2Encoder = new Tacotron2Encoder(model2); // 生成mel频谱
        WaveGlowEncoder waveGlowEncoder = new WaveGlowEncoder(model3); // mel频谱数据生成带噪wav
        DenoiserEncoder denoiserEncoder = new DenoiserEncoder(model4); // 带噪wav去噪得到最终wav

        // 加载四个模型推理器
        try (ZooModel<NDArray, NDArray> speakerEncoderModel =
                     ModelZoo.loadModel(speakerEncoder.criteria());
             Predictor<NDArray, NDArray> speakerEncoderPredictor = speakerEncoderModel.newPredictor();
             ZooModel<NDList, NDArray> tacotron2Model = ModelZoo.loadModel(tacotron2Encoder.criteria());
             Predictor<NDList, NDArray> tacotron2Predictor = tacotron2Model.newPredictor();
             ZooModel<NDArray, NDArray> waveGlowModel =
                     ModelZoo.loadModel(waveGlowEncoder.criteria());
             Predictor<NDArray, NDArray> waveGlowPredictor = waveGlowModel.newPredictor();
             ZooModel<NDArray, NDArray> denoiserModel =
                     ModelZoo.loadModel(denoiserEncoder.criteria());
             Predictor<NDArray, NDArray> denoiserPredictor = denoiserModel.newPredictor()
        ) {


            // 文本转为ID列表
            List<Integer> text_data_org = SequenceUtils.text2sequence(text);
            int[] text_dataa = text_data_org.stream().mapToInt(Integer::intValue).toArray();
            NDArray text_data = manager.create(text_dataa);
            text_data.setName("text");

            // 目标音色作为Speaker Encoder的输入
            // 使用ffmpeg 将目标音色mp3文件转为wav格式
            Path audioFile = new File(sound).toPath();
            NDArray audioArray = FfmpegUtils.load_wav_to_torch(audioFile.toString(), 22050,manager);

//            System.out.println("audioArray:"+audioArray.getClass());

            // 使用 Speaker Embedding 模型进行音色提取
            int partials_n_frames = 160;
            Pair<LinkedList<LinkedList<Integer>>, LinkedList<LinkedList<Integer>>> slices = AudioUtils.compute_partial_slices(audioArray.size(), partials_n_frames, 0.75f, 0.5f);
            LinkedList<LinkedList<Integer>> wave_slices = slices.getLeft();
            LinkedList<LinkedList<Integer>> mel_slices = slices.getRight();
            int max_wave_length = wave_slices.getLast().getLast();
            if (max_wave_length >= audioArray.size()) {
                audioArray = AudioUtils.pad(audioArray, (max_wave_length - audioArray.size()), manager);
            }
            float[][] fframes = AudioUtils.wav_to_mel_spectrogram(audioArray);
            NDArray frames = manager.create(fframes).transpose();
            NDList frameslist = new NDList();
            for (LinkedList<Integer> s : mel_slices) {
                NDArray temp = speakerEncoderPredictor.predict(frames.get(s.getFirst() + ":" + s.getLast()));
                frameslist.add(temp);
            }
            NDArray partial_embeds = NDArrays.stack(frameslist);
            NDArray raw_embed = partial_embeds.mean(new int[]{0});

            NDArray speaker_data = raw_embed.div(((raw_embed.pow(2)).sum()).sqrt());

            Shape shape = speaker_data.getShape();
            System.out.println("音色提取结果:");
            System.out.println("音色shape:" + Arrays.toString(shape.getShape()));
            System.out.println("音色vector:" + Arrays.toString(speaker_data.toFloatArray()));

            // 模型数据
            NDList input = new NDList();
            input.add(text_data);
            input.add(speaker_data);

            // 生成mel频谱数据
            NDArray mels_postnet = tacotron2Predictor.predict(input);
            shape = mels_postnet.getShape();
            System.out.println("mel频谱数据:");
            System.out.println("mel频谱shape:" + Arrays.toString(shape.getShape()));
            System.out.println("mel频谱vector:" + Arrays.toString(mels_postnet.toFloatArray()));

            // 生成wav数据
            NDArray wavWithNoise = waveGlowPredictor.predict(mels_postnet);
            NDArray wav = denoiserPredictor.predict(wavWithNoise);
            SoundUtils.saveWavFile(wav.get(0), 1.0f, new File(out),manager);



            System.out.println("生成成功:");
            System.out.println(out);

        }


    }




}
