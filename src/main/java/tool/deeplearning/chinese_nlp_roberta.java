package tool.deeplearning;


import ai.onnxruntime.*;
import org.opencv.core.Core;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.Normalizer;
import java.util.*;
import java.util.stream.Collectors;

/**
*   @desc : 中文填词预测： roberta
 *
 *
 *
*   @auth : tyf
*   @date : 2022-05-23  14:26:10
*/
public class chinese_nlp_roberta {


    // 模型1
    public static OrtEnvironment env;
    public static OrtSession session;


    // 字典索引映射
    public static Map<String, Integer> token_id_map = new HashMap<>();
    public static Map<Integer, String> id_token_map = new HashMap<>();


    // 环境初始化
    public static void init(String weight) throws Exception{
        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        session = env.createSession(weight, options);


        // 打印模型信息,获取输入输出的shape以及类型：
        System.out.println("---------模型输入-----------");
        session.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型输出-----------");
        session.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });
        session.getMetadata().getCustomMetadata().entrySet().forEach(n->{
            System.out.println("元数据:"+n.getKey()+","+n.getValue());
        });

    }

    // 将字符和id索引进行保存
    public static void initVocab(String path) throws Exception{
        try(BufferedReader br = Files.newBufferedReader(Paths.get(path), StandardCharsets.UTF_8)) {
            String line;
            int index = 0;
            while ((line = br.readLine()) != null) {
                token_id_map.put(line.trim().toLowerCase(), index);
                index ++;
            }
        }
        id_token_map = new HashMap<>();
        for (String key : token_id_map.keySet()) {
            id_token_map.put(token_id_map.get(key), key);
        }
    }

    public static class BasicTokenizer {
        private boolean do_lower_case = true;
        private List<String> never_split;
        private boolean tokenize_chinese_chars = true;
        private List<String> specialTokens;
        public BasicTokenizer(boolean do_lower_case, List<String> never_split, boolean tokenize_chinese_chars) {
            this.do_lower_case = do_lower_case;
            if (never_split == null) {
                this.never_split = new ArrayList<>();
            } else {
                this.never_split = never_split;
            }
            this.tokenize_chinese_chars = tokenize_chinese_chars;
        }

        public BasicTokenizer() {
        }

        public String clean_text(String text) {
            // Performs invalid character removal and whitespace cleanup on text."""

            StringBuilder output = new StringBuilder();
            for (int i = 0; i < text.length(); i++) {
                Character c = text.charAt(i);
                int cp = (int) c;
                if (cp == 0 || cp == 0xFFFD || _is_control(c)) {
                    continue;
                }
                if (_is_whitespace(c)) {
                    output.append(" ");
                } else {
                    output.append(c);
                }
            }
            return output.toString();
        }

        private boolean _is_control(char c) {
            // Checks whether `chars` is a control character.
            // These are technically control characters but we count them as whitespace
            // characters.
            if (c == '\t' || c == '\n' || c == '\r') {
                return false;
            }

            int charType = Character.getType(c);
            if (Character.CONTROL == charType || Character.DIRECTIONALITY_COMMON_NUMBER_SEPARATOR == charType
                    || Character.FORMAT == charType || Character.PRIVATE_USE == charType || Character.SURROGATE == charType
                    || Character.UNASSIGNED == charType) {
                return true;
            }
            return false;
        }


        private boolean _is_whitespace(char c) {
            // Checks whether `chars` is a whitespace character.
            // \t, \n, and \r are technically contorl characters but we treat them
            // as whitespace since they are generally considered as such.
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                return true;
            }

            int charType = Character.getType(c);
            if (Character.SPACE_SEPARATOR == charType) {
                return true;
            }
            return false;
        }

        private boolean _is_chinese_char(int cp) {
            // Checks whether CP is the codepoint of a CJK character."""
            // This defines a "chinese character" as anything in the CJK Unicode block:
            // https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
            //
            // Note that the CJK Unicode block is NOT all Japanese and Korean characters,
            // despite its name. The modern Korean Hangul alphabet is a different block,
            // as is Japanese Hiragana and Katakana. Those alphabets are used to write
            // space-separated words, so they are not treated specially and handled
            // like the all of the other languages.
            if ((cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF)
                    || (cp >= 0x2A700 && cp <= 0x2B73F) || (cp >= 0x2B740 && cp <= 0x2B81F)
                    || (cp >= 0x2B820 && cp <= 0x2CEAF) || (cp >= 0xF900 && cp <= 0xFAFF)
                    || (cp >= 0x2F800 && cp <= 0x2FA1F)) {
                return true;
            }

            return false;
        }


        public String tokenize_chinese_chars(String text) {
            // Adds whitespace around any CJK character.
            StringBuilder output = new StringBuilder();
            for (int i = 0; i < text.length(); i++) {
                Character c = text.charAt(i);
                int cp = (int) c;
                if (_is_chinese_char(cp)) {
                    output.append(" ");
                    output.append(c);
                    output.append(" ");
                } else {
                    output.append(c);
                }
            }
            return output.toString();
        }

        public List<String> whitespace_tokenize(String text) {
            // Runs basic whitespace cleaning and splitting on a piece of text.
            text = text.trim();
            if ((text != null) && (text != "")) {
                return new ArrayList<>(Arrays.asList(text.split("\\s+")));
            }
            return new ArrayList<>();

        }


        public String run_strip_accents(String token) {
            token = Normalizer.normalize(token, Normalizer.Form.NFD);
            StringBuilder output = new StringBuilder();
            for (int i = 0; i < token.length(); i++) {
                Character c = token.charAt(i);
                if (Character.NON_SPACING_MARK != Character.getType(c)) {
                    output.append(c);
                }
            }
            return output.toString();
        }

        private boolean _is_punctuation(char c) {
            // Checks whether `chars` is a punctuation character.
            int cp = (int) c;
            // We treat all non-letter/number ASCII as punctuation.
            // Characters such as "^", "$", and "`" are not in the Unicode
            // Punctuation class but we treat them as punctuation anyways, for
            // consistency.
            if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) {
                return true;
            }
            int charType = Character.getType(c);
            if (Character.CONNECTOR_PUNCTUATION == charType || Character.DASH_PUNCTUATION == charType
                    || Character.END_PUNCTUATION == charType || Character.FINAL_QUOTE_PUNCTUATION == charType
                    || Character.INITIAL_QUOTE_PUNCTUATION == charType || Character.OTHER_PUNCTUATION == charType
                    || Character.START_PUNCTUATION == charType) {
                return true;
            }
            return false;
        }

        public List<String> run_split_on_punc(String token, List<String> never_split) {
            // Splits punctuation on a piece of text.
            List<String> output = new ArrayList<>();
            if (Optional.of(never_split).isPresent()) {
                if(never_split.contains(token)) {
                    output.add(token);
                    return output;
                } else {
                    for(String specialToken : never_split) {
                        if(token.contains(specialToken)) {
                            int specialTokenIndex = token.indexOf(specialToken);
                            if(specialTokenIndex == 0) {
                                String other = token.substring(specialToken.length());
                                output.add(specialToken);
                                output.add(other);
                                return output;
                            } else {
                                String other = token.substring(0, token.indexOf(specialToken));
                                output.add(other);
                                output.add(specialToken);
                                String another = token.substring(specialTokenIndex + specialToken.length());
                                if (another.length() != 0) {
                                    output.add(another);
                                }
                                return output;
                            }
                        }
                    }
                }
            }

            boolean start_new_word = true;
            StringBuilder str = new StringBuilder();
            for (int i = 0; i < token.length(); i++) {
                Character c = token.charAt(i);
                if (_is_punctuation(c)) {
                    if (str.length() > 0) {
                        output.add(str.toString());
                        str.setLength(0);
                    }
                    output.add(c.toString());
                    start_new_word = true;
                } else {
                    if (start_new_word && str.length() > 0) {
                        output.add(str.toString());
                        str.setLength(0);
                    }
                    start_new_word = false;
                    str.append(c);
                }
            }
            if (str.length() > 0) {
                output.add(str.toString());
            }
            return output;
        }

        public List<String> tokenize(String text) {
            text = clean_text(text);
            if (tokenize_chinese_chars) {
                text = tokenize_chinese_chars(text);
            }
            List<String> orig_tokens = whitespace_tokenize(text);
            List<String> split_tokens = new ArrayList<>();
            for (String token : orig_tokens) {
                if (do_lower_case && !never_split.contains(token)) {
                    token = run_strip_accents(token);
                    split_tokens.addAll(run_split_on_punc(token, never_split));
                } else {
                    split_tokens.add(token);
                }
            }
            return whitespace_tokenize(String.join(" ", split_tokens));
        }

    }

    public static class BertTokenizer{

        private boolean do_lower_case = true;
        private boolean do_basic_tokenize = true;
        private List<String> never_split;
        public String unk_token = "[UNK]";
        public String sep_token = "[SEP]";
        public String pad_token = "[PAD]";
        public String cls_token = "[CLS]";
        public String mask_token = "[MASK]";
        private boolean tokenize_chinese_chars = true;
        private BasicTokenizer basic_tokenizer;
        private WordpieceTokenizer wordpiece_tokenizer;

        public BertTokenizer() {
            init();
        }

        private void init() {
            never_split = new ArrayList<>();
            never_split.add(unk_token);
            never_split.add(sep_token);
            never_split.add(pad_token);
            never_split.add(cls_token);
            never_split.add(mask_token);
            if (do_basic_tokenize) {
                this.basic_tokenizer = new BasicTokenizer(do_lower_case, never_split, tokenize_chinese_chars);
            }
            this.wordpiece_tokenizer = new WordpieceTokenizer(token_id_map, unk_token, never_split);
        }

        public List<Integer> maskIndex(List<String > tokens){
            List<Integer> res = new ArrayList<>();
            for (int i = 0; i < tokens.size(); i++) {
                if(tokens.get(i).contains("MASK")){
                    res.add(i);
                }
            }
            return res;
        }

        public List<String> tokenize(String text) {
            List<String> split_tokens = new ArrayList<>();
            if (do_basic_tokenize) {
                for (String token : basic_tokenizer.tokenize(text)) {
                    for (String sub_token : wordpiece_tokenizer.tokenize(token)) {
                        split_tokens.add(sub_token);
                    }
                }
            } else {
                split_tokens = wordpiece_tokenizer.tokenize(text);
            }
            split_tokens.add(0, "[CLS]");
            split_tokens.add("[SEP]");
            return split_tokens;
        }

        public List<Integer> convert_tokens_to_ids(List<String> tokens) {
            List<Integer> output = new ArrayList<>();
            for (String s : tokens) {
                output.add(token_id_map.get(s.toLowerCase()));
            }
            return output;
        }

    }

    public static class WordpieceTokenizer{
        private Map<String, Integer> vocab;
        private String unk_token;
        private int max_input_chars_per_word;
        private List<String> specialTokensList;
        public WordpieceTokenizer(Map<String, Integer> vocab, String unk_token, List<String> specialTokensList) {
            this.vocab = vocab;
            this.unk_token = unk_token;
            this.specialTokensList = specialTokensList;
            this.max_input_chars_per_word = 100;
        }

        public List<String> whitespace_tokenize(String text) {
            // Runs basic whitespace cleaning and splitting on a piece of text.
            text = text.trim();
            if ((text != null) && (text != "")) {
                return new ArrayList<>(Arrays.asList(text.split("\\s+")));
            }
            return new ArrayList<>();

        }


        public List<String> tokenize(String text) {
            List<String> output_tokens = new ArrayList<>();
            if(this.specialTokensList.contains(text)) {
                output_tokens.add(text);
                return output_tokens;
            }
            for (String token : whitespace_tokenize(text)) {
                if (token.length() > max_input_chars_per_word) {
                    output_tokens.add(unk_token);
                    continue;
                }
                boolean is_bad = false;
                int start = 0;

                List<String> sub_tokens = new ArrayList<>();
                while (start < token.length()) {
                    int end = token.length();
                    String cur_substr = "";
                    while (start < end) {
                        String substr = token.substring(start, end);
                        if (start > 0) {
                            substr = "##" + substr;
                        }
                        if (vocab.containsKey(substr)) {
                            cur_substr = substr;
                            break;
                        }
                        end -= 1;
                    }
                    if (cur_substr == "") {
                        is_bad = true;
                        break;
                    }
                    sub_tokens.add(cur_substr);
                    start = end;
                }
                if (is_bad) {
                    output_tokens.add(unk_token);
                } else {
                    output_tokens.addAll(sub_tokens);
                }
            }
            return output_tokens;
        }
    }


    public static class TextObj{

        // 设置初始文本
        String text;
        // 保存预测结果
        List<List<String>> pre;
        public void setText(String text) {
            this.text = text;
            this.pre = new ArrayList<>();
        }
        public int getMaxIndex(float[] array) {
            int maxIndex = 0;
            float maxVal = array[0];
            for (int i = 1; i < array.length; i++) {
                if (array[i] > maxVal) {
                    maxVal = array[i];
                    maxIndex = i;
                }
            }
            return maxIndex;
        }

        public List<Integer> sortByScore(float[] score){
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < score.length; i++) {
                indices.add(i);
            }
            Collections.sort(indices, (i1, i2) -> Float.compare(score[i1], score[i2]));
            return indices;
        }

        public void inference() throws Exception{


            // 用于token和id转换
            BertTokenizer tokenizer = new BertTokenizer();

            // 保存token
            List<String > tokens = tokenizer.tokenize(text);
            // 保存ID
            List<Integer> tokenIds = tokenizer.convert_tokens_to_ids(tokens);

            //MASK index
            List<Integer> maskIndex = tokenizer.maskIndex(tokens);

//            System.out.println("输入个数:"+tokens.size());
//            System.out.println("tokens:"+tokens);
//            System.out.println("tokenIds:"+tokenIds);
//            System.out.println("maskIndex:"+maskIndex);

            // 转为向量
            long[] inputIds = new long[tokenIds.size()];
            long[] attentionMask = new long[tokenIds.size()];
            long[] tokenTypeIds = new long[tokenIds.size()];
            for(int index=0; index < tokenIds.size(); index ++) {
                inputIds[index] = tokenIds.get(index);
                attentionMask[index] = 1;
                tokenTypeIds[index] = 0;
            }
            long[] shape = new long[]{1, inputIds.length};

            // 张量
            OnnxTensor input_ids = OnnxTensor.createTensor(env, OrtUtil.reshape(inputIds, shape));
            OnnxTensor attention_mask = OnnxTensor.createTensor(env, OrtUtil.reshape(attentionMask, shape));
            OnnxTensor token_type_ids = OnnxTensor.createTensor(env, OrtUtil.reshape(tokenTypeIds, shape));

            // ---------模型输入-----------
            // input_ids -> [-1, -1] -> INT64
            // attention_mask -> [-1, -1] -> INT64
            // token_type_ids -> [-1, -1] -> INT64
            Map<String,OnnxTensor> inputs = new HashMap<>();
            inputs.put("input_ids",input_ids);
            inputs.put("attention_mask",attention_mask);
            inputs.put("token_type_ids",token_type_ids);


            // ---------模型输出-----------
            // output_0 -> [-1, -1, 21128] -> FLOAT
            try(OrtSession.Result results = session.run(inputs)) {
                OnnxValue onnxValue = results.get(0);
                // 1 * n * 21128    其中n是输入个数,也就是字数+2,根据前面[mask]所在的index来取 21128数组
                float[][][] labels = (float[][][]) onnxValue.getValue();
                // 遍历每个需要预测的mask位置
                for (int i = 0; i < maskIndex.size(); i++) {
                    // 取预测结果分数21128数组
                    float[] scoreData = labels[0][maskIndex.get(i)];
                    // 表示 21128 种字的可行性,这里将分数按照降序对index进行排序
                    List<Integer> indexs = sortByScore(scoreData);
                    // 取前5个index,转为token
                    String v1 = id_token_map.get(indexs.get(21128-1));
                    String v2 = id_token_map.get(indexs.get(21128-2));
                    String v3 = id_token_map.get(indexs.get(21128-3));
                    String v4 = id_token_map.get(indexs.get(21128-4));
                    String v5 = id_token_map.get(indexs.get(21128-5));
                    List<String> res = new ArrayList<>();
                    res.add(v1);
                    res.add(v2);
                    res.add(v3);
                    res.add(v4);
                    res.add(v5);
                    pre.add(res);

                }
            }

        }
        public void show(){

            System.out.println();
            System.out.println("原始文本:");
            System.out.println(text);
            System.out.println("预测结果:");
            // 打印每个mask的预测结果
            for (int i = 0; i < pre.size(); i++) {
                List<String> data = pre.get(i);
                System.out.println("[MASK]"+i+" ："+data);
            }

        }
    }

    public static void main(String[] args) throws Exception{


        // ---------模型输入-----------
        // input_ids -> [-1, -1] -> INT64
        // attention_mask -> [-1, -1] -> INT64
        // token_type_ids -> [-1, -1] -> INT64
        // ---------模型输出-----------
        // output_0 -> [-1, -1, 21128] -> FLOAT
        init(new File("").getCanonicalPath() +
                "\\model\\deeplearning\\chinese_nlp_roberta\\roberta.onnx");


        // 字典
        initVocab(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\chinese_nlp_roberta\\vocab.txt");

        // 文本处理器
        TextObj textObj = new TextObj();
        textObj.setText("我家后面有一[MASK]大树。但是这[MASK]大树并不大。");

        // 推理
        textObj.inference();

        // 输出结果
        textObj.show();

    }


}
