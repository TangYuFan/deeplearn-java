package tool.deeplearning;


import ai.onnxruntime.*;
import org.opencv.core.Core;

import java.io.BufferedReader;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 *   @desc : 中文分词 lightltp 中文词法分析（分词、词性标注 => 命名实体识别）
 *
 *          推理参考：
 *          https://github.com/zejunwang1/lightltp
 *
 *          参考推理文件：
 *          test_onnxruntime_inference.py
 *
 *          哈工大参考文档：
 *          http://ltp.ai/docs/quickstart.html#id8
 *
 *          ltp：
 *          https://github.com/HIT-SCIR/ltp
 *          是哈工大社会计算和信息检索研究中心（HIT-SCIR）开源的中文自然语言处理工具集，
 *          用户可以使用 ltp 对中文文本进行分词、词性标注、命名实体识别、语义角色标注、依存句法分析、语义依存分析等等工作。
 *
 *   @auth : tyf
 *   @date : 2022-05-17  09:21:23
 */
public class chinese_segment_lightltp_2 {



    // 模型
    public static OrtEnvironment env1;
    public static OrtSession session1;

    // 模型
    public static OrtEnvironment env2;
    public static OrtSession session2;


    // 字典索引映射
    public static Map<String, Integer> token_id_map = new HashMap<>();
    public static Map<Integer, String> id_token_map = new HashMap<>();

    // 27种词性列表
    public static String[] pos_map_en = new String[]{
            "n", "v", "wp", "u", "d",
            "a", "m", "p", "r", "ns",
            "c", "q", "nt", "nh", "nd",
            "j", "i", "b", "ni", "nz",
            "nl", "z", "k", "ws", "o",
            "h", "e"};

    // 27种词性列表的中文
    public static String[] pos_map_ch = new String[]{
            "名词", "动词", "标点", "助词", "副词",
            "形容词", "数词", "介词", "代词", "地名",
            "连词", "量词", "机构团体", "人名", "方位词",
            "简称", "成语", "习用语", "其他名词", "专有名词",
            "其他专名", "字符", "后缀", "外文字符", "拟声词",
            "前缀", "错误字符"
    };

    // 命名实体分为位置标识和类型标识
    // B -> 实体开始词
    // I -> 实体中间词
    // E -> 实体结束词
    // S -> 单独构成实体
    // O -> 不构成实体
    // Nh->人名 Ni->机构名 Ns->地名
    public static String[] ner_map_en = new String[]{
            "O", "S-Ns", "S-Nh", "B-Ni", "E-Ni", "I-Ni", "S-Ni", "B-Ns", "E-Ns", "I-Ns", "B-Nh", "E-Nh", "I-Nh"
    };

    // 环境初始化
    public static void init1(String weight) throws Exception{
        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env1 = OrtEnvironment.getEnvironment();
        session1 = env1.createSession(weight, new OrtSession.SessionOptions());

        // 打印模型信息,获取输入输出的shape以及类型：
        System.out.println("---------模型1输入-----------");
        session1.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型1输出-----------");
        session1.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });
        session1.getMetadata().getCustomMetadata().entrySet().forEach(n->{
            System.out.println("元数据:"+n.getKey()+","+n.getValue());
        });




    }

    // 环境初始化
    public static void init2(String weight) throws Exception{
        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env2 = OrtEnvironment.getEnvironment();
        session2 = env2.createSession(weight, new OrtSession.SessionOptions());

        // 打印模型信息,获取输入输出的shape以及类型：
        System.out.println("---------模型2输入-----------");
        session2.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型2输出-----------");
        session2.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });
        session2.getMetadata().getCustomMetadata().entrySet().forEach(n->{
            System.out.println("元数据:"+n.getKey()+","+n.getValue());
        });


    }

    public static class TextObj{
        // 需要处理的文本
        String text;
        // 保存每个分割的单词
        ArrayList<String> words = new ArrayList<>();
        // 保存每个分割的单词的词性
        ArrayList<String> poss_en = new ArrayList<>();
        ArrayList<String> poss_ch = new ArrayList<>();
        // 保存 word_input 和 word_attention_mask 作为模型2的输入
        float[][][] word_input = null;
        public TextObj(String text) {
            this.text = text;
        }
        public static int getMaxIndex(float[] array) {
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
        // 模型推理
        public void inference1() throws Exception{

            System.out.println("----------------------------inference------------------------------");

            // 每个字符
            char[] chars = text.toCharArray();
            List<String> charSrt = new ArrayList<>();
            for (int i = 0; i < chars.length; i++) {
                charSrt.add(String.valueOf(chars[i]));
            }
            charSrt.add(0,"[CLS]");
            charSrt.add("[SEP]");
            // 文本转id索引
            List<Integer> ids = new ArrayList<>();
            for (int i = 0; i < charSrt.size(); i++) {
                String s = charSrt.get(i);
                Integer id = token_id_map.get(s.toLowerCase());
                // 可能存在字库中不存在的字,使用固定的id
                if(id!=null){
                    ids.add(id);
                }else {
                    ids.add(1);
                }
            }


            // 模型输入
            int count = ids.size();
            long[] inputIds = new long[count];
            long[] attentionMask = new long[count];
            long[] tokenTypeIds = new long[count];
            long[] shape = new long[]{1, count};

            // ---------模型输入-----------
            // input_ids -> [-1, -1] -> INT64
            // attention_mask -> [-1, -1] -> INT64
            // token_type_ids -> [-1, -1] -> INT64
            for(int i=0; i < ids.size(); i ++) {
                inputIds[i] = ids.get(i);
                attentionMask[i] = 1;
                tokenTypeIds[i] = 0;
            }

            // ---------模型输入-----------
            // input_ids -> [-1, -1] -> INT64
            // attention_mask -> [-1, -1] -> INT64
            // token_type_ids -> [-1, -1] -> INT64
            // 输入数组转为张量
            OnnxTensor input_ids = OnnxTensor.createTensor(env1, OrtUtil.reshape(inputIds, shape));
            OnnxTensor attention_mask = OnnxTensor.createTensor(env1, OrtUtil.reshape(attentionMask, shape));
            OnnxTensor token_type_ids = OnnxTensor.createTensor(env1, OrtUtil.reshape(tokenTypeIds, shape));

            Map<String,OnnxTensor> input = new HashMap<>();
            input.put("input_ids",input_ids);
            input.put("attention_mask",attention_mask);
            input.put("token_type_ids",token_type_ids);

            // 推理
            OrtSession.Result out = session1.run(input);


            // ---------模型输出-----------
            // seg_output -> [-1, -1, 2] -> FLOAT
            // pos_output -> [-1, -1, 27] -> FLOAT
            // char_input -> [-1, -1, 256] -> FLOAT
            OnnxValue onnxValue1 = out.get(0);
            OnnxValue onnxValue2 = out.get(1);
            OnnxValue onnxValue3 = out.get(2);

            // 1 * n * 2 其中n就是输入文本的长度,2代表当前字是否是一个新的词语的开始
            float[][][] seg_output = (float[][][]) onnxValue1.getValue();
            // 1 * n * 27 其中n就是输入文本的长度,27代表了27中词性的概率
            float[][][] pos_output = (float[][][]) onnxValue2.getValue();
            // 这个用于后续的命名实体识别
            float[][][] char_input = (float[][][]) onnxValue3.getValue();


            StringBuffer tmp = new StringBuffer();
            // 遍历每个字
            for(int i=0;i<seg_output[0].length;i++){
                // 2维,表示每个字是否属于单词的开始
                float seg_1 = seg_output[0][i][0];
                float seg_2 = seg_output[0][i][1];
                // 遇到了单词的开始
                if(seg_1<seg_2){
                    // 保存上一个单词
                    if(tmp.toString().length()>0){
                        words.add(tmp.toString());
                    }
                    // 重新创建一个单词,并拼接当前字
                    tmp = new StringBuffer();
                    tmp.append(chars[i]);
                    // 每次一个新开始的单词,取第一个字的词性作为单词词性
                    // 27维,表示词性的概率,获取最大索引
                    float[] pos = pos_output[0][i];
                    int maxIndex = getMaxIndex(pos);
                    // 根据最大概率索引从词性列表中取出词性字符串
                    poss_en.add(pos_map_en[maxIndex]);
                    poss_ch.add(pos_map_ch[maxIndex]);
                }else{
                    tmp.append(chars[i]);
                }
            }
            // 保存最后一个单词
            if(tmp.toString().length()>0){
                words.add(tmp.toString());
            }


            // 保存 word_input 作为模型2的输入
            this.word_input = char_input;
        }

        // 命名实体识别
        public void inference2() throws Exception{

            // 字符个数
            int count = this.word_input[0].length;

            // 打印输入
            float[][][] word_input = this.word_input;
            long[][] word_attention_mask = new long[1][count];
            for(int i=0;i<count;i++){
                word_attention_mask[0][i] = 1;
            }

            // ---------模型2输入-----------
            // word_input -> [-1, -1, 256] -> FLOAT
            // word_attention_mask -> [-1, -1] -> INT64
            OnnxTensor word_input_tens = OnnxTensor.createTensor(env2,word_input);
            OnnxTensor word_attention_mask_tens = OnnxTensor.createTensor(env2,word_attention_mask);

            Map<String,OnnxTensor> input = new HashMap<>();
            input.put("word_input",word_input_tens);
            input.put("word_attention_mask",word_attention_mask_tens);

            // ---------模型2输出-----------
            // ner_output -> [-1, -1, 13] -> FLOAT   1 * n * 13 表示每个字属于哪一种命名实体的概率
            OrtSession.Result out = session2.run(input);
            OnnxValue onnxValue1 = out.get(0);
            float[][][] ner_output = (float[][][]) onnxValue1.getValue();

            // 依然是取每个单词开头的第一个字计算 13 中命名实体的概率作为每个单词的命名实体类型,所以要遍历前面已经得到的分析
            ArrayList<Integer> index = new ArrayList<>();
            Integer tmp = 0;
            // 单词个数
            int wordCount  = words.size();
            for (int i = 0; i < wordCount; i++) {
                // 当前单词
                String now_word = words.get(i);
                index.add(tmp);
                tmp = tmp + now_word.length();
            }

            // 获取每个位置的概率数组
            ArrayList nerList = new ArrayList();
            for (int i = 0; i < ner_output[0].length; i++) {
                float[] gailv = ner_output[0][i];
                int in = getMaxIndex(gailv);
                String ner = ner_map_en[in];
                nerList.add(ner);
            }
            System.out.println("原始文本:");
            System.out.println(text);
            System.out.println("单词分割:");
            System.out.println(Arrays.toString(words.toArray()));
            System.out.println("命名实体:");
            System.out.println(Arrays.toString(nerList.toArray()));

        }


        // 打印结果
        public void show(){

        }
    }

    // 将字符和id索引进行保存
    public static void initVocab(String path) throws Exception{

        // 字符 => index
        try(BufferedReader br = Files.newBufferedReader(Paths.get(path), StandardCharsets.UTF_8)) {
            String line;
            int index = 0;
            while ((line = br.readLine()) != null) {
                token_id_map.put(line.trim().toLowerCase(), index);
                index ++;
            }
        }

        // index => 字符
        for (String key : token_id_map.keySet()) {
            id_token_map.put(token_id_map.get(key), key);
        }

    }


    public static void main(String[] args) throws Exception{

        // ---------模型输入-----------
        // input_ids -> [-1, -1] -> INT64
        // attention_mask -> [-1, -1] -> INT64
        // token_type_ids -> [-1, -1] -> INT64
        // ---------模型输出-----------
        // seg_output -> [-1, -1, 2] -> FLOAT
        // pos_output -> [-1, -1, 27] -> FLOAT
        // char_input -> [-1, -1, 256] -> FLOAT
        String n1 = "segpos_cpu_opt.onnx";


        // ---------模型2输入-----------
        // word_input -> [-1, -1, 256] -> FLOAT
        // word_attention_mask -> [-1, -1] -> INT64
        // ---------模型2输出-----------
        // ner_output -> [-1, -1, 13] -> FLOAT
        String n2 = "ner_cpu.onnx";

        // 模型1
        init1(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\chinese_segment_lightltp\\"+n1);

        // 模型2
        init2(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\chinese_segment_lightltp\\"+n2);

        // 加载字符库,每个字需要转为字符库中的索引
        initVocab(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\chinese_segment_lightltp\\"+"vocab.txt");

        // 需要处理的文本
        String text = "四川省成都市金牛区一环路北二段，唐于凡，13388236293";

        TextObj textObj = new TextObj(text);

        // 模型1: 实现分词 + 词性标注
        textObj.inference1();

        // 模型2: 实现命名实体识别
        textObj.inference2();

        // 输出结果
        textObj.show();

    }

}
