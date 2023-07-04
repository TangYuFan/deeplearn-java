package tool.deeplearning;


import ai.onnxruntime.*;
import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.Core;

import java.io.BufferedReader;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/**
*   @desc : 加载macbert模型，中文拼写纠错
 *
 *
 *          其他中文纠错模型：
 *
 *          Kenlm模型：本项目基于Kenlm统计语言模型工具训练了中文NGram语言模型，结合规则方法、混淆集可以纠正中文拼写错误，方法速度快，扩展性强，效果一般
 *          https://github.com/shibing624/pycorrector/blob/master/pycorrector/corrector.py
 *
 *          MacBERT模型【推荐】：本项目基于PyTorch实现了用于中文文本纠错的MacBERT4CSC模型，模型加入了错误检测和纠正网络，适配中文拼写纠错任务，效果好
 *          https://github.com/shibing624/pycorrector/blob/master/pycorrector/macbert
 *
 *          Seq2Seq模型：本项目基于PyTorch实现了用于中文文本纠错的Seq2Seq模型、ConvSeq2Seq模型，其中ConvSeq2Seq在NLPCC-2018的中文语法纠错比赛中，使用单模型并取得第三名，可以并行训练，模型收敛快，效果一般
 *          https://github.com/shibing624/pycorrector/blob/master/pycorrector/seq2seq
 *
 *          T5模型：本项目基于PyTorch实现了用于中文文本纠错的T5模型，使用Langboat/mengzi-t5-base的预训练模型fine-tune中文纠错数据集，模型改造的潜力较大，效果好
 *          https://github.com/shibing624/pycorrector/blob/master/pycorrector/t5
 *
 *          BERT模型：本项目基于PyTorch实现了基于原生BERT的fill-mask能力进行纠正错字的方法，效果差
 *          https://github.com/shibing624/pycorrector/blob/master/pycorrector/bert
 *
 *          ELECTRA模型：本项目基于PyTorch实现了基于原生ELECTRA的fill-mask能力进行纠正错字的方法，效果差
 *          https://github.com/shibing624/pycorrector/blob/master/pycorrector/electra
 *
 *          ERNIE_CSC模型：本项目基于PaddlePaddle实现了用于中文文本纠错的ERNIE_CSC模型，模型在ERNIE-1.0上fine-tune，模型结构适配了中文拼写纠错任务，效果好
 *          https://github.com/shibing624/pycorrector/blob/master/pycorrector/ernie_csc
 *
 *          DeepContext模型：本项目基于PyTorch实现了用于文本纠错的DeepContext模型，该模型结构参考Stanford University的NLC模型，2014英文纠错比赛得第一名，效果一般
 *         https://github.com/shibing624/pycorrector/blob/master/pycorrector/deepcontext
 *
 *          Transformer模型：本项目基于PyTorch的fairseq库调研了Transformer模型用于中文文本纠错，效果一般
 *          https://github.com/shibing624/pycorrector/blob/master/pycorrector/transformer
 *
*   @auth : tyf
*   @date : 2022-05-17  10:49:16
*/
public class chinese_error_recovery_macbert4csc {

    // 模型1
    public static OrtEnvironment env;
    public static OrtSession session;

    // 字典索引映射
    public static Map<String, Integer> token_id_map = new HashMap<>();
    public static Map<Integer, String> id_token_map = new HashMap<>();
    // 环境初始化
    public static void init1(String weight) throws Exception{
        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env = OrtEnvironment.getEnvironment();
        session = env.createSession(weight, new OrtSession.SessionOptions());

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


    public static class TextObj{
        // 设置初始文本
        String text;
        // 保存纠正后的文本
        String text_c;
        public void setText(String text) {
            this.text = text;
        }
        public void inference() throws Exception{

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
                ids.add(id);
            }

            // 打印文本和id索引
            System.out.println("打印文本和id索引");
            System.out.println(Arrays.toString(charSrt.toArray()));
            System.out.println(Arrays.toString(ids.toArray()));

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

            // 输入数组转为张量
            OnnxTensor input_ids = OnnxTensor.createTensor(env, OrtUtil.reshape(inputIds, shape));
            OnnxTensor attention_mask = OnnxTensor.createTensor(env, OrtUtil.reshape(attentionMask, shape));
            OnnxTensor token_type_ids = OnnxTensor.createTensor(env, OrtUtil.reshape(tokenTypeIds, shape));

            Map<String,OnnxTensor> input = new HashMap<>();
            input.put("input_ids",input_ids);
            input.put("attention_mask",attention_mask);
            input.put("token_type_ids",token_type_ids);

            // 推理
            OrtSession.Result out = session.run(input);

            // ---------模型输出-----------
            // output_0 -> [-1, -1, 21128] -> FLOAT
            OnnxValue onnxValue = out.get(0);
            float[][][] labels = (float[][][]) onnxValue.getValue();
            // 转为二位数组, 字个数 * 21128
            INDArray indArrayLabels = Nd4j.create(labels[0]);
            // 每个字在 21128 个概率中的最大索引
            INDArray index = Nd4j.argMax(indArrayLabels, -1);
            int[] predIndex = index.toIntVector();

            // 纠正后的文本
            StringBuffer predTokens = new StringBuffer();
            // 索引转文字
            for(int idx = 1; idx < predIndex.length -1; idx++) {
                // id转字符
                Integer id = predIndex[idx];
                String token = id_token_map.get(id);
                predTokens.append(token);
            }

            // 保存纠正后的文本
            text_c = predTokens.toString();

        }
        public void show(){

            System.out.println("原始文本:"+text);
            System.out.println("纠正文本:"+text_c);

        }
    }


    public static void main(String[] args) throws Exception{

        // ---------模型输入-----------
        // input_ids -> [-1, -1] -> INT64
        // attention_mask -> [-1, -1] -> INT64
        // token_type_ids -> [-1, -1] -> INT64
        // ---------模型输出-----------
        // output_0 -> [-1, -1, 21128] -> FLOAT
        init1(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\chinese_error_recovery_macbert4csc\\macbert4csc.onnx");

        // 字典
        initVocab(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\chinese_error_recovery_macbert4csc\\vocab.txt");


        // 文本处理器
        TextObj textObj = new TextObj();
        textObj.setText("清问你在干什么？");

        // 推理
        textObj.inference();

        // 输出结果
        textObj.show();

    }

}
