package tool.deeplearning;


import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
*   @desc : 英文命名实体识别 HuggingFace-Roberta NER (NLP)
 *
 *
 *          模型onnx导出，直接登录网站在线转换（安装环境->onnx->合并压缩zip,下载里面包含了模型和token）：
 *          https://colab.research.google.com/drive/1kZx9XOnExVfPoAGHhHRUrdQnioiLloBW#revisionId=0BwKss6yztf4KS0NKaWRiQjc0RGRvQkd6ZFp3OUFhR1lTclBNPQ&scrollTo=aWB2G_kLHou2
 *
*   @auth : tyf
*   @date : 2022-05-23  14:22:35
*/
public class english_segment_ner {

    static long[] inputIds;
    static long[] inputAttentionMask;
    static Tokenizer tokenizer;
    static String persons = "";
    static String locations = "";
    static String organizations = "";
    static String misc = "";

    // token 处理器,需要调用 ai.djl,huggingface 的处理器
    public static class Tokenizer {
        private Encoding encoding;
        private final HuggingFaceTokenizer tokenizer;
        public Tokenizer(String tokenizerJsonPath) throws IOException {
            tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(tokenizerJsonPath));
        }
        public void encode(String inputText) {
            encoding = tokenizer.encode(inputText);
        }
        public long[] getIds() {
            return encoding.getIds();
        }
        public long[] getAttentionMask() {
            return encoding.getAttentionMask();
        }
        public String[] getTokens() {
            return encoding.getTokens();
        }
    }

    public static int findMaxIndex(float[] arr) {
        int maxIndex = 0;
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] > arr[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public static void post(int class_, String token, String persons_,
                            String locations_, String organizations_, String misc_) {
        /*
        seperates tokens into arrays according to class ids
        below is the relation from class id to the label
        "id2label": {
        "0": "B-LOC",
        "1": "B-MISC",
        "2": "B-ORG",
        "3": "I-LOC",
        "4": "I-MISC",
        "5": "I-ORG",
        "6": "I-PER",
        "7": "O"
        * */
        if (class_ == 6) persons = persons_ + token;
        else if (class_ == 2 || class_ == 5) organizations = organizations_ + token;
        else if (class_ == 3 || class_ == 0) locations = locations_ + token;
        else if (class_ == 1 || class_ == 4) misc = misc_ + token;
    }

    public static void main(String[] args) throws Exception{

        try {


            String model_path =  new File("").getCanonicalPath()+
                    "\\model\\deeplearning\\english_segment_ner\\onnx\\model.onnx";

            String token_path = new File("").getCanonicalPath()+
                    "\\model\\deeplearning\\english_segment_ner\\tokenizer.json";

            // 模型加载
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            OrtSession session = env.createSession(model_path, new OrtSession.SessionOptions());

            // 输入输出信息
            System.out.println("输入信息:");
            System.out.println(session.getInputNames());
            System.out.println(session.getInputInfo());
            System.out.println("输出信息:");
            System.out.println(session.getOutputNames());
            System.out.println(session.getOutputInfo());


            // Token 编码
            tokenizer = new Tokenizer(token_path);
            tokenizer.encode("Ahwar wants to work at Google in london. EU rejected German call to boycott British lamb.");


            // 输入
            inputIds = tokenizer.getIds(); // get Input Ids
            inputAttentionMask = tokenizer.getAttentionMask(); // get Attention mask


            // 加纬度
            // from [input_ids] to [[input_ids]]
            long[][] newInputIds = new long[1][inputIds.length];
            System.arraycopy(inputIds, 0, newInputIds[0], 0, inputIds.length);

            // 加纬度
            // from [attention_mask] to [[attention_mask]]
            long[][] newAttentionMask = new long[1][inputAttentionMask.length];
            System.arraycopy(inputAttentionMask, 0, newAttentionMask[0], 0, inputAttentionMask.length);

            // 张量输入
            OnnxTensor idsTensor = OnnxTensor.createTensor(env, newInputIds);
            OnnxTensor maskTensor = OnnxTensor.createTensor(env, newAttentionMask);

            // 推理
            Map<String,OnnxTensor> model_inputs = new HashMap<>();
            model_inputs.put("input_ids",idsTensor);
            model_inputs.put("attention_mask",maskTensor);
            OrtSession.Result result = session.run(model_inputs);

            // 后处理
            float[][][] logits = (float[][][]) result.get(0).getValue();
            String[] tokens = tokenizer.getTokens(); // tokenize the text

            for (int i = 0; i < logits[0].length; i++) {
                try {
                    // id转文字
                    post(findMaxIndex(logits[0][i]), tokens[i], persons, locations, organizations, misc);
                } catch (Exception exception) {
                    exception.printStackTrace();
                }
            }

            // 输出
            String tokensSpecialChar = String.valueOf(tokenizer.getTokens()[1].charAt(0)); // word seperators in tokens
            System.out.println("All persons in the text: " + Arrays.toString(persons.split(tokensSpecialChar)));
            System.out.println("All Organizations in the text: " + Arrays.toString(organizations.split(tokensSpecialChar)));
            System.out.println("All Locations in the text: " + Arrays.toString(locations.split(tokensSpecialChar)));
            System.out.println("All Miscellanous entities in the text: " + Arrays.toString(misc.split(tokensSpecialChar)));

        } catch (OrtException e) {
            e.printStackTrace();
        }

    }


}
