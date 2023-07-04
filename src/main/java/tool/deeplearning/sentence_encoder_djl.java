package tool.deeplearning;


import ai.djl.Device;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ai.djl.Model;
import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.bert.BertFullTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.StackBatchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.List;

/**
*   @desc : 多语言文本编码 , 文本向量相似度计算（es） ,用于文本搜索等等 djl 推理
 *
 *          参考 Elasticsearch 官方用例：
 *          https://blog.csdn.net/UbuntuTouch/article/details/128557942
 *
 *          模型使用：
 *          使用在 Models - Hugging Face 发布的 sentence-transformers/distiluse-base-multilingual-cased-v1 模型。
 *          根据介绍，这个 model 支持 Arabic, Chinese, Dutch, English, French, German,
 *          Italian, Korean, Polish, Portuguese, Russian, Spanish, Turkish 等语言。
 *
*   @auth : tyf
*   @date : 2022-06-16  10:00:50
*/
public class sentence_encoder_djl {


    // 模型
    public static class SentenceEncoder {

        private static final Logger logger = LoggerFactory.getLogger(SentenceEncoder.class);

        String modelPath;
        public SentenceEncoder(String modelPath) {
            this.modelPath = modelPath;
        }

        public Criteria<String, float[]> criteria() {

            Criteria<String, float[]> criteria =
                    Criteria.builder()
                            .setTypes(String.class, float[].class)
                            .optModelPath(new File(modelPath).toPath())
                            .optTranslator(new SentenceTransTranslator())
                            .optEngine("PyTorch") // Use PyTorch engine
                            .optDevice(Device.cpu())
                            .optProgress(new ProgressBar())
                            .build();

            return criteria;
        }
    }


    // 模型处理
    public static class SentenceTransTranslator implements Translator<String, float[]> {

        //  private Vocabulary vocabulary;
        //  private BertTokenizer tokenizer; //不切分subword
        private final int maxSequenceLength = 128;
        private DefaultVocabulary vocabulary;
        private BertFullTokenizer tokenizer;

        @Override
        public Batchifier getBatchifier() {
            return new StackBatchifier();
        }

        @Override
        public void prepare(TranslatorContext ctx) throws IOException {
            Model model = ctx.getModel();
            URL url = model.getArtifact("vocab.txt");
            vocabulary = DefaultVocabulary.builder()
                            .optMinFrequency(1)
                            .addFromTextFile(url)
                            .optUnknownToken("[UNK]")
                            .build();
            //    tokenizer = new BertTokenizer();
            tokenizer = new BertFullTokenizer(vocabulary, false);
        }

        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            NDArray array = null;
            // 下面的排序非固定，每次运行顺序可能会变
            //  input_ids
            //  token_type_ids
            //  attention_mask
            //  token_embeddings: (13, 384) cpu() float32
            //  cls_token_embeddings: (384) cpu() float32
            //  sentence_embedding: (384) cpu() float32
            for (NDArray ndArray : list) {
                String name = ndArray.getName();
                if (name.equals("sentence_embedding")) {
                    array = ndArray;
                    break;
                }
            }
            float[] result = array.toFloatArray();
            return result;
        }

        @Override
        public NDList processInput(TranslatorContext ctx, String input) {
            List<String> tokens = tokenizer.tokenize(input);
            if (tokens.size() > maxSequenceLength - 2) {
                tokens = tokens.subList(0, maxSequenceLength - 2);
            }
            long[] indices = tokens.stream().mapToLong(vocabulary::getIndex).toArray();
            long[] input_ids = new long[tokens.size() + 2];
            input_ids[0] = vocabulary.getIndex("[CLS]");
            input_ids[input_ids.length - 1] = vocabulary.getIndex("[SEP]");

            System.arraycopy(indices, 0, input_ids, 1, indices.length);

            long[] token_type_ids = new long[input_ids.length];
            Arrays.fill(token_type_ids, 0);
            long[] attention_mask = new long[input_ids.length];
            Arrays.fill(attention_mask, 1);

            NDManager manager = ctx.getNDManager();
            //        input_features = {'input_ids': input_ids, 'token_type_ids': input_type_ids,
            // 'attention_mask': input_mask}
            //        input_ids
            //        tensor([[  101 [CLS],  2023 this,  7705 framework, 19421 generates,  7861 em,
            //        8270 ##bed,  4667 ##ding,  2015 ##s,  2005 for,  2169 each, 7953 input,  6251
            // sentence,   102 [SEP]]])
            //        token_type_ids
            //        tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
            //        attention_mask
            //        tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

            //    long[] input_ids =
            //        new long[] {101, 2023, 7705, 19421, 7861, 8270, 4667, 2015, 2005, 2169, 7953, 6251,
            // 102};
            NDArray indicesArray = manager.create(input_ids);
            indicesArray.setName("input.input_ids");

            //    long[] token_type_ids = new long[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            NDArray tokenIdsArray = manager.create(token_type_ids);
            tokenIdsArray.setName("input.token_type_ids");

            //    long[] attention_mask = new long[] {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
            NDArray attentionMaskArray = manager.create(attention_mask);
            attentionMaskArray.setName("input.attention_mask");
            return new NDList(indicesArray, attentionMaskArray);
        }
    }

    // 计算向量相似度
    public static class FeatureComparison {
        private FeatureComparison() {}

        public static float cosineSim(float[] feature1, float[] feature2) {
            float ret = 0.0f;
            float mod1 = 0.0f;
            float mod2 = 0.0f;
            int length = feature1.length;
            for (int i = 0; i < length; ++i) {
                ret += feature1[i] * feature2[i];
                mod1 += feature1[i] * feature1[i];
                mod2 += feature2[i] * feature2[i];
            }
            //    dot(x, y) / (np.sqrt(dot(x, x)) * np.sqrt(dot(y, y))))
            return (float) (ret / Math.sqrt(mod1) / Math.sqrt(mod2));
        }

        public static float dot(float[] feature1, float[] feature2) {
            float ret = 0.0f;
            int length = feature1.length;
            // dot(x, y)
            for (int i = 0; i < length; ++i) {
                ret += feature1[i] * feature2[i];
            }

            return ret;
        }
    }


    public static void main(String[] args) throws Exception{


        // 模型,里面包含 vocab.txt 自动解压和加载
        String modelPath = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\sentence_encoder_djl\\distiluse-base-multilingual-cased-v1.zip";

        // 文本
        String text1 = "所借的钱是否可以提现？";
        String text2 = "该笔借款可以提现吗！";
        String text3 = "为什么我申请额度输入密码就一直是那个页面";


        // 加载模型
        Criteria<String, float[]> criteria =new SentenceEncoder(modelPath).criteria();
        ZooModel<String, float[]> model = criteria.loadModel();
        Predictor<String, float[]> predictor = model.newPredictor();


        // 预测 512 纬向量
        float[] tenser1 = predictor.predict(text1);
        float[] tenser2 = predictor.predict(text2);
        float[] tenser3 = predictor.predict(text3);

        // 计算相似度
        float sim1 = FeatureComparison.cosineSim(tenser1,tenser2);
        float sim2 = FeatureComparison.cosineSim(tenser2,tenser3);
        float sim3 = FeatureComparison.cosineSim(tenser1,tenser3);

        System.out.println(text1+" => "+text2+" , 相似度:"+sim1);
        System.out.println(text2+" => "+text3+" , 相似度:"+sim2);
        System.out.println(text1+" => "+text3+" , 相似度:"+sim3);



    }


}
