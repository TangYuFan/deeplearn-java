package tool.deeplearning;


import ai.djl.repository.zoo.Criteria;
import ai.djl.training.util.ProgressBar;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.file.Paths;
import ai.djl.Model;
import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.bert.BertFullTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.paddlepaddle.engine.PpNDArray;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;
import org.apache.commons.lang3.StringUtils;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.util.Arrays;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
/**
*   @desc : 中文文本色情词汇检测 , pp开源模型, djl 推理
 *
 *
 *
*   @auth : tyf
*   @date : 2022-06-16  10:22:47
*/
public class pp_text_porn_rec_djl {


    // 模型加载
    public static class ReviewDetection {

        private static final Logger logger = LoggerFactory.getLogger(ReviewDetection.class);

        String modelPath;
        public ReviewDetection(String modelPath) {
            this.modelPath = modelPath;
        }

        public Criteria<String, float[]> criteria() {

            Criteria<String, float[]> criteria =
                    Criteria.builder()
                            .setTypes(String.class, float[].class)
                            .optModelPath(new File(modelPath).toPath())
                            .optTranslator(new ReviewTranslator())
                            .optEngine("PaddlePaddle") // Use PyTorch engine

                            .optProgress(new ProgressBar())
                            .build();

            return criteria;
        }
    }


    public static class ReviewTranslator implements Translator<String, float[]> {
        ReviewTranslator() {}

        private DefaultVocabulary vocabulary;
        private BertFullTokenizer tokenizer;
        private Map<String, String> word2id_dict = new HashMap<String, String>();
        private String unk_id = "";
        private String pad_id = "";
        private int maxLength = 256;

        @Override
        public void prepare(TranslatorContext ctx) throws IOException {
            Model model = ctx.getModel();
            try (InputStream is = model.getArtifact("assets/word_dict.txt").openStream()) {
                List<String> words = Utils.readLines(is, true);
                for (int i = 0; i < words.size(); i++) {
                    word2id_dict.put(words.get(i), "" + i); // 文字是key,id是value - Text is the key, ID is the value.
                }
            }
            unk_id = "" + word2id_dict.get("<UNK>"); // 文字是key,id是value - Text is the key, ID is the value.
            pad_id = "" + word2id_dict.get("<PAD>"); // 文字是key,id是value - Text is the key, ID is the value.

            vocabulary =
                    DefaultVocabulary.builder()
                            .optMinFrequency(1)
                            .addFromTextFile(model.getArtifact("assets/vocab.txt"))
                            // .addFromTextFile(vocabPath)
                            .optUnknownToken("<UNK>")
                            .build();
            tokenizer = new BertFullTokenizer(vocabulary, false);
        }

        @Override
        public NDList processInput(TranslatorContext ctx, String input) {

            NDManager manager = ctx.getNDManager();
            List<Long> lodList = new ArrayList<>(0);
            lodList.add(new Long(0));
            List<Long> sh = tokenizeSingleString(manager, input, lodList);

            int size = Long.valueOf(lodList.get(lodList.size() - 1)).intValue();
            long[] array = new long[size];
            for (int i = 0; i < size; i++) {
                if (sh.size() > i) {
                    array[i] = sh.get(i);
                } else {
                    array[i] = 0;
                }
            }
            NDArray ndArray = manager.create(array, new Shape(lodList.get(lodList.size() - 1), 1));

            ndArray.setName("words");
            long[][] lod = new long[1][2];
            lod[0][0] = 0;
            lod[0][1] = lodList.get(lodList.size() - 1);
            ((PpNDArray) ndArray).setLoD(lod);
            return new NDList(ndArray);
        }

        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            // index = 0 negative
            // index = 1 positive
            // [0.05931241 0.9406876 ]
            float[] result = list.get(0).toFloatArray();
            return result;
        }

        private List<Long> tokenizeSingleString(NDManager manager, String input, List<Long> lod) {
            List<Long> word_ids = new ArrayList<>();
            List<String> list = tokenizer.tokenize(input);
            for (String word : list) {
                word = word.replace("#", "");
                String word_id = word2id_dict.get(word);
                word_ids.add(Long.valueOf(StringUtils.isBlank(word_id) ? unk_id : word_id));
            }
            if (word_ids.size() < maxLength) {
                int diff = maxLength - word_ids.size();
                for (int i = 0; i < diff; i++) {
                    word_ids.add(Long.parseLong(pad_id));
                }
            }
            lod.add((long) word_ids.size());
            return word_ids;
        }

        @Override
        public Batchifier getBatchifier() {
            return null;
        }
    }



    public static void main(String[] args) throws IOException, TranslateException, ModelException {


        // 模型 assets/word_dict.txt 和 assets/vocab.txt 会自动解压
        String modelPath = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\pp_text_porn_rec_djl\\review_detection_lstm.zip";


        // 待检测文本
        String input = "你在干什么啊";


        ReviewDetection reviewDetection = new ReviewDetection(modelPath);
        Criteria<String, float[]> SentaCriteria = reviewDetection.criteria();

        try (ZooModel<String, float[]> model = SentaCriteria.loadModel();
                Predictor<String, float[]>predictor = model.newPredictor()) {



            // 文本检查 Text check
            float[] result = predictor.predict(input);
            System.out.println("非色情内文概率 : " + result[0]);
            System.out.println("色情内容概率: " + result[1]);


        }
    }


}
