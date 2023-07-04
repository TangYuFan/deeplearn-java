package tool.deeplearning;


import ai.djl.repository.zoo.Criteria;
import ai.djl.training.util.ProgressBar;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.net.MalformedURLException;
import java.nio.file.Paths;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;
import org.apache.commons.lang3.StringUtils;
import java.io.IOException;
import java.io.InputStream;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import java.util.Arrays;
import ai.djl.ndarray.NDArrays;
import ai.djl.paddlepaddle.engine.PpNDArray;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 *   @desc : 翻译 中文=>英语  中文分词+翻译, djl 推理
 *
 *          pp开源模型:
 *          https://www.paddlepaddle.org.cn/hubdetail?name=transformer_zh-en&en_category=MachineTranslation
 *
 *   @auth : tyf
 *   @date : 2022-06-13  19:44:36
 */
public class chinese2english_translation_djl {


    public static class Lac {

        private static final Logger logger = LoggerFactory.getLogger(Lac.class);

        String modelPath;
        public Lac(String modelPath) {
            this.modelPath = modelPath;
        }

        public Criteria<String, String[][]> criteria() {
            Criteria<String, String[][]> criteria =
                    Criteria.builder()
                            .setTypes(String.class, String[][].class)
                            .optModelPath(new File(modelPath).toPath())
                            .optTranslator(new LacTranslator())
                            .optProgress(new ProgressBar())
                            .optEngine("PaddlePaddle") // Use PaddlePaddle engine
                            .build();

            return criteria;
        }
    }


    public static class LacTranslator implements Translator<String, String[][]> {
        LacTranslator() {}

        private Map<String, String> word2id_dict = new HashMap<String, String>();
        private Map<String, String> id2word_dict = new HashMap<String, String>();
        private Map<String, String> label2id_dict = new HashMap<String, String>();
        private Map<String, String> id2label_dict = new HashMap<String, String>();
        private Map<String, String> word_replace_dict = new HashMap<String, String>();
        private String oov_id;
        private String input;

        @Override
        public void prepare(TranslatorContext ctx) throws IOException {
            Model model = ctx.getModel();
            try (InputStream is = model.getArtifact("lac/word.dic").openStream()) {
                List<String> words = Utils.readLines(is, true);
                words.stream()
                        .filter(word -> (word != null && word != ""))
                        .forEach(
                                word -> {
                                    String[] ws = word.split("	");
                                    if (ws.length == 1) {
                                        word2id_dict.put("", ws[0]); // 文字是key,id是value - Text is the key, ID is the value.
                                        id2word_dict.put(ws[0], "");
                                    } else {
                                        word2id_dict.put(ws[1], ws[0]); // 文字是key,id是value - Text is the key, ID is the value.
                                        id2word_dict.put(ws[0], ws[1]);
                                    }
                                });
            }
            try (InputStream is = model.getArtifact("lac/tag.dic").openStream()) {
                List<String> words = Utils.readLines(is, true);
                words.stream()
                        .filter(word -> (word != null && word != ""))
                        .forEach(
                                word -> {
                                    String[] ws = word.split("	");
                                    label2id_dict.put(ws[1], ws[0]); // 文字是key,id是value - Text is the key, ID is the value.
                                    id2label_dict.put(ws[0], ws[1]);
                                });
            }
            try (InputStream is = model.getArtifact("lac/q2b.dic").openStream()) {
                List<String> words = Utils.readLines(is, true);
                words.stream()
                        .forEach(
                                word -> {
                                    if (StringUtils.isBlank(word)) {
                                        word_replace_dict.put("　", " "); // 文字是key,id是value - Text is the key, ID is the value.
                                    } else {
                                        String[] ws = word.split("	");
                                        if (ws.length == 1) {
                                            if (ws[0] != null) {
                                                word_replace_dict.put(ws[0], ""); // 文字是key,id是value - Text is the key, ID is the value.
                                            } else {
                                                word_replace_dict.put("", ws[1]); // 文字是key,id是value - Text is the key, ID is the value.
                                            }
                                        } else {
                                            word_replace_dict.put(ws[0], ws[1]); // 文字是key,id是value - Text is the key, ID is the value.
                                        }
                                    }
                                });
            }
            oov_id = word2id_dict.get("OOV");
        }

        @Override
        public NDList processInput(TranslatorContext ctx, String input) {
            this.input = input;

            NDManager manager = ctx.getNDManager();

            NDList inputList = new NDList();
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
        public String[][] processOutput(TranslatorContext ctx, NDList list) {
            String[] s = input.replace(" ", "").split("");

            List<String> sent_out = new ArrayList<>();
            List<String> tags_out = new ArrayList<>();

            long[] array = list.get(0).toLongArray();
            List<String> tags = new ArrayList<>();

            // ['今天是个好日子']
            // [[1209]
            // [ 113]
            // [1178]
            // [3186]
            // [ 517]
            // [ 418]
            // [  90]]

            // lod:
            // [[0, 7]]

            // output:
            // [[54]
            // [55]
            // [38]
            // [28]
            // [14]
            // [15]
            // [15]]

            // ['TIME-B', 'TIME-I', 'v-B', 'q-B', 'n-B', 'n-I', 'n-I']

            for (int i = 0; i < array.length; i++) {
                tags.add(id2label_dict.get(String.valueOf(array[i])));
            }
            for (int i = 0; i < tags.size(); i++) {
                String tag = tags.get(i);
                if (sent_out.size() == 0 || tag.endsWith("B") || tag.endsWith("S")) {
                    sent_out.add(s[i]);
                    tags_out.add(tag.substring(0, tag.length() - 2));
                    continue;
                }

                //      今
                //              ['今']
                //      是
                //              ['今天', '是']
                //      个
                //              ['今天', '是', '个']
                //      好
                //              ['今天', '是', '个', '好']
                sent_out.set(sent_out.size() - 1, sent_out.get(sent_out.size() - 1) + s[i]);
                // ['TIME-B', 'TIME-I', 'v-B', 'q-B', 'n-B', 'n-I', 'n-I']
                // ['TIME', 'TIME', 'v', 'q', 'n', 'n', 'n']
                tags_out.set(tags_out.size() - 1, tag.substring(0, tag.length() - 2));
            }
            String[][] result = new String[2][sent_out.size()];

            result[0] = (String[]) sent_out.toArray(new String[sent_out.size()]);
            result[1] = (String[]) tags_out.toArray(new String[tags_out.size()]);

            return result;
        }

        private List<Long> tokenizeSingleString(NDManager manager, String input, List<Long> lod) {
            List<Long> word_ids = new ArrayList<>();
            String[] s = input.replace(" ", "").split("");
            for (String word : s) {
                String newword = word_replace_dict.get(word);
                word = StringUtils.isBlank(newword) ? word : newword;
                String word_id = word2id_dict.get(word);
                word_ids.add(Long.valueOf(StringUtils.isBlank(word_id) ? oov_id : word_id));
            }
            lod.add((long) word_ids.size());
            return word_ids;
        }

        private NDArray stackInputs(List<NDList> tokenizedInputs, int index, String inputName) {
            NDArray stacked =
                    NDArrays.stack(
                            tokenizedInputs.stream()
                                    .map(list -> list.get(index).expandDims(0))
                                    .collect(Collectors.toCollection(NDList::new)));
            stacked.setName(inputName);
            return stacked;
        }

        private NDArray tokenizeSingle(NDManager manager, String[] inputs, List<Integer> lod) {
            List<Integer> word_ids = new ArrayList<>();
            for (int i = 0; i < inputs.length; i++) {
                String input = inputs[i];
                String[] s = input.replace(" ", "").split("");
                for (String word : s) {
                    String newword = word_replace_dict.get(word);
                    word = StringUtils.isBlank(newword) ? word : newword;
                    String word_id = word2id_dict.get(word);
                    word_ids.add(Integer.valueOf(StringUtils.isBlank(word_id) ? oov_id : word_id));
                }
                lod.add(word_ids.size() + lod.get(i));
            }
            return manager.create(word_ids.stream().mapToLong(l -> Long.valueOf(l)).toArray());
        }

        @Override
        public Batchifier getBatchifier() {
            return null;
        }
    }


    public static class Translation {

        private static final Logger logger = LoggerFactory.getLogger(Translation.class);

        String modelPath;
        public Translation(String modelPath) {
            this.modelPath = modelPath;
        }
        public Criteria<String[], String[]> criteria() throws MalformedURLException {

            Criteria<String[], String[]> criteria =
                    Criteria.builder()
                            .setTypes(String[].class, String[].class)
                            .optModelPath(new File(modelPath).toPath())
                            .optTranslator(new TranslationTranslator())
                            .optEngine("PaddlePaddle") // Use PyTorch engine
                            .optModelName("inference")

                            .optProgress(new ProgressBar())
                            .build();

            return criteria;
        }
    }


    public static class TranslationTranslator implements Translator<String[], String[]> {
        TranslationTranslator() {}

        private Map<String, String> src_word2id_dict = new HashMap<String, String>();
        private Map<String, String> trg_id2word_dict = new HashMap<String, String>();
        private String bos_id = "0";
        private String eos_id = "1";
        private String eos_token = "<e>";
        private String unk_id = "2";
        private String pad_factor = "8";
        private int maxLength = 256;

        @Override
        public void prepare(TranslatorContext ctx) throws IOException {
            Model model = ctx.getModel();
            try (InputStream is = model.getArtifact("assets/vocab.zh").openStream()) {
                List<String> words = Utils.readLines(is, true);
                for (int i = 0; i < words.size(); i++) {
                    src_word2id_dict.put(words.get(i), "" + i); // 文字是key,id是value
                }
            }

            try (InputStream is = model.getArtifact("assets/vocab.en").openStream()) {
                List<String> words = Utils.readLines(is, true);
                for (int i = 0; i < words.size(); i++) {
                    trg_id2word_dict.put("" + i, words.get(i)); // id是key,文字是value
                }
            }
        }

        @Override
        public NDList processInput(TranslatorContext ctx, String[] input) {
            NDManager manager = ctx.getNDManager();
            List<Long> list = tokenizeSingleString(manager, input);
            long[] array = list.stream().mapToLong(Long::valueOf).toArray();
            NDArray ndArray = null;
            if (array.length > maxLength) {
                long[] newArr = (long[]) Arrays.copyOf(array, maxLength);
                ndArray = manager.create(newArr, new Shape(1, maxLength));
            } else {
                //    array = new long[] {6336, 914, 1652, 2051, 2, 44, 1};
                ndArray = manager.create(array, new Shape(1, array.length));
            }
            return new NDList(ndArray);
        }

        @Override
        public String[] processOutput(TranslatorContext ctx, NDList list) {
            // index = 0 negative
            // index = 1 positive
            // [0.05931241 0.9406876 ]
            NDArray ndArray = list.get(0);
            //    ndArray = ndArray.transpose(0, 2, 1);
            //    ndArray = ndArray.squeeze(0);
            long[] array = ndArray.toLongArray();

            Shape shape = ndArray.getShape();
            int rows = (int) shape.get(2);
            int cols = (int) shape.get(1);
            long[][] ids = new long[rows][cols];

            for (int col = 0; col < cols; col++) {
                for (int row = 0; row < rows; row++) {
                    ids[row][col] = array[col * rows + row];
                }
            }

            String[][] wordsArray = new String[rows][cols];

            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    wordsArray[row][col] = trg_id2word_dict.get("" + ids[row][col]);
                }
            }

            String[] result = new String[rows];
            for (int row = 0; row < rows; row++) {
                result[row] = "";
                for (int col = 0; col < cols; col++) {
                    if (wordsArray[row][col].equals(eos_token)) continue;
                    result[row] = result[row] + " " + wordsArray[row][col];
                }
            }

            return result;
        }

        private List<Long> tokenizeSingleString(NDManager manager, String[] input) {
            List<Long> word_ids = new ArrayList<>();
            for (String word : input) {
                String word_id = src_word2id_dict.get(word);
                word_ids.add(Long.valueOf(StringUtils.isBlank(word_id) ? unk_id : word_id));
            }
            word_ids.add(Long.valueOf(eos_id));
            return word_ids;
        }

        @Override
        public Batchifier getBatchifier() {
            return null;
        }
    }


    public static void main(String[] args) throws IOException, TranslateException, ModelException {


        // 分词模型  word.dic tag.dic  q2b.dic  会自动加压并加载
        String model1 = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\chinese2english_translation_djl\\lac.zip";


        // 翻译模型  vocab.en vocab.ch 会自动加压并加载
        String model2 = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\chinese2english_translation_djl\\translation_zh_en.zip";




        // 分词 - Split words
        Lac lac = new Lac(model1);
        Criteria<String, String[][]> lacCriteria = lac.criteria();
        // 翻译 - Translation
        Translation senta = new Translation(model2);
        Criteria<String[], String[]> SentaCriteria = senta.criteria();

        try (ZooModel<String, String[][]> lacModel = lacCriteria.loadModel();
             Predictor<String, String[][]> lacPredictor = lacModel.newPredictor();
             ZooModel<String[], String[]> sentaModel = SentaCriteria.loadModel();
             Predictor<String[], String[]> sentaPredictor = sentaModel.newPredictor()) {

            String input = "今天天气怎么样？";
            System.out.println("输入句子: " + input);

            String[][] lacResult = lacPredictor.predict(input);
            // 分词 - Split words
            System.out.println("Words : " + Arrays.toString(lacResult[0]));
            // 词性 - tag
            System.out.println("Tags : " + Arrays.toString(lacResult[1]));

            // 翻译结果 - Translation result
            String[] translationResult = sentaPredictor.predict(lacResult[0]);
            for (int i = 0; i < translationResult.length; i++) {
                System.out.println("T" + i + ": " + translationResult[i]);
            }
//            logger.info(Arrays.toString(translationResult));
        }
    }

}
