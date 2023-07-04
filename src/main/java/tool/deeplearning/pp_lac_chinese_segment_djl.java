package tool.deeplearning;


import ai.djl.Device;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.paddlepaddle.engine.PpNDArray;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;
import org.apache.commons.lang3.StringUtils;

import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.stream.Collectors;


/**
*   @desc : 中文 分词 + 词性标注 + 命名实体识别 。 pp开源模型，djl推理
 *
 *
*   @auth : tyf
*   @date : 2022-06-16  10:51:25
*/
public class pp_lac_chinese_segment_djl {


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
                                        word2id_dict.put("", ws[0]); // 文字是key,id是value
                                        id2word_dict.put(ws[0], "");
                                    } else {
                                        word2id_dict.put(ws[1], ws[0]); // 文字是key,id是value
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
                                    label2id_dict.put(ws[1], ws[0]); // 文字是key,id是value
                                    id2label_dict.put(ws[0], ws[1]);
                                });
            }
            try (InputStream is = model.getArtifact("lac/q2b.dic").openStream()) {
                List<String> words = Utils.readLines(is, true);
                words.stream()
                        .forEach(
                                word -> {
                                    if (StringUtils.isBlank(word)) {
                                        word_replace_dict.put("　", " "); // 文字是key,id是value
                                    } else {
                                        String[] ws = word.split("	");
                                        if (ws.length == 1) {
                                            if (ws[0] != null) {
                                                word_replace_dict.put(ws[0], ""); // 文字是key,id是value
                                            } else {
                                                word_replace_dict.put("", ws[1]); // 文字是key,id是value
                                            }
                                        } else {
                                            word_replace_dict.put(ws[0], ws[1]); // 文字是key,id是value
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


    public static void main(String[] args) throws Exception{

        // 模型  word.dic tag.dic  q2b.dic  会自动加压并加载
        String modelPath = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\pp_lac_chinese_segment_djl\\lac.zip";

        Lac lac = new Lac(modelPath);
        Criteria<String, String[][]> criteria = lac.criteria();


        String input = "今天是个好日子，我们一起去青城山玩吧？他的电话是 13388236293 吗？";

        try (ZooModel<String, String[][]> model = criteria.loadModel();
             Predictor<String, String[][]> predictor = model.newPredictor()) {


            String[][] result = predictor.predict(input);
            System.out.println("Words : " + Arrays.toString(result[0]));
            System.out.println("Tags : " + Arrays.toString(result[1]));


        }


    }



}
