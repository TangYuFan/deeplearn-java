package tool.deeplearning;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.audio.Audio;
import ai.djl.modality.audio.AudioFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.Model;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Pair;
import ai.djl.audio.processor.AudioProcessor;
import ai.djl.audio.processor.LogMelSpectrogram;
import ai.djl.audio.processor.PadOrTrim;
import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.JsonUtils;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.io.Serializable;
import java.lang.reflect.Type;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import com.google.gson.reflect.TypeToken;
import org.bytedeco.ffmpeg.global.avutil;

import java.io.Reader;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
*   @desc : whisper 语音识别（英文）,转文本（openai在9月发布的一个开源语音识别翻译模型）
 *
 *            使用的模型:
 *             https://resources.djl.ai/demo/pytorch/whisper/whisper_en.zip
 *
 *
*   @auth : tyf
*   @date : 2022-06-13  11:48:57
*/
public class whisper_speech2text_englinsh_djl {



    public static class WhisperTranslator implements NoBatchifyTranslator<Audio, String> {

        private List<AudioProcessor> processors;
        private Vocabulary vocabulary;

        /** Constructs a new instance of {@code WhisperTranslator}. */
        public WhisperTranslator() {
            processors = new ArrayList<>();
        }

        /** {@inheritDoc} */
        @Override
        public void prepare(TranslatorContext ctx) throws IOException {
            Path path = ctx.getModel().getModelPath();
            Path melFile = path.resolve("mel_80_filters.npz");

            processors.add(new PadOrTrim(480000));
            // Use model's NDManager
            NDManager modelManager = ctx.getModel().getNDManager();
            processors.add(LogMelSpectrogram.newInstance(melFile, 80, modelManager));

            Map<String, Integer> vocab;
            Map<String, Integer> added;
            Type type = new TypeToken<Map<String, Integer>>() {}.getType();
            try (Reader reader = Files.newBufferedReader(path.resolve("vocab.json"))) {
                vocab = JsonUtils.GSON.fromJson(reader, type);
            }
            try (Reader reader = Files.newBufferedReader(path.resolve("added_tokens.json"))) {
                added = JsonUtils.GSON.fromJson(reader, type);
            }
            String[] result = new String[vocab.size() + added.size()];
            vocab.forEach((key, value) -> result[value] = key);
            added.forEach((key, value) -> result[value] = key);
            vocabulary = new DefaultVocabulary(Arrays.asList(result));
        }

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, Audio input) throws Exception {
            NDArray samples = ctx.getNDManager().create(input.getData());
            for (AudioProcessor processor : processors) {
                samples = processor.extractFeatures(samples.getManager(), samples);
            }
            samples = samples.expandDims(0);
            NDArray placeholder = ctx.getNDManager().create("");
            placeholder.setName("module_method:generate");
            return new NDList(samples, placeholder);
        }

        /** {@inheritDoc} */
        @Override
        public String processOutput(TranslatorContext ctx, NDList list) throws Exception {
            NDArray result = list.singletonOrThrow();
            List<String> sentence = new ArrayList<>();
            for (long ele : result.toLongArray()) {
                sentence.add(vocabulary.getToken(ele));
                if ("<|endoftext|>".equals(vocabulary.getToken(ele))) {
                    break;
                }
            }
            String output = String.join(" ", sentence);
            return output.replaceAll("[^a-zA-Z0-9<|> ,.!]", "");
        }
    }

    public static class WhisperTranslatorFactory implements TranslatorFactory, Serializable {

        private static final long serialVersionUID = 1L;

        private static final Set<Pair<Type, Type>> SUPPORTED_TYPES = new HashSet<>();

        static {
            SUPPORTED_TYPES.add(new Pair<>(Audio.class, String.class));
        }

        /** {@inheritDoc} */
        @Override
        public Set<Pair<Type, Type>> getSupportedTypes() {
            return SUPPORTED_TYPES;
        }

        /** {@inheritDoc} */
        @Override
        @SuppressWarnings("unchecked")
        public <I, O> Translator<I, O> newInstance(
                Class<I> input, Class<O> output, Model model, Map<String, ?> arguments) {
            if (input == Audio.class && output == String.class) {
                return (Translator<I, O>) new WhisperTranslator();
            }
            throw new IllegalArgumentException("Unsupported input/output types.");
        }
    }

    public static class WhisperModel implements AutoCloseable {

        ZooModel<Audio, String> whisperModel;

        public WhisperModel(String model) throws ModelException, IOException {
            Criteria<Audio, String> criteria = Criteria.builder()
                            .setTypes(Audio.class, String.class)
                            .optModelPath(new File(model).toPath())
                            .optEngine("PyTorch")
                            .optTranslatorFactory(new WhisperTranslatorFactory())
                            .build();
            whisperModel = criteria.loadModel();
        }

        public String speechToText(Audio speech) throws Exception {
            try (Predictor<Audio, String> predictor = whisperModel.newPredictor()) {
                return predictor.predict(speech);
            }
        }

        public String speechToText(String path) throws Exception {
            Path file = new File(path).toPath();
            Audio audio = AudioFactory.newInstance()
                            .setChannels(1)
                            .setSampleRate(16000)
                            .setSampleFormat(avutil.AV_SAMPLE_FMT_S16)
                            .fromFile(file);
            return speechToText(audio);
        }

        /** {@inheritDoc} */
        @Override
        public void close() {
            whisperModel.close();
        }
    }



    public static void main(String[] args) throws ModelException, IOException, TranslateException, InterruptedException {


        // 模型
        String path = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\whisper_speech2text_englinsh_djl\\whisper_en.pt";

        // 测试音频
        //        DownloadUtils.download("https://resources.djl.ai/audios/jfk.flac", "C:\\Users\\tyf\\Desktop\\xxx\\jfk.flac");
        String audio = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\whisper_speech2text_englinsh_djl\\jfk.flac";


        System.setProperty("ai.djl.pytorch.graph_optimizer", "false");

        try {
            // 推理
            WhisperModel model = new WhisperModel(path);
            String res = model.speechToText(audio);
            System.out.println("翻译结果：");
            System.out.println(res);
        }
        catch (Exception e){
            e.printStackTrace();
        }

        System.clearProperty("ai.djl.pytorch.graph_optimizer");


    }


}
