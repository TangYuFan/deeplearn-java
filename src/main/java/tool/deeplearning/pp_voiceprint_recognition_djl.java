package tool.deeplearning;


import ai.djl.Device;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.repository.zoo.Criteria;
import ai.djl.training.util.ProgressBar;

import java.io.File;
import java.nio.file.Paths;
import com.jlibrosa.audio.JLibrosa;
import com.jlibrosa.audio.exception.FileFormatNotSupportedException;
import com.jlibrosa.audio.wavFile.WavFileException;
import org.apache.commons.math3.complex.Complex;
import java.io.IOException;



/**
 *  @Desc: 声纹识别 , 提取声音特征进行说话人身份匹配 , djl 推理
 *  @Date: 2022-06-15 20:10:45
 *  @auth: TYF
 */
public class pp_voiceprint_recognition_djl {



    public static class Voiceprint {
        String modelPath;
        public Voiceprint(String modelPath) {
            this.modelPath = modelPath;
        }
        public Criteria<float[][], float[]> criteria() {
            Criteria<float[][], float[]> criteria =
                    Criteria.builder()
                            .setTypes(float[][].class, float[].class)
                            .optModelPath(new File(modelPath).toPath())
                            .optTranslator(new VoiceprintTranslator())
                            .optEngine("PaddlePaddle") // Use PaddlePaddle engine
                            .optProgress(new ProgressBar())
                            .build();
            return criteria;
        }

    }



    public static class VoiceprintTranslator implements Translator<float[][], float[]> {

        VoiceprintTranslator() {}

        @Override
        public NDList processInput(TranslatorContext ctx, float[][] mag) {
            NDManager manager = ctx.getNDManager();

            int spec_len = 257;
            NDArray magNDArray = manager.create(mag);

            NDArray spec_mag = magNDArray.get(":, :" + spec_len);

            // 按列计算均值
            // Calculate the mean by column
            NDArray mean = spec_mag.mean(new int[] {0}, true);
            NDArray std = manager.create(JLibrasaEx.std(spec_mag, mean)).reshape(1, spec_len);

            spec_mag = spec_mag.sub(mean).div(std.add(1e-5));
            spec_mag = spec_mag.expandDims(0); // (1,257,201)
            spec_mag = spec_mag.expandDims(0); // (1,1,257,201)

            return new NDList(spec_mag);
        }

        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            NDArray feature = list.singletonOrThrow();

            return feature.toFloatArray();
        }

        @Override
        public Batchifier getBatchifier() {
            return null;
        }
    }



    public static class JLibrasaEx {
        public static float[][] magnitude(NDManager manager, String audioFilePath)
                throws FileFormatNotSupportedException, IOException, WavFileException {
            int defaultAudioDuration = -1; // -1 value implies the method to process complete audio duration

            JLibrosa jLibrosa = new JLibrosa();

            // 读取音频数据
            // Reading audio data
            float audioFeatureValues[] = jLibrosa.loadAndRead(audioFilePath, 16000, defaultAudioDuration);

            float[] reverseArray = new float[audioFeatureValues.length];
            for (int i = 0; i < audioFeatureValues.length; i++) {
                reverseArray[i] = audioFeatureValues[audioFeatureValues.length - i - 1];
            }
            NDArray reverse = manager.create(reverseArray);

            // 数据拼接
            // Data concatenation
            NDArray extended_wav = manager.create(audioFeatureValues).concat(reverse);

            Complex[][] stftComplexValues =
                    jLibrosa.generateSTFTFeatures(extended_wav.toFloatArray(), -1, -1, 512, -1, 160);
            float[][] mag = JLibrasaEx.magnitude(stftComplexValues, 1);
            return mag;
        }

        public static float[][] magnitude(Complex[][] stftComplexValues, float power) {
            int rows = stftComplexValues.length;
            int cols = stftComplexValues[0].length;
            float[][] mag = new float[rows][cols];

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    mag[i][j] = (float) stftComplexValues[i][j].abs();
                    mag[i][j] = (float) Math.pow(mag[i][j], power);
                }
            }

            return mag;
        }

        // 计算全局标准差
        // Calculating global standard deviation
        public static float[] std(NDArray array, NDArray mean) {
            // 按列减去均值
            // Subtracting mean by column
            array = array.sub(mean);

            // 计算全局标准差
            // Calculating global standard deviation
            int cols = (int) array.getShape().get(1);
            float[] stds = new float[cols];
            for (int i = 0; i < cols; i++) {
                NDArray col = array.get(":," + i);
                stds[i] = std(col);
            }
            return stds;
        }

        // 计算全局标准差
        // Calculating global standard deviation
        public static float std(NDArray array) {
            array = array.square();
            float[] doubleResult = array.toFloatArray();
            float std = 0;
            for (int i = 0; i < doubleResult.length; i++) {
                std = std + doubleResult[i];
            }
            std = (float) Math.sqrt(std / doubleResult.length);
            return std;
        }
    }


    // 计算向量相似度
    public static float calculSimilar(float[] feature1, float[] feature2) {

        float ret = 0.0f;
        float mod1 = 0.0f;
        float mod2 = 0.0f;
        int length = feature1.length;
        for (int i = 0; i < length; ++i) {
            ret += feature1[i] * feature2[i];
            mod1 += feature1[i] * feature1[i];
            mod2 += feature2[i] * feature2[i];
        }
        return (float) (ret / Math.sqrt(mod1) / Math.sqrt(mod2));

    }

    public static void main(String[] args) throws Exception{

        String modelPath = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\pp_voiceprint_recognition_djl\\voiceprint.zip";

        // 语音
        String audioFilePath1 = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\pp_voiceprint_recognition_djl\\a_1.wav";
        String audioFilePath2 = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\pp_voiceprint_recognition_djl\\a_2.wav";
        String audioFilePath3 = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\pp_voiceprint_recognition_djl\\b_1.wav";


        NDManager manager = NDManager.newBaseManager(Device.cpu());

        float[][] mag1 = JLibrasaEx.magnitude(manager, audioFilePath1);
        float[][] mag2 = JLibrasaEx.magnitude(manager, audioFilePath2);
        float[][] mag3 = JLibrasaEx.magnitude(manager, audioFilePath3);

        Voiceprint voiceprint = new Voiceprint(modelPath);
        Criteria<float[][], float[]> criteria = voiceprint.criteria();

        try (ZooModel<float[][], float[]> model = criteria.loadModel();
             Predictor<float[][], float[]> predictor = model.newPredictor()) {


            float[] feature1 = predictor.predict(mag1);
            float[] feature2 = predictor.predict(mag2);
            float[] feature3 = predictor.predict(mag3);

            // 计算相似度
            calculSimilar(feature1, feature2);
            calculSimilar(feature1, feature3);

            System.out.println("a_1.wav,a_2.wav 相似度:"+calculSimilar(feature1, feature2));
            System.out.println("a_1.wav,b_1.wav 相似度:"+calculSimilar(feature1, feature3));


        }


    }



}
