package tool.deeplearning;


import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.audio.Audio;
import ai.djl.modality.audio.AudioFactory;
import ai.djl.modality.audio.translator.SpeechRecognitionTranslatorFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import java.io.File;
import java.io.IOException;

import javax.sound.sampled.UnsupportedAudioFileException;

/**
*   @desc : 语音识别（英文） wav2vec2 ,语言转文字 , djl 推理
 *
 *      使用的模型:
 *      https://resources.djl.ai/test-models/pytorch/wav2vec2.zip
 *
*   @auth : tyf
*   @date : 2022-06-13  20:37:41
*/
public class wav2vec2_speech2text_englinsh_djl {



    public static class SpeechRecognition {

        String model;
        private SpeechRecognition(String model) {
            this.model = model;
        }

        public String predict(String audioPath) throws IOException, ModelException, TranslateException {
            // Load model.
            // Wav2Vec2 model is a speech model that accepts a float array corresponding to the raw
            // waveform of the speech signal.
            Criteria<Audio, String> criteria =
                    Criteria.builder()
                            .setTypes(Audio.class, String.class)
                            .optModelPath(new File(model).toPath())
                            .optTranslatorFactory(new SpeechRecognitionTranslatorFactory())
                            .optModelName("wav2vec2.ptl")
                            .optEngine("PyTorch")
                            .build();

            // Read in audio file
            Audio audio = AudioFactory.newInstance().fromUrl(new File(audioPath).toURL());
            try (ZooModel<Audio, String> model = criteria.loadModel();
                 Predictor<Audio, String> predictor = model.newPredictor()) {
                return predictor.predict(audio);
            }
        }
    }


    public static void main(String[] args) throws UnsupportedAudioFileException, IOException, TranslateException, ModelException {


        // 模型
        String path = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\wav2vec2_speech2text_englinsh_djl\\wav2vec2.ptl";

        // 测试音频
        String audio = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\wav2vec2_speech2text_englinsh_djl\\speech.wav";


        // 推理
        SpeechRecognition recognition = new SpeechRecognition(path);
        String res = recognition.predict(audio).toLowerCase();

        System.out.println("结果:");
        System.out.println(res);

    }


}
