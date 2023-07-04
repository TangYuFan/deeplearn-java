package tool.deeplearning;

import java.io.IOException;

import javax.sound.sampled.AudioInputStream;

import marytts.LocalMaryInterface;
import marytts.exceptions.MaryConfigurationException;
import marytts.exceptions.SynthesisException;
import marytts.util.data.audio.MaryAudioUtils;

/**
 *  @Desc: text2wav 美国英语 女性声音 (隐藏半马尔可夫模型-卡内基梅隆大学提供)
 *  @Date: 2022-06-09 13:38:34
 *  @auth: TYF
 */
public class tts_mary_us_english {


    public static void main(String[] args) throws MaryConfigurationException {

        // 需要转换的英文句子
        String inputText = "A female US English Hidden semi-Markov model voice, built from voice recordings provided by Carnegie Mellon University";

        // 生成wav文件
        String outputFileName = "out.wav";

        // init mary
        LocalMaryInterface mary = null;
        try {
            mary = new LocalMaryInterface();
        } catch (MaryConfigurationException e) {
            System.err.println("Could not initialize MaryTTS interface: " + e.getMessage());
            throw e;
        }

        // synthesize
        AudioInputStream audio = null;
        try {
            audio = mary.generateAudio(inputText);
        } catch (SynthesisException e) {
            System.err.println("Synthesis failed: " + e.getMessage());
            System.exit(1);
        }

        // write to output
        double[] samples = MaryAudioUtils.getSamplesAsDoubleArray(audio);
        try {
            MaryAudioUtils.writeWavFile(samples, outputFileName, audio.getFormat());
            System.out.println("Output written to " + outputFileName);
        } catch (IOException e) {
            System.err.println("Could not write to file: " + outputFileName + "\n" + e.getMessage());
            System.exit(1);
        }
    }









}
