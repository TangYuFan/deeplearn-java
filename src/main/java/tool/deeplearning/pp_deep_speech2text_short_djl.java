package tool.deeplearning;

import ai.djl.Device;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.DataType;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import org.apache.commons.lang3.tuple.Pair;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.nio.Buffer;
import java.nio.ShortBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import ai.djl.Model;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.ArrayList;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.jtransforms.fft.DoubleFFT_1D;

/**
*   @desc : deep sepeech （中英文）端到端语言识别模型,pp开源模型  djl推理
 *
 *
*   @auth : tyf
*   @date : 2022-06-15  13:52:19
*/
public class pp_deep_speech2text_short_djl {


    // 音频处理一些参数
    static int mel_window_step = 10;
    static float max_gain_db = 300.0f;
    static int sample_rate = 16000;
    static float eps = 1e-14f;
    static float scale = (float) 1.0 / Float.valueOf(1 << ((8 * 2) - 1));
    static int n_bytes = 2;


    // FFT 工具类
    public static class FFT {

        // Compute the fast fourier transform
        public static double[] fft(double[] raw) {
            double[] in = raw;
            DoubleFFT_1D fft = new DoubleFFT_1D(in.length);
            fft.realForward(in);
            return in;
        }

        /**
         * Computes the physical layout of the fast fourier transform.
         * See jTransform documentation for more information.
         * http://incanter.org/docs/parallelcolt/api/edu/emory/mathcs/jtransforms/fft/DoubleFFT_1D.html#realForward(double[])
         *
         * @param fft the fast fourier transform
         */
        public static float[][] rfft(double[] fft) {
            float[][] result = null;

            int n = fft.length;
            if (n % 2 == 0) {
                // n is even
                result = new float[2][n / 2 + 1];
                for (int i = 0; i < n / 2; i++) {
                    result[0][i] = (float) fft[2 * i]; //the real part fo the fast fourier transform
                    result[1][i] = (float) fft[2 * i + 1]; //the imaginary part of the fast fourier transform
                }
                result[1][0] = 0;
                result[0][n / 2] = (float) fft[1];
            } else {
                // n is odd
                result = new float[2][(n + 1) / 2];
                for (int i = 0; i < n / 2; i++) {
                    result[0][i] = (float) fft[2 * i];  //the real part fo the fast fourier transform
                    result[1][i] = (float) fft[2 * i + 1];  //the imaginary part of the fast fourier transform
                }
                result[1][0] = 0;
                result[1][(n - 1) / 2] = (float) fft[1];

            }

            return result;
        }


        public static float[] abs(float[][] complex) {
            float[] re = complex[0]; //the real part fo the fast fourier transform
            float[] im = complex[1]; //the imaginary part of the fast fourier transform
            float[] abs = new float[re.length];
            for (int i = 0; i < re.length; i++) {
                abs[i] = (float) Math.hypot(re[i], im[i]);
            }
            return abs;
        }

        /**
         * Returns the Discrete Fourier Transform sample frequencies.
         * See numpy.fft.rfftfreq for more information.
         *
         * @param n Window length
         * @param d Sample spacing
         * @return Array of length n + 1 containing the sample frequencies
         */
        public static double[] rfftfreq(int n, double d) {
            double val = 1.0 / (n * d);
            int N = n / 2 + 1;
            double[] results = new double[N];
            for (int i = 0; i < N; i++) {
                results[i] = i * val;
            }
            return results;
        }
    }


    // 音频片段定义
    public static final class AudioSegment {
        public final float[] samples;
        public final Integer sampleRate;
        public final Integer audioChannels;

        public AudioSegment(float[] samples, Integer sampleRate, Integer audioChannels) {
            this.samples = samples;
            this.sampleRate = sampleRate;
            this.audioChannels = audioChannels;
        }
    }


    // 音频预处理工具
    public static class AudioProcess {

        public static NDArray processUtterance(NDManager manager, String npzDataPath,String audioPath) throws Exception {
            // 获取音频的float数组
            // Process audio utterance given the file path
            float[] floatArray = audioSegment(audioPath).samples;
            // System.out.println(Arrays.toString(floatArray));

            // 提取语音片段的特征
            // Extract features of audio segment
            NDArray specgram = featurize(manager, floatArray);

            // 使用均值和标准值计算音频特征的归一化值
            // Normalize audio feature using mean and std values
            specgram = apply(manager, npzDataPath, specgram);
            // System.out.println(specgram.toDebugString(1000000000, 1000, 10, 1000));

            return specgram;
        }

        /**
         * 使用均值和标准值计算音频特征的归一化值
         * Calculate the normalized value of audio features using mean and standard values
         *
         * @param manager
         * @param npzDataPath: 均值和标准值的文件路径 - file path of mean and standard values
         * @param features: 需要归一化的音频 - audio features that need to be normalized
         * @return
         * @throws Exception
         */
        public static NDArray apply(NDManager manager, String npzDataPath, NDArray features) throws Exception {
            //https://github.com/deepjavalibrary/djl/blob/master/api/src/test/java/ai/djl/ndarray/NDSerializerTest.java
            //https://github.com/deepjavalibrary/djl/blob/master/api/src/test/java/ai/djl/ndarray/NDListTest.java

            byte[] data = Files.readAllBytes(Paths.get(npzDataPath));
            NDList decoded = NDList.decode(manager, data);
            ByteArrayOutputStream bos = new ByteArrayOutputStream(data.length + 1);
            decoded.encode(bos, true);
            NDList list = NDList.decode(manager, bos.toByteArray());
            NDArray meanNDArray = list.get(0);//mean
            meanNDArray = meanNDArray.toType(DataType.FLOAT32, false);
            NDArray stdNDArray = list.get(1);//std
            stdNDArray = stdNDArray.toType(DataType.FLOAT32, false);

            // (features - self._mean) / (self._std + eps)
            stdNDArray = stdNDArray.add(eps);
            features = features.sub(meanNDArray).div(stdNDArray);
            return features;
        }


        /**
         * 从AudioSegment或SpeechSegment中提取音频特征
         * Extracts audio features from AudioSegment or SpeechSegment
         *
         * @param manager
         * @param floatArray
         * @return
         * @throws Exception
         */
        public static NDArray featurize(NDManager manager, float[] floatArray) {
            // 音频归一化
            // Audio normalization
            NDArray samples = manager.create(floatArray);
            float rmsDb = rmsDb(samples);
            // 返回以分贝为单位的音频均方根能量
            // Returns the root mean square energy of the audio in decibels
            System.out.println("Root Mean Square energy of audio:  " + rmsDb);

            // 提取特征前将音频归一化至-20 dB(以分贝为单位)
            // Normalize audio to -20 dB (in decibels) before feature extraction
            float target_dB = -20f;
            samples = normalize(samples, target_dB);

            // 生成帧的跨步大小(以毫秒为单位)
            // Frame step size in milliseconds
            float stride_ms = 10f;
            // 用于生成帧的窗口大小(毫秒)
            // Window size in milliseconds used for generating frames
            float window_ms = 20f;
            // 用快速傅里叶变换计算线性谱图
            // Calculate linear spectrogram using fast Fourier transform
            NDArray specgram = linearSpecgram(manager, samples, stride_ms, window_ms);
            // System.out.println(specgram.toDebugString(1000000000, 1000, 10, 1000));

            return specgram;
        }

        /**
         * 将音频归一化，使其具有所需的有效值(以分贝为单位) Target RMS value in decibels. This value should be less than 0.0 as
         * 0.0 is full-scale audio.
         *
         * @param samples
         * @param target_db
         * @return
         * @throws Exception
         */
        public static NDArray normalize(NDArray samples, float target_db) {
            float gain = target_db - rmsDb(samples);
            gain = Math.min(gain, max_gain_db);
            // 对音频施加分贝增益
            // Gain in decibels to apply to samples
            float factor = (float) Math.pow(10f, gain / 20f);
            samples = samples.mul(factor);
            return samples;
        }

        /**
         * 生成以分贝为单位的音频均方根能量 Root mean square energy in decibels.
         *
         * @param samples
         * @return
         */
        public static float rmsDb(NDArray samples) {
            samples = samples.pow(2);
            samples = samples.mean();
            samples = samples.log10().mul(10);
            return samples.toFloatArray()[0];
        }

        /**
         * 获取音频文件的float数组,sampleRate,audioChannels
         * Get the float array, sample rate, and audio channels of the audio file
         *
         * @param path
         * @return
         * @throws FrameGrabber.Exception
         */
        public static AudioSegment audioSegment(String path) throws FrameGrabber.Exception {
            AudioSegment audioSegment = null;
            int sampleRate = -1;
            int audioChannels = -1;
            //  Audio sample type is usually integer or float-point.
            //  Integers will be scaled to [-1, 1] in float32.
            float scale = (float) 1.0 / Float.valueOf(1 << ((8 * 2) - 1));
            List<Float> floatList = new ArrayList<>();

            try (FFmpegFrameGrabber audioGrabber = new FFmpegFrameGrabber(path)) {
                try {
                    audioGrabber.start();
                    sampleRate = audioGrabber.getSampleRate();
                    audioChannels = audioGrabber.getAudioChannels();
                    Frame frame;
                    while ((frame = audioGrabber.grabFrame()) != null) {
                        Buffer[] buffers = frame.samples;

                        Buffer[] copiedBuffers = new Buffer[buffers.length];
                        for (int i = 0; i < buffers.length; i++) {
                            deepCopy((ShortBuffer) buffers[i], (ShortBuffer) copiedBuffers[i]);
                        }

                        ShortBuffer sb = (ShortBuffer) buffers[0];
                        for (int i = 0; i < sb.limit(); i++) {
                            floatList.add(new Float(sb.get() * scale));
                        }
                    }
                } catch (FrameGrabber.Exception e) {
                    e.printStackTrace();
                }

                float[] floatArray = new float[floatList.size()];
                int i = 0;
                for (Float f : floatList) {
                    floatArray[i++] = (f != null ? f : Float.NaN); // Or whatever default you want.
                }
                audioSegment = new AudioSegment(floatArray, sampleRate, audioChannels);
                return audioSegment;
            }
        }


        private static ShortBuffer deepCopy(ShortBuffer source, ShortBuffer target) {

            int sourceP = source.position();
            int sourceL = source.limit();

            if (null == target) {
                target = ShortBuffer.allocate(source.remaining());
            }
            target.put(source);
            target.flip();

            source.position(sourceP);
            source.limit(sourceL);
            return target;
        }


        /**
         * 用快速傅里叶变换计算线性谱图
         * Compute linear spectrogram with fast Fourier transform.
         *
         * @param manager
         * @param samples
         * @param stride_ms
         * @param window_ms
         * @return
         */
        public static NDArray linearSpecgram( NDManager manager, NDArray samples, float stride_ms, float window_ms) {

            int strideSize = (int) (0.001 * sample_rate * stride_ms);
            int windowSize = (int) (0.001 * sample_rate * window_ms);
            long truncateSize = (samples.size() - windowSize) % strideSize;
            long len = samples.size() - truncateSize;
            samples = samples.get(":" + len);

            // Shape nshape = new Shape(windowSize, (samples.size() - windowSize) / strideSize + 1);    //
            // 320 ,838
            // nstrides = (samples.strides[0], samples.strides[0] * stride_size)
            // strides[0] = 4 个字节, 由于已经转为float类型，所以对应当前samples中一个元素
            // np.lib.stride_tricks.as_strided(samples, shape=nshape, strides=nstrides)
            int rows = windowSize; // 320
            int cols = ((int) samples.size() - windowSize) / strideSize + 1; // 838

            float[] floatArray = samples.toFloatArray();
            float[][] windows = new float[rows][cols];
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    windows[row][col] = floatArray[row + col * strideSize];
                }
            }

            // 快速傅里叶变换
            // Fast Fourier Transform
            float[] weighting = hanningWindow(windowSize);

            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    windows[row][col] = windows[row][col] * weighting[row];
                }
            }

            double[] arr = null;
            NDList fftList = new NDList();
            for (int col = 0; col < cols; col++) {
                arr = new double[rows];
                for (int row = 0; row < rows; row++) {
                    arr[row] = windows[row][col];
                }
                double[] fft = FFT.fft(arr);
                float[][] complex = FFT.rfft(fft);

                NDArray array = manager.create(FFT.abs(complex));
                fftList.add(array);
            }

            NDArray fft = NDArrays.stack(fftList).transpose();
            fft = fft.pow(2);

            NDArray weightingArray = manager.create(weighting);

            weightingArray = weightingArray.pow(2);
            NDArray scale = weightingArray.sum().mul(sample_rate);

            NDArray middle = fft.get("1:-1,:");
            middle = middle.mul(2).div(scale);
            NDArray head = fft.get("0,:").div(scale).reshape(1, fft.getShape().get(1));
            NDArray tail = fft.get("-1,:").div(scale).reshape(1, fft.getShape().get(1));
            NDList list = new NDList(head, middle, tail);
            fft = NDArrays.concat(list, 0);

            NDArray freqsArray = manager.arange(fft.getShape().get(0));
            freqsArray = freqsArray.mul(sample_rate / windowSize);

            float[] freqs = freqsArray.toFloatArray();
            int ind = 0;
            for (int i = 0; i < freqs.length; i++) {
                if (freqs[i] <= (sample_rate / 2)) {
                    ind = i;
                } else {
                    break;
                }
            }
            ind = ind + 1;

            fft = fft.get(":" + ind + ",:").add(eps);
            fft = fft.log();

            //        System.out.println(fft.toDebugString(1000000000, 1000, 10, 1000));
            return fft;
        }


        /**
         * Hanning窗
         * The Hanning window is a taper formed by using a weighted cosine.
         *
         * @param size
         * @return
         */
        public static float[] hanningWindow(int size) {
            float[] data = new float[size];
            for (int n = 1; n < size; n++) {
                data[n] = (float) (0.5 * (1 - Math.cos((2 * Math.PI * n)
                        / (size - 1))));
            }
            return data;
        }


    }


    // CTC贪婪(最佳路径)解码器-模型输出解码
    public static class CTCGreedyDecoder {

        /**
         * 由最可能的令牌组成的路径将被进一步后处理到去掉连续重复和所有空白
         * The path consisting of the most probable tokens is further post-processed to remove consecutive duplicates and all blanks
         *
         * @param manager
         * @param probs_seq: 每一条都是2D的概率表。每个元素都是浮点数概率的列表一个字符
         *                   a list of 2D probability tables. Each element is a list of floating point probabilities for a character
         * @param vocabulary: 词汇列表  - vocabulary list
         * @param blank_index: 需要移除的空白索引 - blank index that needs to be removed
         * @return 解码后得到的 score,字符串 - the score and string obtained after decoding
         * @throws Exception
         */
        public static Pair greedyDecoder(
                NDManager manager, NDArray probs_seq, List<String> vocabulary, long blank_index) {
            // 获得每个时间步的最佳索引
            // Get the best index for each time step
            float[] floats = probs_seq.toFloatArray();
            int rows = (int) probs_seq.getShape().get(0);
            int cols = (int) probs_seq.getShape().get(1);

            long[] max_index_list = probs_seq.argMax(1).toLongArray();

            List<Float> max_prob_list = new ArrayList<>();
            for (int i = 0; i < rows; i++) {
                if (max_index_list[i] != blank_index) {
                    max_prob_list.add(probs_seq.getFloat(i, max_index_list[i]));
                }
            }

            // 删除连续的重复"索引"
            // Remove consecutive duplicate "indices"
            List<Long> index_list = new ArrayList<>();
            long current = max_index_list[0];
            index_list.add(current);
            for (int i = 1; i < max_index_list.length; i++) {
                if (max_index_list[i] != current) {
                    index_list.add(max_index_list[i]);
                    current = max_index_list[i];
                }
            }

            // 删除空索引
            // Remove blank indices
            List<Long> pure_index_list = new ArrayList<>();
            for (Long value : index_list) {
                if (value != blank_index) {
                    pure_index_list.add(value);
                }
            }

            // 索引列表转换为字符串
            // Convert index list to string
            StringBuffer sb = new StringBuffer();
            for (Long value : pure_index_list) {
                sb.append(vocabulary.get(value.intValue()));
            }

            float score = 0;
            if (max_prob_list.size() > 0) {
                float sum = 0;
                for (Float value : max_prob_list) {
                    sum += value;
                }
                score = (sum / max_prob_list.size()) * 100.0f;
            }

            return Pair.of(score, sb.toString());
        }
    }


    // 模型处理器
    public static class AudioTranslator implements Translator<NDArray, Pair> {
        AudioTranslator() {}

        private List<String> vocabulary = null;

        @Override
        public void prepare(TranslatorContext ctx) throws IOException {
            Model model = ctx.getModel();
            try (InputStream is = model.getArtifact("zh_vocab.txt").openStream()) {
                vocabulary = Utils.readLines(is, true);
            }
        }

        @Override
        public NDList processInput(TranslatorContext ctx, NDArray audioFeature) {
            NDManager manager = ctx.getNDManager();

            long audio_len = audioFeature.getShape().get(1);
            long mask_shape0 = (audioFeature.getShape().get(0) - 1) / 2 + 1;
            long mask_shape1 = (audioFeature.getShape().get(1) - 1) / 3 + 1;
            long mask_max_len = (audio_len - 1) / 3 + 1;

            NDArray mask_ones = manager.ones(new Shape(mask_shape0, mask_shape1));
            NDArray mask_zeros = manager.zeros(new Shape(mask_shape0, mask_max_len - mask_shape1));
            NDArray maskArray = NDArrays.concat(new NDList(mask_ones, mask_zeros), 1);
            maskArray = maskArray.reshape(1, mask_shape0, mask_max_len);
            NDList list = new NDList();
            for (int i = 0; i < 32; i++) {
                list.add(maskArray);
            }
            NDArray mask = NDArrays.concat(list, 0);

            NDArray audio_data = audioFeature.expandDims(0);
            NDArray seq_len_data = manager.create(new long[] {audio_len});
            NDArray masks = mask.expandDims(0);
            //    System.out.println(maskArray.toDebugString(1000000000, 1000, 10, 1000));
            return new NDList(audio_data, seq_len_data, masks);
        }

        @Override
        public Pair processOutput(TranslatorContext ctx, NDList list) {
            NDArray probs_seq = list.singletonOrThrow();
            Pair pair = CTCGreedyDecoder.greedyDecoder(ctx.getNDManager(), probs_seq, vocabulary, 0);
            return pair;
        }

        @Override
        public Batchifier getBatchifier() {
            return null;
        }
    }


    // 加载模型
    public static class SpeechRecognition {

        String modelPath;
        public SpeechRecognition(String modelPath) {
            this.modelPath = modelPath;
        }

        public Criteria<NDArray, Pair> criteria() {
            Criteria<NDArray, Pair> criteria =
                    Criteria.builder()
                            .setTypes(NDArray.class, Pair.class)
                            .optModelPath(new File(modelPath).toPath())
                            .optTranslator(new AudioTranslator())
                            .optEngine("PaddlePaddle") // Use PaddlePaddle engine
                            .optProgress(new ProgressBar())
                            .build();

            return criteria;
        }
    }


    public static void main(String[] args) throws Exception {


        // 模型
        String modelPath = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\pp_deep_speech2text_short_djl\\deep_speech.zip";

        // 音频归一化处理
        String meatStd = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\pp_deep_speech2text_short_djl\\mean_std.npz";

        // 测试音频,
        String wavPath = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\pp_deep_speech2text_short_djl\\test.wav";


        NDManager manager = NDManager.newBaseManager(Device.cpu());
        // 传入音频处理
        NDArray audioFeature = AudioProcess.processUtterance(manager, meatStd,wavPath);
        SpeechRecognition speakerEncoder = new SpeechRecognition(modelPath);
        Criteria<NDArray, Pair> criteria = speakerEncoder.criteria();

        try (ZooModel<NDArray, Pair> model = criteria.loadModel();
             Predictor<NDArray, Pair> predictor = model.newPredictor()) {
            Pair result = predictor.predict(audioFeature);
            System.out.println("Score : " + result.getLeft());
            System.out.println("Words : " + result.getRight());
        }
    }

}
