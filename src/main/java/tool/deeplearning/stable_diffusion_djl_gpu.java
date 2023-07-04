package tool.deeplearning;





import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

/**
 *   @desc : stable_diffusion AI画图，文生图  djl 推理
 *   @auth : tyf
 *   @date : 2022-06-15  15:37:13
 */
public final class stable_diffusion_djl_gpu {


    // 图片编码器和解码器
    public static class ImageDecoder implements NoBatchifyTranslator<NDArray, Image> {

        @Override
        public NDList processInput(TranslatorContext ctx, NDArray input) throws Exception {
            input = input.div(0.18215);
            return new NDList(input);
        }

        @Override
        public Image processOutput(TranslatorContext ctx, NDList output) throws Exception {
            NDArray scaled = output.get(0).div(2).add(0.5).clip(0, 1);
            scaled = scaled.transpose(0, 2, 3, 1);
            scaled = scaled.mul(255).round().toType(DataType.INT8, true).get(0);
            return ImageFactory.getInstance().fromNDArray(scaled);
        }
    }

    // 图片编码器和解码器
    public static class ImageEncoder implements NoBatchifyTranslator<Image, NDArray> {

        private int height;
        private int width;

        public ImageEncoder(int height, int width) {
            this.height = height;
            this.width = width;
        }

        @Override
        public NDArray processOutput(TranslatorContext ctx, NDList list) throws Exception {
            NDArray result = list.singletonOrThrow();
            result = result.mul(0.18215f);
            result.detach();
            return result;
        }

        @Override
        public NDList processInput(TranslatorContext ctx, Image input) throws Exception {
            NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
            // model take 32-based size
            int[] size = resize32(height, width);

            array = NDImageUtils.resize(array, size[1], size[0]);
            array = array.transpose(2, 0, 1).div(255f); // HWC -> CHW RGB
            array = array.mul(2).sub(1);
            array = array.expandDims(0);
            return new NDList(array);
        }

        private int[] resize32(double h, double w) {
            double min = Math.min(h, w);
            if (min < 32) {
                h = 32.0 / min * h;
                w = 32.0 / min * w;
            }
            int h32 = (int) h / 32;
            int w32 = (int) w / 32;
            return new int[] {h32 * 32, w32 * 32};
        }
    }


    // 文本编码器
    public static class TextEncoder implements NoBatchifyTranslator<String, NDList> {

        private static final int MAX_LENGTH = 77;

        HuggingFaceTokenizer tokenizer;

        /** {@inheritDoc} */
        @Override
        public void prepare(TranslatorContext ctx) throws IOException {

            String pt_root = new File("").getCanonicalPath() +"\\model\\deeplearning\\stable_diffusion_djl_gpu\\";

            tokenizer =
                    HuggingFaceTokenizer.builder()
                            .optPadding(true)
                            .optPadToMaxLength()
                            .optMaxLength(MAX_LENGTH)
                            .optTruncation(true)
                            .optTokenizerPath(Paths.get(pt_root+"tokenizer.json"))
                            .build();
        }

        /** {@inheritDoc} */
        @Override
        public NDList processOutput(TranslatorContext ctx, NDList list) {
            list.detach();
            return list;
        }

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, String input) {
            Encoding encoding = tokenizer.encode(input);
            Shape shape = new Shape(1, encoding.getIds().length);
            return new NDList(ctx.getNDManager().create(encoding.getIds(), shape));
        }
    }


    // 进度
    public static class PndmScheduler {

        private static final int TRAIN_TIMESTEPS = 1000;
        private static final float BETA_START = 0.00085f;
        private static final float BETA_END = 0.012f;

        private NDManager manager;
        private NDArray alphasCumProd;
        private float finalAlphaCumProd;
        private int counter;
        private NDArray curSample;
        private NDList ets;
        private int stepSize;
        private int[] timesteps;

        public PndmScheduler(NDManager manager) {
            this.manager = manager;
            NDArray betas = manager.linspace(
                            (float) Math.sqrt(BETA_START),
                            (float) Math.sqrt(BETA_END),
                            TRAIN_TIMESTEPS)
                    .square();
            NDArray alphas = manager.ones(betas.getShape()).sub(betas);
            alphasCumProd = alphas.cumProd(0);
            finalAlphaCumProd = alphasCumProd.get(0).toFloatArray()[0];
            ets = new NDList();
        }

        public NDArray addNoise(NDArray latent, NDArray noise, int timesteps) {
            float alphaProd = alphasCumProd.get(timesteps).toFloatArray()[0];
            float sqrtOneMinusAlphaProd = (float) Math.sqrt(1 - alphaProd);
            latent = latent.mul(alphaProd).add(noise.mul(sqrtOneMinusAlphaProd));
            return latent;
        }

        public void initTimesteps(int inferenceSteps, int offset) {
            stepSize = TRAIN_TIMESTEPS / inferenceSteps;
            NDArray timestepsNd = manager.arange(0, inferenceSteps).mul(stepSize).add(offset);

            // np.concatenate([self._timesteps[:-1], self._timesteps[-2:-1],
            // self._timesteps[-1:]])[::-1]
            NDArray part1 = timestepsNd.get(new NDIndex(":-1"));
            NDArray part2 = timestepsNd.get(new NDIndex("-2:-1"));
            NDArray part3 = timestepsNd.get(new NDIndex("-1:"));
            NDList list = new NDList();
            list.add(part1);
            list.add(part2);
            list.add(part3);
            // timesteps = timesteps.get(new NDIndex("::-1"));
            timesteps = NDArrays.concat(list).flatten().flip(0).toIntArray();
        }

        public NDArray step(NDArray modelOutput, int timestep, NDArray sample) {
            int prevTimestep = timestep - stepSize;
            if (counter != 1) {
                ets.add(modelOutput);
            } else {
                prevTimestep = timestep;
                timestep -= stepSize;
            }

            if (ets.size() == 1 && counter == 0) {
                curSample = sample;
            } else if (ets.size() == 1 && counter == 1) {
                modelOutput = modelOutput.add(ets.get(0)).div(2);
                sample = curSample;
            } else if (ets.size() == 2) {
                NDArray firstModel = ets.get(ets.size() - 1).mul(3);
                NDArray secondModel = ets.get(ets.size() - 2).mul(-1);
                modelOutput = firstModel.add(secondModel);
                modelOutput = modelOutput.div(2);
            } else if (ets.size() == 3) {
                NDArray firstModel = ets.get(ets.size() - 1).mul(23);
                NDArray secondModel = ets.get(ets.size() - 2).mul(-16);
                NDArray thirdModel = ets.get(ets.size() - 3).mul(5);
                modelOutput = firstModel.add(secondModel).add(thirdModel);
                modelOutput = modelOutput.div(12);
            } else {
                NDArray firstModel = ets.get(ets.size() - 1).mul(55);
                NDArray secondModel = ets.get(ets.size() - 2).mul(-59);
                NDArray thirdModel = ets.get(ets.size() - 3).mul(37);
                NDArray fourthModel = ets.get(ets.size() - 4).mul(-9);
                modelOutput = firstModel.add(secondModel).add(thirdModel).add(fourthModel);
                modelOutput = modelOutput.div(24);
            }

            NDArray prevSample = getPrevSample(sample, timestep, prevTimestep, modelOutput);
            prevSample.setName("prev_sample");
            counter++;

            return prevSample;
        }

        public int[] getTimesteps() {
            return timesteps;
        }

        public void setTimesteps(int[] timesteps) {
            this.timesteps = timesteps;
        }

        private NDArray getPrevSample(
                NDArray sample, int timestep, int prevTimestep, NDArray modelOutput) {
            float alphaProdT = alphasCumProd.toFloatArray()[timestep];
            float alphaProdTPrev;

            if (prevTimestep >= 0) {
                alphaProdTPrev = alphasCumProd.toFloatArray()[prevTimestep];
            } else {
                alphaProdTPrev = finalAlphaCumProd;
            }

            float betaProdT = 1 - alphaProdT;
            float betaProdTPrev = 1 - alphaProdTPrev;

            float sampleCoeff = (float) Math.sqrt(alphaProdTPrev / alphaProdT);
            float modelOutputCoeff = alphaProdT * (float) Math.sqrt(betaProdTPrev)
                    + (float) Math.sqrt(alphaProdT * betaProdT * alphaProdTPrev);

            sample = sample.mul(sampleCoeff);
            modelOutput = modelOutput.mul(alphaProdTPrev - alphaProdT).div(modelOutputCoeff).neg();
            return sample.add(modelOutput);
        }
    }



    // stable diffusion
    public static class StableDiffusionModel {

        private static final int HEIGHT = 512;
        private static final int WIDTH = 512;
        private static final int OFFSET = 1;
        private static final float GUIDANCE_SCALE = 7.5f;
        private static final float STRENGTH = 0.75f;

        private Predictor<Image, NDArray> vaeEncoder;
        private Predictor<NDArray, Image> vaeDecoder;
        private Predictor<String, NDList> textEncoder;
        private Predictor<NDList, NDList> unetExecutor;
        private Device device;

        public String pt_root = new File("").getCanonicalPath() +"\\model\\deeplearning\\stable_diffusion_djl_gpu\\";

        public StableDiffusionModel(Device device) throws ModelException, IOException {
            this.device = device;
            String type = device.getDeviceType();
            if (!"cpu".equals(type) && !"gpu".equals(type)) {
                throw new UnsupportedOperationException(type + " device not supported!");
            }
            Criteria<NDList, NDList> unetCriteria = Criteria.builder()
                    .setTypes(NDList.class, NDList.class)
                    .optModelPath(Paths.get(pt_root+"unet_traced_model.pt"))
                    .optEngine("PyTorch")
                    .optProgress(new ProgressBar())
                    .optTranslator(new NoopTranslator())
                    .optDevice(device)
                    .build();
            this.unetExecutor = unetCriteria.loadModel().newPredictor();


            Criteria<NDArray, Image> decoderCriteria = Criteria.builder()
                    .setTypes(NDArray.class, Image.class)
                    .optModelPath(Paths.get(pt_root+"vae_decode_model.pt"))
                    .optEngine("PyTorch")
                    .optTranslator(new ImageDecoder())
                    .optProgress(new ProgressBar())
                    .optDevice(device)
                    .build();
            this.vaeDecoder = decoderCriteria.loadModel().newPredictor();


            Criteria<String, NDList> criteria = Criteria.builder()
                    .setTypes(String.class, NDList.class)
                    .optModelPath(Paths.get(pt_root+"text_encoder.pt"))
                    .optEngine("PyTorch")
                    .optProgress(new ProgressBar())
                    .optTranslator(new TextEncoder())
                    .optDevice(device)
                    .build();
            this.textEncoder = criteria.loadModel().newPredictor();
        }

        public Image generateImageFromText(String prompt, int steps)
                throws ModelException, IOException, TranslateException {
            return generateImageFromImage(prompt, null, steps);
        }

        public Image generateImageFromImage(String prompt, Image image, int steps)
                throws ModelException, IOException, TranslateException {
            // TODO: implement this part
            try (NDManager manager = NDManager.newBaseManager(device, "PyTorch")) {
                // Step 1: Build text embedding（提示编码）
                NDList textEncoding = textEncoder.predict(prompt);
                NDList uncondEncoding = textEncoder.predict("");
                textEncoding.attach(manager);
                uncondEncoding.attach(manager);
                NDArray textEncodingArray = textEncoding.get(1);
                NDArray uncondEncodingArray = uncondEncoding.get(1);
                NDArray embeddings = textEncodingArray.concat(uncondEncodingArray);
                // Step 2: Build latent（潜在图像）
                PndmScheduler scheduler = new PndmScheduler(manager);
                scheduler.initTimesteps(steps, OFFSET);
                Shape latentInitShape = new Shape(1, 4, HEIGHT / 8, WIDTH / 8);
                NDArray latent;
                if (image != null) {
                    loadImageEncoder();
                    latent = vaeEncoder.predict(image);
                    NDArray noise = manager.randomNormal(latent.getShape());
                    // Step 2.5: reset timestep to reflect on the given image
                    int initTimestep = (int) (steps * STRENGTH) + OFFSET;
                    initTimestep = Math.min(initTimestep, steps);
                    int[] timestepsArr = scheduler.getTimesteps();
                    int timesteps = timestepsArr[timestepsArr.length - initTimestep];
                    latent = scheduler.addNoise(latent, noise, timesteps);
                    int tStart = Math.max(steps - initTimestep + OFFSET, 0);
                    scheduler.setTimesteps(
                            Arrays.copyOfRange(timestepsArr, tStart, timestepsArr.length));
                } else {
                    latent = manager.randomNormal(latentInitShape);
                }
                // Step 3: Start iterating/generating（加噪）
                ProgressBar pb = new ProgressBar("Generating", steps);
                pb.start(0);
                for (int i = 0; i < steps; i++) {
                    long t = scheduler.getTimesteps()[i];
                    NDArray latentModelOutput = latent.concat(latent);
                    NDArray noisePred = unetExecutor
                            .predict(
                                    new NDList(
                                            latentModelOutput, manager.create(t),
                                            embeddings))
                            .get(0);
                    NDList splitNoisePred = noisePred.split(2);
                    NDArray noisePredText = splitNoisePred.get(0);
                    NDArray noisePredUncond = splitNoisePred.get(1);
                    NDArray scaledNoisePredUncond = noisePredText.add(noisePredUncond.neg());
                    scaledNoisePredUncond = scaledNoisePredUncond.mul(GUIDANCE_SCALE);
                    noisePred = noisePredUncond.add(scaledNoisePredUncond);
                    latent = scheduler.step(noisePred, (int) t, latent);
                    pb.increment(1);
                }
                pb.end();
                // Step 4: get final image（结束）
                return vaeDecoder.predict(latent);
            }
        }

        private void loadImageEncoder() throws ModelException, IOException {
            if (vaeEncoder != null) {
                return;
            }
            Criteria<Image, NDArray> criteria = Criteria.builder()
                    .setTypes(Image.class, NDArray.class)
                    .optModelPath(Paths.get(pt_root+"vae_encode_model.pt"))
                    .optEngine("PyTorch")
                    .optTranslator(new ImageEncoder(HEIGHT, WIDTH))
                    .optProgress(new ProgressBar())
                    .optDevice(device)
                    .build();
            vaeEncoder = criteria.loadModel().newPredictor();
        }
    }


    public static void main(String[] args) throws ModelException, IOException, TranslateException {

 stable_diffusion_djl_cpu.StableDiffusionModel model = new stable_diffusion_djl_cpu.StableDiffusionModel(Device.cpu());

        String text = "A pair of young Chinese lovers, wearing jackets and jeans, sitting on the roof, the background is Beijing in the 1990s, and the opposite building can be seen";

        String text2 = "Vincent van Gogh’s painting of Emma Watson";

        String text3 = "Anime Kawaii Beauty Naked,full body";

        Image result = model.generateImageFromText(text3, 50);

        BufferedImage imgOut = (BufferedImage)result.getWrappedImage();

        // 显示
        // 弹窗显示
        JFrame frame = new JFrame("Image");
        frame.setSize(imgOut.getWidth(), imgOut.getHeight());
        JPanel panel = new JPanel();
        panel.add(new JLabel(new ImageIcon(imgOut)));
        frame.getContentPane().add(panel);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setResizable(false);

    }

}

