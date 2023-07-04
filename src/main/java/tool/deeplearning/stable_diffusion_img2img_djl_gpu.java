package tool.deeplearning;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.BufferedImageFactory;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.ndarray.NDArrays;

import javax.swing.*;

/**
 *   @desc : stable_diffusion AI画图，图生图  djl 推理
 *   @auth : tyf
 *   @date : 2022-06-15  15:37:13
 */
public class stable_diffusion_img2img_djl_gpu {

    public static class EncoderTranslator implements Translator<Image, NDArray> {
        List<String> classes;

        public EncoderTranslator() {
        }

        @Override
        public void prepare(TranslatorContext ctx) throws IOException {
        }

        public NDArray processOutput(TranslatorContext ctx, NDList list) {
            NDArray result = list.singletonOrThrow();
            result = result.mul(0.18215f);
            result.detach();
            return result;
        }

        public NDList processInput(TranslatorContext ctx, Image input) throws Exception {
            NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
            // model take 32-based size
            int h = 512;
            int w = 512;
            int[] size = resize32(h, w);
            array = NDImageUtils.resize(array,size[1],size[0]);
            array = array.transpose(2, 0, 1).div(255f);  // HWC -> CHW RGB
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

        public Batchifier getBatchifier() {
            return null;
        }
    }



    public static class StableDiffusionPNDMScheduler {
        private final int numTrainTimesteps = 1000;
        private int numInferenceSteps;
        private float betaStart = (float) 0.00085;
        private float betaEnd = (float) 0.012;
        private NDManager manager;
        private NDArray betas;
        private NDArray alphas;
        private NDArray alphasCumProd;
        private NDArray finalAlphaCumProd;
        private int counter = 0;
        private NDArray curSample = null;
        private NDList ets = new NDList();
        private int stepSize;
        public NDArray timesteps;

        private StableDiffusionPNDMScheduler() {
        }

        public StableDiffusionPNDMScheduler(NDManager mgr) {
            manager = mgr;
            // scaled_linear
            betas = manager.linspace((float) Math.sqrt(betaStart), (float) Math.sqrt(betaEnd), numTrainTimesteps);
            betas = betas.mul(betas);
            alphas = manager.ones(betas.getShape()).add(betas.neg());
            alphasCumProd = manager.create(cumProd(alphas));
            finalAlphaCumProd = alphasCumProd.get(0);
        }

        private float[] cumProd(NDArray array) {
            float cumulative = 1;
            float[] alphasCumProdArr = new float[numTrainTimesteps];
            float[] alphasArr = array.toFloatArray();
            for (int i = 0; i < alphasCumProdArr.length; i++) {
                alphasCumProdArr[i] = alphasArr[i] * cumulative;
                cumulative = alphasCumProdArr[i];
            }
            return alphasCumProdArr;
        }

        public NDArray addNoise(NDArray latent, NDArray noise, int timesteps){
            NDArray alphaProd = alphasCumProd.get(timesteps);
            NDArray sqrtAlphaProd = alphaProd.sqrt();

            NDArray one = manager.create(new float[]{1});
            NDArray sqrtOneMinusAlphaProd = one.sub(alphaProd).sqrt();

            latent = latent.mul(sqrtAlphaProd).add(noise.mul(sqrtOneMinusAlphaProd));
            return latent;
        }

        public void setTimesteps(int inferenceSteps, int offset) {
            numInferenceSteps = inferenceSteps;
            stepSize = numTrainTimesteps / numInferenceSteps;
            timesteps = manager.arange(0, numInferenceSteps).mul(stepSize).add(offset);
            // np.concatenate([self._timesteps[:-1], self._timesteps[-2:-1], self._timesteps[-1:]])[::-1]
            NDArray part1 = timesteps.get(new NDIndex(":-1"));
            NDArray part2 = timesteps.get(new NDIndex("-2:-1"));
            NDArray part3 = timesteps.get(new NDIndex("-1:"));
            NDList list = new NDList();
            list.add(part1);
            list.add(part2);
            list.add(part3);
            // [::-1]
            timesteps = NDArrays.concat(list).flip(0);
        }

        public NDArray step(NDArray modelOutput, NDArray timestep, NDArray sample) {
            NDArray prevTimestep = manager.create(timestep.getInt() - stepSize);
            if (counter != 1) {
                ets.add(modelOutput);
            } else {
                prevTimestep = timestep.duplicate();
                timestep.add(-stepSize);
            }

            if (ets.size() == 1 && counter == 0) {
                curSample = sample;
            } else if (ets.size() == 1 && counter == 1) {
                modelOutput = modelOutput.add(ets.get(0)).div(2);
                sample = curSample;
                curSample = null;
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

        private NDArray getPrevSample(NDArray sample, NDArray timestep, NDArray prevTimestep, NDArray modelOutput) {
            NDArray alphaProdT = alphasCumProd.get(timestep);

            NDArray alphaProdTPrev;
            if (prevTimestep.getInt() >= 0) {
                alphaProdTPrev = alphasCumProd.get(prevTimestep);
            } else {
                alphaProdTPrev = finalAlphaCumProd;
            }

            NDArray one = manager.create(new float[]{1});
            NDArray betaProdT = one.sub(alphaProdT);
            NDArray betaProdTPrev = one.sub(alphaProdTPrev);

            NDArray sampleCoeff = alphaProdTPrev.div(alphaProdT).sqrt();

            NDArray partA = betaProdTPrev.sqrt().mul(alphaProdT);
            NDArray partB = alphaProdT.mul(betaProdT).mul(alphaProdTPrev).sqrt();

            NDArray modelOutputCoeff = partA.add(partB);

            sample = sample.mul(sampleCoeff);
            modelOutput = modelOutput.mul(alphaProdTPrev.sub(alphaProdT));
            modelOutput = modelOutput.div(modelOutputCoeff);
            modelOutput = modelOutput.neg();
            return sample.add(modelOutput);
        }
    }



    public static class Img2ImgStableDiffusionGPU {

        private static final Logger logger = LoggerFactory.getLogger(Img2ImgStableDiffusionGPU.class);
        private static final float strength = 0.75f;
        private static final int UNKNOWN_TOKEN = 49407;
        private static final int MAX_LENGTH = 77;
        private static final int steps = 25;
        private static final int offset = 1;
        private static final float guidanceScale = (float) 7.5;
        private static final Engine engine = Engine.getEngine("PyTorch");
        private static final NDManager manager = NDManager.newBaseManager(engine.defaultDevice(), engine.getEngineName());
        private static HuggingFaceTokenizer tokenizer;

        private static String vocab_path;
        private static String model_path_text_encoder_model_gpu0;
        private static String model_path_unet_traced_model_gpu0;
        private static String model_path_vae_decode_model_gpu0;
        private static String model_path_vae_encoder_model_gpu0;

        private Img2ImgStableDiffusionGPU() {
        }


        public static void main(
                String model1,
                String model2,
                String model3,
                String model4,
                String token,
                String vocab,
                String img,
                String prompt,
                String negative_prompt
        ) throws IOException, TranslateException, ModelException {


            // token初始化
            tokenizer = HuggingFaceTokenizer.builder()
                    .optManager(manager)
                    .optPadding(true)
                    .optPadToMaxLength()
                    .optMaxLength(MAX_LENGTH)
                    .optTruncation(true)
                    .optTokenizerPath(new File(token).toPath())
                    .build();

            // 模型路径设置
            model_path_text_encoder_model_gpu0 = model1;
            model_path_unet_traced_model_gpu0 = model2;
            model_path_vae_decode_model_gpu0 = model3;
            model_path_vae_encoder_model_gpu0 = model4;
            vocab_path = vocab;

            Path imageFile = new File(img).toPath();
            Image image = ImageFactory.getInstance().fromFile(imageFile);

            NDList textEncoding = SDTextEncoder(SDTextTokenizer(prompt));
            NDList uncondEncoding = SDTextEncoder(SDTextTokenizer(negative_prompt));

            NDArray textEncodingArray = textEncoding.get(1);
            NDArray uncondEncodingArray = uncondEncoding.get(1);

            NDArray embeddings = uncondEncodingArray.concat(textEncodingArray);

            StableDiffusionPNDMScheduler scheduler = new StableDiffusionPNDMScheduler(manager);
            scheduler.setTimesteps(steps, offset);
            int initTimestep = (int) (steps * strength) + offset;
            initTimestep = Math.min(initTimestep, steps);
            int timesteps = scheduler.timesteps.get(new NDIndex("-" + initTimestep)).toIntArray()[0];


            NDArray latent = SDEncoder(image);
            NDArray noise = manager.randomNormal(latent.getShape());
            latent = scheduler.addNoise(latent, noise, timesteps);

            int tStart = Math.max(steps - initTimestep + offset, 0);
            int[] timestepArr = scheduler.timesteps.get(new NDIndex(tStart + ":")).toIntArray();

            Predictor<NDList, NDList> predictor = SDUNetPredictor();

            for (int i = 0; i < timestepArr.length; i++) {
                NDArray t = manager.create(timestepArr[i]);
                NDArray latentModelInput = latent.concat(latent);
                // embeddings 2,77,768
                // t tensor 981
                // latentModelOutput 2,4,64,64

                NDArray noisePred = predictor.predict(buildUnetInput(embeddings, t, latentModelInput)).get(0);

                NDList splitNoisePred = noisePred.split(2);
                NDArray noisePredUncond = splitNoisePred.get(0);
                NDArray noisePredText = splitNoisePred.get(1);

                NDArray scaledNoisePredUncond = noisePredText.add(noisePredUncond.neg());
                scaledNoisePredUncond = scaledNoisePredUncond.mul(guidanceScale);
                noisePred = noisePredUncond.add(scaledNoisePredUncond);

                latent = scheduler.step(noisePred, t, latent);
            }
            saveImage(latent);

            logger.info("Stable diffusion image generated from prompt: \"{}\".", prompt);
        }

        private static void saveImage(NDArray input) throws TranslateException, ModelNotFoundException,
                MalformedModelException, IOException {
            input = input.div(0.18215);

            NDList encoded = new NDList();
            encoded.add(input);

            NDList decoded = SDDecoder(encoded);
            NDArray scaled = decoded.get(0).div(2).add(0.5).clip(0, 1);

            scaled = scaled.transpose(0, 2, 3, 1);
            scaled = scaled.mul(255).round().toType(DataType.INT8, true).get(0);
            Image image = BufferedImageFactory.getInstance().fromNDArray(scaled);

            // 弹窗显示
            BufferedImage backup = (BufferedImage)image.getWrappedImage();
            JFrame frame = new JFrame("Image");
            frame.setSize(backup.getWidth(), backup.getHeight());
            JPanel panel = new JPanel();
            panel.add(new JLabel(new ImageIcon(backup)));
            frame.getContentPane().add(panel);
            frame.setVisible(true);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setResizable(false);

        }

        private static NDList buildUnetInput(NDArray input, NDArray timestep, NDArray latents) {
            input.setName("encoder_hidden_states");
            NDList list = new NDList();
            list.add(latents);
            list.add(timestep);
            list.add(input);
            return list;
        }

        private static NDList SDTextEncoder(NDList input)
                throws ModelNotFoundException, MalformedModelException, IOException,
                TranslateException {
            Criteria<NDList, NDList> criteria =
                    Criteria.builder()
                            .setTypes(NDList.class, NDList.class)
                            .optModelPath(new File(model_path_text_encoder_model_gpu0).toPath())
                            .optModelName("text_encoder_model_gpu0")
                            .optEngine(engine.getEngineName())
//                        .optOption("mapLocation", "true")
                            .optDevice(Device.gpu())
                            .optProgress(new ProgressBar())
                            .optTranslator(new NoopTranslator())
                            .build();

            ZooModel<NDList, NDList> model = criteria.loadModel();
            Predictor<NDList, NDList> predictor = model.newPredictor();
            NDList output = predictor.predict(input);
            model.close();
            return output;
        }

        private static Predictor<NDList, NDList> SDUNetPredictor()
                throws ModelNotFoundException, MalformedModelException, IOException {
            Criteria<NDList, NDList> criteria =
                    Criteria.builder()
                            .setTypes(NDList.class, NDList.class)
                            .optModelPath(new File(model_path_unet_traced_model_gpu0).toPath())
                            .optModelName("unet_traced_model_gpu0")
                            .optEngine(engine.getEngineName())
//                        .optOption("mapLocation", "true")
                            .optDevice(Device.gpu())
                            .optProgress(new ProgressBar())
                            .optTranslator(new NoopTranslator())
                            .build();

            ZooModel<NDList, NDList> model = criteria.loadModel();
            return model.newPredictor();
        }

        private static NDArray SDEncoder(Image input)
                throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {
            Criteria<Image, NDArray> criteria =
                    Criteria.builder()
                            .setTypes(Image.class, NDArray.class)
                            .optModelPath(new File(model_path_vae_encoder_model_gpu0).toPath())
                            .optModelName("vae_encoder_model_gpu0")
                            .optEngine(engine.getEngineName())
//                        .optOption("mapLocation", "true")
                            .optDevice(Device.gpu())
                            .optTranslator(new EncoderTranslator())
                            .optProgress(new ProgressBar())
                            .build();

            ZooModel<Image, NDArray> model = criteria.loadModel();
            Predictor<Image, NDArray> predictor = model.newPredictor();
            NDArray output = predictor.predict(input);
            model.close();
            return output;
        }

        private static NDList SDDecoder(NDList input)
                throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {
            Criteria<NDList, NDList> criteria =
                    Criteria.builder()
                            .setTypes(NDList.class, NDList.class)
                            .optModelPath(new File(model_path_vae_decode_model_gpu0).toPath())
                            .optModelName("vae_decode_model_gpu0")
                            .optEngine(engine.getEngineName())
//                        .optOption("mapLocation", "true")
                            .optDevice(Device.gpu())
                            .optTranslator(new NoopTranslator())
                            .optProgress(new ProgressBar())
                            .build();

            ZooModel<NDList, NDList> model = criteria.loadModel();
            Predictor<NDList, NDList> predictor = model.newPredictor();
            NDList output = predictor.predict(input);
            predictor.close();
            return output;
        }



        private static NDList SDTextTokenizer(String prompt) {
            List<String> tokens = tokenizer.tokenize(prompt);
            int[][] tokenValues = new int[1][MAX_LENGTH];
            ObjectMapper mapper = new ObjectMapper();
            File fileObj = new File(vocab_path); //vocab_dictionary.json full_vocab.json
            try {
                Map<String, Integer> mapObj =
                        mapper.readValue(fileObj, new TypeReference<Map<String, Integer>>() {
                        });
                int counter = 0;
                for (String token : tokens) {
                    if (mapObj.get(token) != null) {
                        tokenValues[0][counter] = mapObj.get(token);
                    } else {
                        tokenValues[0][counter] = UNKNOWN_TOKEN;
                    }
                    counter++;
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            NDArray ndArray = manager.create(tokenValues);
            return new NDList(ndArray);
        }
    }


    public static void main(String[] args) throws ModelException, TranslateException, IOException {


        // 模型地址
        String model1 = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\stable_diffusion_img2img_djl_gpu\\text_encoder_model_gpu0.pt";

        String model2 = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\stable_diffusion_img2img_djl_gpu\\unet_traced_model_gpu0.pt";

        String model3 = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\stable_diffusion_img2img_djl_gpu\\vae_decode_model_gpu0.pt";

        String model4 = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\stable_diffusion_img2img_djl_gpu\\vae_encoder_model_gpu0.pt";


        // 提示编码的token
        String token = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\stable_diffusion_img2img_djl_gpu\\tokenizer.json";

        // vocab
        String vocab = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\stable_diffusion_img2img_djl_gpu\\vocab_dictionary.json";

        // 图片
        String img = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\stable_diffusion_img2img_djl_gpu\\sketch-mountains-input.jpg";

        // 正面和负面的提示
        String prompt = "A fantasy landscape, trending on artstation";
        String negative_prompt = "";


        // 图生图
        Img2ImgStableDiffusionGPU.main(
                model1,
                model2,
                model3,
                model4,
                token, // 字库
                vocab, // 字库
                img,    // 原始图片
                prompt,     // 提示
                negative_prompt    // 反面提示
        );


    }



}
